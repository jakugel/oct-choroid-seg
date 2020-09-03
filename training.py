import tensorflow as tf
import data_generator as data_gen
from keras.callbacks import ModelCheckpoint, TensorBoard
import h5py
import os
import numpy as np
import common
import training_callbacks
import parameters


def save_config_file(save_foldername, model_name, timestamp, train_params, train_imdb, val_imdb, opt):
    config_filename = save_foldername + "/config.hdf5"

    config_file = h5py.File(config_filename, 'w')

    config_file.attrs["timestamp"] = np.array(timestamp, dtype='S100')
    config_file.attrs["model_name"] = np.array(model_name, dtype='S1000')
    config_file.attrs["train_imdb"] = np.array(train_imdb.filename, dtype='S100')
    config_file.attrs["val_imdb"] = np.array(val_imdb.filename, dtype='S100')
    config_file.attrs["epochs"] = train_params.epochs
    config_file.attrs["dim_names"] = np.array(train_imdb.dim_names, dtype='S100')
    config_file.attrs["type"] = np.array(train_imdb.type, dtype='S100')
    if train_imdb.type == 'patch':
        config_file.attrs["patch_size"] = np.array((train_imdb.image_width, train_imdb.image_height))

    if train_imdb.dim_inds is None:
        config_file.attrs["train_dim_inds"] = np.array("all", dtype='S100')
    else:
        dim_count = 0
        for dim_ind in train_imdb.dim_inds:
            if dim_ind is not None:
                config_file.attrs["train_dim_ind: " + train_imdb.dim_names[dim_count]] = dim_ind
            dim_count += 1

    if val_imdb.dim_inds is None:
        config_file.attrs["val_dim_inds"] = np.array("all", dtype='S100')
    else:
        dim_count = 0
        for dim_ind in val_imdb.dim_inds:
            if dim_ind is not None:
                config_file.attrs["val_dim_ind: " + val_imdb.dim_names[dim_count]] = dim_ind
            dim_count += 1

    config_file.attrs["metric"] = np.array(train_params.metric_name, dtype='S100')
    config_file.attrs["loss"] = np.array(train_params.loss_name, dtype='S100')
    config_file.attrs["batch_size"] = train_params.batch_size
    config_file.attrs["shuffle"] = train_params.shuffle
    if train_imdb.padding is not None:
        config_file.attrs["padding"] = np.array(train_imdb.padding)

    config_file.attrs["aug_mode"] = np.array(train_params.aug_mode, dtype='S100')
    if train_params.aug_mode != 'none':
        for aug_ind in range(len(train_params.aug_fn_args)):
            aug_fn = train_params.aug_fn_args[aug_ind][0]
            aug_arg = train_params.aug_fn_args[aug_ind][1]

            aug_desc = aug_fn(None, None, None, aug_arg, True)

            if type(aug_arg) is not dict:
                config_file.attrs["aug_" + str(aug_ind + 1)] = np.array(aug_desc, dtype='S1000')
            else:
                config_file.attrs["aug_" + str(aug_ind + 1)] = np.array(aug_fn.__name__, dtype='S100')

                for aug_param_key in aug_arg.keys():
                    val = aug_arg[aug_param_key]
                    if type(val) is int or type(val) is float:
                        config_file.attrs["aug_" + str(aug_ind + 1) + "_param: " + aug_param_key] = np.array(val)
                    elif type(val) is str:
                        config_file.attrs["aug_" + str(aug_ind + 1) + "_param: " + aug_param_key] = np.array(val,
                                                                                                             dtype='S100')
                    elif type(val) is list and (type(val[0]) is int or type(val[0]) is str or type(val[0]) is float):
                        config_file.attrs["aug_" + str(aug_ind + 1) + "_param: " + aug_param_key] = np.array(str(val), dtype='S100')

            if train_params.aug_mode == 'one':
                config_file.attrs["aug_probs"] = \
                    np.array(train_params.aug_probs)

        config_file.attrs["aug_fly"] = train_params.aug_fly
        config_file.attrs["aug_val"] = train_params.aug_val

    config_file.attrs["optimizer"] = np.array(train_params.opt_con.__name__, dtype='S100')

    opt_config = opt.get_config()

    for key in opt_config:
        if type(opt_config[key]) is dict:
            config_file.attrs["opt_param: " + key] = np.string_(str(opt_config[key]))
        else:
            config_file.attrs["opt_param: " + key] = opt_config[key]

    config_file.attrs["normalise"] = np.array(train_params.normalise)

    config_file.attrs["ram_load"] = train_imdb.ram_load


def train_network(train_imdb, val_imdb, train_params):
    with tf.device('/gpu:0'):
        [model, model_name, model_name_short] = train_params.network_model
        optimizer_con = train_params.opt_con
        optimizer_params = train_params.opt_params

        optimizer = optimizer_con(**optimizer_params)

        loss = train_params.loss
        metric = train_params.metric
        epochs = train_params.epochs

        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

        model.summary()

        batch_size = train_params.batch_size
        aug_fn_args = train_params.aug_fn_args
        aug_mode = train_params.aug_mode
        aug_probs = train_params.aug_probs
        aug_fly = train_params.aug_fly
        aug_val = train_params.aug_val
        use_gen = train_params.use_gen
        use_tensorboard = train_params.use_tensorboard
        ram_load = train_imdb.ram_load

        if use_gen is False and ram_load == 0:
            print("Incompatible parameter selection")
            exit(1)
        elif ram_load == 0 and aug_fly is False and aug_mode != 'none':
            print("Incompatible parameter selection")
            exit(1)

        if aug_val is False:
            aug_val_mode = 'none'
            aug_val_fn_args = []
            aug_val_probs = []
            aug_val_fly = False
        else:
            aug_val_mode = aug_mode
            aug_val_fn_args = aug_fn_args
            aug_val_probs = aug_probs
            aug_val_fly = aug_fly

        shuffle = train_params.shuffle
        normalise = train_params.normalise
        monitor = train_params.model_save_monitor

        save_best = train_params.model_save_best

        dataset_name = train_imdb.name

        timestamp = common.get_timestamp()
        save_foldername = parameters.RESULTS_LOCATION + timestamp + " " + model_name_short + " " + dataset_name

        if not os.path.exists(save_foldername):
            os.makedirs(save_foldername)
        else:
            count = 2
            testsave_foldername = parameters.RESULTS_LOCATION + timestamp + " " + str(count) + " " + model_name_short + " " + dataset_name
            while os.path.exists(testsave_foldername):
                count += 1
                testsave_foldername = parameters.RESULTS_LOCATION + timestamp + " " + str(count) + " " + model_name_short + " " + dataset_name

            save_foldername = testsave_foldername
            os.makedirs(save_foldername)

        epoch_model_name = "model_epoch{epoch:02d}.hdf5"

        savemodel = ModelCheckpoint(filepath=save_foldername + "/" + epoch_model_name, save_best_only=save_best,
                                    monitor=monitor[0], mode=monitor[1])

        history = training_callbacks.SaveEpochInfo(save_folder=save_foldername,
                                train_params=train_params,
                                train_imdb=train_imdb)

        if use_tensorboard is True:
            tensorboard = TensorBoard(log_dir=save_foldername + "/tensorboard", write_grads=False, write_graph=False,
                                      write_images=True, histogram_freq=1, batch_size=batch_size)
            callbacks_list = [savemodel, history, tensorboard]
        else:
            callbacks_list = [savemodel, history]

        save_config_file(save_foldername, model_name, timestamp, train_params, train_imdb, val_imdb, optimizer)

        if use_gen is True:
            train_gen = data_gen.DataGenerator(train_imdb, batch_size, aug_fn_args, aug_mode, aug_probs, aug_fly, shuffle,
                                               normalise=normalise, ram_load=ram_load)
            val_gen = data_gen.DataGenerator(val_imdb, batch_size, aug_val_fn_args, aug_val_mode, aug_val_probs,
                                             aug_val_fly, shuffle, normalise=normalise, ram_load=ram_load)

            if train_params.class_weight is None:
                model_info = model.fit_generator(generator=train_gen,
                                                 validation_data=val_gen, epochs=epochs, callbacks=callbacks_list,
                                                 verbose=1)
            else:
                model_info = model.fit_generator(generator=train_gen,
                                                 validation_data=val_gen, epochs=epochs, callbacks=callbacks_list,
                                                 verbose=1, class_weight=train_params.class_weight)
        else:
            x_train = train_imdb.images
            y_train = train_imdb.labels

            x_val = val_imdb.images
            y_val = val_imdb.labels

            if train_params.class_weight is None:
                model_info = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                                       callbacks=callbacks_list, validation_data=[x_val, y_val], shuffle=True)
            else:
                model_info = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                                       callbacks=callbacks_list, validation_data=[x_val, y_val], shuffle=True,
                                       class_weight=train_params.class_weight)
