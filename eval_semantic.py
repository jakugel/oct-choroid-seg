import data_generator
import eval_helper
import numpy as np
from keras.utils import to_categorical
from keras import backend as K
import image_database as image_db
import time
import augmentation as aug
import graph_search
from keras.models import Model
import evaluation_output as eoutput


def evaluate_single_images(eval_params, imdb):
    # pass images to network one at a time
    eval_outputs = []

    for ind in imdb.image_range:
        eval_output = eoutput.EvaluationOutput()

        cur_raw_image = imdb.get_image(ind)
        cur_label = imdb.get_label(ind)
        cur_image_name = imdb.get_image_name(ind)
        cur_seg = imdb.get_seg(ind)

        eval_output.raw_image = cur_raw_image
        eval_output.raw_label = cur_label
        eval_output.image_name = cur_image_name
        eval_output.raw_seg = cur_seg

        if eval_params.verbosity >= 2:
            print("Evaluating image number: " + str(ind + 1) + " (" + cur_image_name + ")...")

        if eval_params.save_params.disable is False:
            if eval_helper.check_exists(eval_params.save_foldername, cur_image_name):
                # if the file for this image exists then we have already begun this at some point
                print("File already exists")
            else:
                eval_helper.save_initial_attributes(eval_params, cur_image_name)

            status = eval_helper.get_complete_status(eval_params.save_foldername, cur_image_name,
                                                     eval_params.boundaries)
        else:
            status = 'none'

        if status == 'none' and (eval_params.eval_mode == 'both' or eval_params.eval_mode == 'network'):
            # PERFORM STEP 1: evaluate/predict patches with network

            if eval_params.verbosity >= 2:
                print("Augmenting data using augmentation: " + eval_params.aug_desc + "...")

            aug_fn = eval_params.aug_fn_arg[0]
            aug_arg = eval_params.aug_fn_arg[1]

            # augment raw full sized image and label
            augment_image, augment_label, augment_seg, _, augment_time = \
                aug_fn(cur_raw_image, cur_label, cur_seg, aug_arg, sample_ind=ind, set=imdb.set)

            eval_output.aug_image = augment_image
            eval_output.aug_label = augment_label
            eval_output.aug_seg = augment_seg

            if eval_params.verbosity >= 2:
                print("Running network predictions...")

            images = np.expand_dims(augment_image, axis=0)
            labels = np.expand_dims(augment_label, axis=0)
            single_image_imdb = image_db.ImageDatabase(images=images,
                                                       labels=labels)

            # use a generator to supply data to model (predict_generator)

            start_gen_time = time.time()
            gen = data_generator.DataGenerator(single_image_imdb, eval_params.batch_size,
                                               aug_fn_args=[],
                                               aug_mode='none', aug_probs=[], aug_fly=False, shuffle=False,
                                               transpose=eval_params.transpose,
                                               normalise=eval_params.normalise_input)
            end_gen_time = time.time()
            gen_time = end_gen_time - start_gen_time

            start_predict_time = time.time()

            if not eval_params.ensemble:
                predicted_labels = eval_params.loaded_model.predict_generator(gen,
                                                                              verbose=eval_params.predict_verbosity)
            else:
                predicted_labels = []

                for i in range(len(eval_params.loaded_models)):
                    predicted_labels.append(eval_params.loaded_models[i].predict_generator(gen,
                                                                                         verbose=eval_params.predict_verbosity))

            end_predict_time = time.time()

            predict_time = end_predict_time - start_predict_time

            if eval_params.save_params.activations is True:
                if not eval_params.ensemble:
                    if eval_params.save_params.act_layers is None:
                        layer_outputs = [layer.output for layer in eval_params.loaded_model.layers[1:len(eval_params.loaded_model.layers)]]
                    else:
                        layer_outputs = [layer.output for layer in eval_params.save_params.act_layers]

                    activation_model = Model(inputs=eval_params.loaded_model.input,
                                             outputs=layer_outputs)
                    # Creates a model that will return these outputs, given the model input

                    if eval_params.normalise_input:
                        images_norm = images / 255
                    else:
                        images_norm = images

                    activations = activation_model.predict(images_norm)
                else:
                    layer_outputs = []
                    activations = []

                    # TODO: implement write handling for ensemble activations
                    for i in range(len(eval_params.loaded_models)):
                        layer_outputs.append([layer.output for layer in
                                         eval_params.loaded_models[i].layers[1:len(eval_params.loaded_models[i].layers)]])

                        activation_model = Model(inputs=eval_params.loaded_models[i].input,
                                                 outputs=layer_outputs[i])
                        # Creates a model that will return these outputs, given the model input

                        if eval_params.normalise_input:
                            images_norm = images / 255
                        else:
                            images_norm = images

                        activations.append(activation_model.predict(images_norm))
            else:
                activations = None
                layer_outputs = None

            if eval_params.verbosity >= 2:
                print("Converting predictions to boundary maps...")

            if not eval_params.ensemble:
                if eval_params.transpose is True:
                    predicted_labels = np.transpose(predicted_labels, axes=(0, 2, 1, 3))

                # convert predictions to usable boundary probability maps

                start_convert_time = time.time()

                [comb_area_map, area_maps] = eval_helper.perform_argmax(predicted_labels, ensemble=False, bin=eval_params.binarize)
            else:
                if eval_params.transpose is True:
                    for i in range(len(eval_params.loaded_models)):
                        predicted_labels[i] = np.transpose(predicted_labels[i], axes=(0, 2, 1, 3))

                # convert predictions to usable boundary probability maps

                start_convert_time = time.time()

                [comb_area_map_sep, area_maps_sep] = eval_helper.perform_argmax(predicted_labels, ensemble=True, bin=eval_params.binarize)

                # ensemble using majority voting scheme
                [comb_area_map, area_maps] = eval_helper.perform_ensemble(area_maps_sep)

                print(area_maps.shape)

                if eval_params.binarize_after is True:
                    num_maps = area_maps.shape[1]

                    if eval_params.use_thresh:
                        area_maps[:, 1][area_maps[:, 1] >= eval_params.thresh] = 1
                        area_maps[:, 1][area_maps[:, 1] < eval_params.thresh] = 0

                        area_maps[:, 0][area_maps[:, 0] < eval_params.thresh] = 1
                        area_maps[:, 0][area_maps[:, 0] >= eval_params.thresh] = 0
                        area_maps = np.argmax(area_maps, axis=1)
                    else:
                        area_maps = np.argmax(area_maps, axis=1)

                    area_maps = to_categorical(area_maps, num_maps)

                    area_maps = np.transpose(area_maps, axes=(0, 3, 1, 2))

                print(area_maps.shape)

            eval_output.comb_area_map = comb_area_map
            eval_output.area_maps = area_maps

            if eval_params.boundaries is False or eval_params.save_params.boundary_maps is False:
                boundary_maps = None
            else:
                if eval_params.vertical_graph_search is False:
                    boundary_maps = convert_predictions_to_maps_semantic(np.array(area_maps), bg_ilm=eval_params.bg_ilm, bg_csi=eval_params.bg_csi)
                elif eval_params.vertical_graph_search is True:
                    boundary_maps = convert_predictions_to_maps_semantic_vertical(np.array(area_maps), bg_ilm=eval_params.bg_ilm, bg_csi=eval_params.bg_csi)
                elif eval_params.vertical_graph_search == "ilm_vertical":
                    ilm_map = np.expand_dims(convert_predictions_to_maps_semantic_vertical(np.array(area_maps), bg_ilm=eval_params.bg_ilm, bg_csi=eval_params.bg_csi)[0, 0], axis=0)
                    other_maps = convert_predictions_to_maps_semantic(np.array(area_maps), bg_ilm=eval_params.bg_ilm, bg_csi=eval_params.bg_csi)[0][1:]

                    boundary_maps = np.expand_dims(np.concatenate([ilm_map, other_maps], axis=0), axis=0)

            eval_output.boundary_maps = boundary_maps

            end_convert_time = time.time()
            convert_time = end_convert_time - start_convert_time

            if eval_params.dice_errors is True:
                dices = eval_helper.calc_dice(eval_params, area_maps, labels)
            else:
                dices = None

            area_maps = np.squeeze(area_maps)
            comb_area_map = np.squeeze(comb_area_map)
            boundary_maps = np.squeeze(boundary_maps)

            # save data to files
            if eval_params.save_params.disable is False:
                eval_helper.intermediate_save_semantic(eval_params, imdb, cur_image_name, boundary_maps,
                                                       predict_time, augment_time, gen_time, augment_image,
                                                       augment_label, augment_seg, cur_raw_image, cur_label,
                                                       cur_seg, area_maps, comb_area_map, dices, convert_time,
                                                       activations, layer_outputs)
        if eval_params.save_params.disable is False:
            status = eval_helper.get_complete_status(eval_params.save_foldername, cur_image_name,
                                                     eval_params.boundaries)
        else:
            status = 'predict'

        if status == 'predict' and eval_params.boundaries is True and \
                (eval_params.eval_mode == 'both' or eval_params.eval_mode == 'gs'):
            cur_image_name = imdb.get_image_name(ind)
            cur_seg = imdb.get_seg(ind)
            cur_raw_image = imdb.get_image(ind)
            cur_label = imdb.get_label(ind)

            aug_fn = eval_params.aug_fn_arg[0]
            aug_arg = eval_params.aug_fn_arg[1]

            # augment raw full sized image and label
            augment_image, augment_label, augment_seg, _, _ = \
                aug_fn(cur_raw_image, cur_label, cur_seg, aug_arg, sample_ind=ind, set=imdb.set)

            if eval_params.save_params.disable is False and eval_params.save_params.boundary_maps is True:
                boundary_maps = eval_helper.load_dataset_extra(eval_params, cur_image_name, "boundary_maps")
                if eval_params.dice_errors is True:
                    dices = eval_helper.load_dataset(eval_params, cur_image_name, "dices")
                else:
                    dices = None

            # PERFORM STEP 2: segment probability maps using graph search
            eval_output = eval_helper.eval_second_step(eval_params, boundary_maps,
                                         augment_seg, cur_image_name, augment_image, augment_label,
                                                       imdb, dices, eval_output)
        elif eval_params.boundaries is False:
            if eval_params.save_params.disable is False and eval_params.save_params.attributes is True:
                eval_helper.save_final_attributes(eval_params, cur_image_name, graph_time=None)

        if eval_params.save_params.disable is False and eval_params.save_params.temp_extra is True:
            eval_helper.delete_loadsaveextra_file(eval_params, cur_image_name)

        if eval_params.verbosity >= 2:
            print("DONE image number: " + str(ind + 1) + " (" + cur_image_name + ")...")
            print("______________________________")

    return eval_outputs


# TODO implement ensemble capabilities
def evaluate_all_images(eval_params, imdb):
    # pass all images to network at once
    if eval_params.verbosity >= 2:
        print("Evaluating images...")

    if eval_helper.check_allimages_exists(eval_params.save_foldername):
        print("All images file already_exists")
    else:
        eval_helper.save_initial_attributes_all_images(eval_params)

    status_all = eval_helper.get_complete_status_allimages(eval_params.save_foldername)

    if status_all == 'none' and (eval_params.eval_mode == 'both' or eval_params.eval_mode == 'network'):
        # PERFORM STEP 1: evaluate/predict images with network

        if eval_params.verbosity >= 2:
            print("Augmenting data using augmentation: " + eval_params.aug_desc + "...")

        # augment raw full sized image and label
        augment_images, augment_labels, augment_segs, _, augment_time = \
            aug.augment_dataset(imdb.get_images_range(), imdb.get_labels_range(), imdb.get_segs_range(),
                                eval_params.aug_fn_arg)

        if eval_params.verbosity >= 2:
            print("Running network predictions...")

        augmented_images_imdb = image_db.ImageDatabase(images=augment_images,
                                                       labels=augment_labels)

        # use a generator to supply data to model (predict_generator)
        start_gen_time = time.time()
        gen = data_generator.DataGenerator(augmented_images_imdb, eval_params.batch_size,
                                           aug_fn_args=[],
                                           aug_mode='none', aug_probs=[], aug_fly=False, shuffle=False,
                                           transpose=eval_params.transpose,
                                           normalise=eval_params.normalise_input)
        end_gen_time = time.time()
        gen_time = end_gen_time - start_gen_time

        start_predict_time = time.time()
        predicted_labels = eval_params.loaded_model.predict_generator(gen, verbose=eval_params.predict_verbosity)
        end_predict_time = time.time()
        predict_time = end_predict_time - start_predict_time

        if eval_params.transpose is True:
            predicted_labels = np.transpose(predicted_labels, axes=(0, 2, 1, 3))

        # convert predictions to usable boundary probability maps
        if eval_params.verbosity >= 2:
            print("Converting predictions to boundary maps...")

        start_convert_time = time.time()
        [comb_area_map, area_maps] = eval_helper.perform_argmax(predicted_labels)

        if eval_params.save_params.boundary_maps is False:
            boundary_maps = None
        else:
            if eval_params.vertical_graph_search is False:
                boundary_maps = convert_predictions_to_maps_semantic(np.array(area_maps), bg_ilm=eval_params.bg_ilm,
                                                                     bg_csi=eval_params.bg_csi)
            else:
                boundary_maps = convert_predictions_to_maps_semantic_vertical(np.array(area_maps),
                                                                              bg_ilm=eval_params.bg_ilm,
                                                                              bg_csi=eval_params.bg_csi)

        end_convert_time = time.time()
        convert_time = end_convert_time - start_convert_time

        if eval_params.dice_errors is True:
            dices = eval_helper.calc_dice(eval_params, area_maps, augment_labels)
        else:
            dices = None

        # save data to file
        eval_helper.all_maps_save(eval_params, imdb, boundary_maps,
                                  predict_time, augment_time, gen_time, dices)

    status_all = eval_helper.get_complete_status_allimages(eval_params.save_foldername)
    if status_all == 'predict' and eval_params.boundaries is True and \
            (eval_params.eval_mode == 'both' or eval_params.eval_mode == 'gs'):
        [all_boundary_maps, allimages_file] = \
            eval_helper.load_dataset_all_images_nonram(eval_params, "boundary_maps")
        if eval_params.dice_errors is True:
            [all_dices, results_file] = eval_helper.load_dataset_results_nonram(eval_params, "dices")
        else:
            all_dices = None
            results_file = None

        for ind in imdb.image_range:
            cur_image_name = imdb.get_image_name(ind)
            if eval_helper.check_exists(eval_params.save_foldername, cur_image_name):
                # if the file for this image exists then we have already begun this at some point
                print("File: " + cur_image_name + " already exists")
            else:
                eval_helper.save_initial_attributes(eval_params, cur_image_name)

            status = eval_helper.get_complete_status(eval_params.save_foldername, cur_image_name,
                                                     eval_params.boundaries)
            if status != 'graph':
                cur_seg = imdb.get_seg(ind)
                cur_raw_image = imdb.get_image(ind)
                cur_label = imdb.get_label(ind)

                aug_fn = eval_params.aug_fn_arg[0]
                aug_arg = eval_params.aug_fn_arg[1]

                # augment raw full sized image and label
                augment_image, augment_label, augment_seg, _, _ = \
                    aug_fn(cur_raw_image, cur_label, cur_seg, aug_arg)

                boundary_maps = all_boundary_maps[ind]

                if eval_params.dice_errors is True:
                    dices = all_dices[ind]
                else:
                    dices = None

                # PERFORM STEP 2: segment using graph search

                eval_helper.eval_second_step(eval_params, boundary_maps, augment_seg, cur_image_name,
                                             augment_image, imdb, dices)

                if eval_params.verbosity >= 2:
                    print("DONE image number: " + str(ind + 1) + " (" + cur_image_name + ")...")
                    print("______________________________")

        allimages_file.close()

        if results_file is not None:
            results_file.close()

    if eval_params.save_params.temp_extra is True:
        eval_helper.delete_allimages_file(eval_params)


def evaluate_semantic_network(eval_params, imdb):
    if eval_params.comb_pred is False:
        eval_outputs = evaluate_single_images(eval_params, imdb)
    else:
        evaluate_all_images(eval_params, imdb)

    if eval_params.recalc_errors is True:
        # recalculate errors from saved delineations
        for ind in imdb.image_range:
            cur_image_name = imdb.get_image_name(ind)
            cur_seg = imdb.get_seg(ind)
            delineations = eval_helper.load_dataset(eval_params, cur_image_name, 'delineations')

            num_boundaries = delineations.shape[0]
            width = delineations.shape[1]

            errors = np.zeros((num_boundaries, width), dtype='float64')

            for boundary in range(num_boundaries):
                errors[boundary, :] = graph_search.calc_errors(delineations[boundary, :], cur_seg[boundary, :])

            overall_errors = graph_search.calculate_overall_errors(errors, eval_params.col_error_range)

            if eval_params.verbosity == 3:
                eval_helper.print_error_summary(overall_errors, imdb, cur_image_name)

            [mean_abs_err, mean_err, abs_err_sd, err_sd] = overall_errors

            if eval_params.save_params.errors is True:
                eval_helper.save_dataset(eval_params, cur_image_name, "errors", 'float64', errors)
                eval_helper.save_dataset(eval_params, cur_image_name, "mean_abs_err", 'float64', mean_abs_err)
                eval_helper.save_dataset(eval_params, cur_image_name, "mean_err", 'float64', mean_err)
                eval_helper.save_dataset(eval_params, cur_image_name, "abs_err_sd", 'float64', abs_err_sd)
                eval_helper.save_dataset(eval_params, cur_image_name, "err_sd", 'float64', err_sd)

    return eval_outputs


def convert_predictions_to_maps_semantic(categorical_pred, bg_ilm=True, bg_csi=False):
    num_samples = categorical_pred.shape[0]
    img_width = categorical_pred.shape[2]
    img_height = categorical_pred.shape[3]
    num_maps = categorical_pred.shape[1]

    boundary_maps = np.zeros((num_samples, num_maps - 1, img_width, img_height), dtype='uint8')

    for sample_ind in range(num_samples):
        for map_ind in range(1, num_maps):  # don't care about boundary for top region

            if (map_ind == 1 and bg_ilm is True) or (map_ind == num_maps-1 and bg_csi is True):
                cur_map = categorical_pred[sample_ind, map_ind - 1, :, :]

                grad_map = np.gradient(cur_map, axis=1)

                grad_map = -grad_map

                grad_map[grad_map < 0] = 0

                grad_map *= 2  # scale map to between 0 and 1

                rolled_grad = np.roll(grad_map, -1, axis=1)

                grad_map -= rolled_grad
                grad_map[grad_map < 0] = 0
                boundary_maps[sample_ind, map_ind - 1, :, :] = eval_helper.convert_maps_uint8(grad_map)
            else:
                cur_map = categorical_pred[sample_ind, map_ind, :, :]

                grad_map = np.gradient(cur_map, axis=1)

                grad_map[grad_map < 0] = 0

                grad_map *= 2  # scale map to between 0 and 1

                rolled_grad = np.roll(grad_map, -1, axis=1)

                grad_map -= rolled_grad
                grad_map[grad_map < 0] = 0
                boundary_maps[sample_ind, map_ind - 1, :, :] = eval_helper.convert_maps_uint8(grad_map)

    return boundary_maps


def convert_predictions_to_maps_semantic_vertical(categorical_pred, bg_ilm=True, bg_csi=False):
    num_samples = categorical_pred.shape[0]
    img_width = categorical_pred.shape[2]
    img_height = categorical_pred.shape[3]
    num_maps = categorical_pred.shape[1]

    boundary_maps = np.zeros((num_samples, num_maps - 1, img_width, img_height), dtype='uint8')

    for sample_ind in range(num_samples):
        for map_ind in range(1, num_maps):  # don't care about boundary for top region

            if map_ind == 1 and bg_ilm is True:
                cur_map = categorical_pred[sample_ind, map_ind - 1, :, :]

                grad_map = np.gradient(cur_map, axis=(0, 1))

                grad_map[1] = -grad_map[1]

                grad_map[1][grad_map[1] < 0] = 0

                grad_map[1] *= 2  # scale map to between 0 and 1

                grad_map[0] = np.abs(grad_map[0])

                grad_map[0] *= 2

                rolled_grad = np.roll(grad_map[1], -1, axis=1)

                grad_map[1] -= rolled_grad

                grad_map[1][grad_map[1] < 0] = 0

                grad_map = np.add(grad_map[0], grad_map[1])

                boundary_maps[sample_ind, map_ind - 1, :, :] = eval_helper.convert_maps_uint8(grad_map)
            elif map_ind == num_maps - 1 and bg_csi is True:
                cur_map = categorical_pred[sample_ind, map_ind - 1, :, :]

                grad_map = np.gradient(cur_map, axis=(0, 1))

                grad_map[1] = -grad_map[1]

                grad_map[1][grad_map[1] < 0] = 0

                grad_map[1] *= 2  # scale map to between 0 and 1

                grad_map[0] = np.abs(grad_map[0])

                grad_map[0] *= 2

                rolled_grad = np.roll(grad_map[1], -1, axis=1)

                grad_map[1] -= rolled_grad

                grad_map[1][grad_map[1] < 0] = 0

                grad_map = np.add(grad_map[0], grad_map[1])

                boundary_maps[sample_ind, map_ind - 1, :, :] = eval_helper.convert_maps_uint8(grad_map)
            else:
                cur_map = categorical_pred[sample_ind, map_ind, :, :]

                grad_map = np.gradient(cur_map, axis=(0, 1))

                grad_map[1][grad_map[1] < 0] = 0

                grad_map[1] *= 2  # scale map to between 0 and 1

                grad_map[0] = np.abs(grad_map[0])

                grad_map[0] *= 2

                rolled_grad = np.roll(grad_map[1], -1, axis=1)

                grad_map[1] -= rolled_grad
                grad_map[1][grad_map[1] < 0] = 0

                grad_map = np.add(grad_map[0], grad_map[1])

                boundary_maps[sample_ind, map_ind - 1, :, :] = eval_helper.convert_maps_uint8(grad_map)

    return boundary_maps
