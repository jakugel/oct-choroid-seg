import time
import dataset_construction as datacon
import data_generator
import eval_helper
import numpy as np
import image_database as image_db
import evaluation_output as eoutput


def evaluate_patch_based_network(eval_params, imdb):
    # patches need to be constructed and passed to the generator for one image at a time
    if eval_params.save_params.output_var is True:
        eval_outputs = []
    else:
        eval_outputs = None

    for ind in imdb.image_range:
        if eval_params.save_params.output_var is True:
            eval_output = eoutput.EvaluationOutput()
        else:
            eval_output = None

        cur_full_image = imdb.get_image(ind)
        cur_patch_labels = imdb.get_patch_label(ind)
        cur_image_name = imdb.get_image_name(ind)
        cur_seg = imdb.get_seg(ind)

        if eval_params.save_params.output_var is True:
            eval_output.raw_image = cur_full_image
            eval_output.raw_label = cur_patch_labels
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
                                                 boundaries=eval_params.boundaries)
        else:
            status = 'none'

        if status == 'none' and (eval_params.eval_mode == 'both' or eval_params.eval_mode == 'network'):
            # PERFORM STEP 1: evaluate/predict patches with network

            if eval_params.verbosity >= 2:
                print("Augmenting data using augmentation: " + eval_params.aug_desc + "...")

            aug_fn = eval_params.aug_fn_arg[0]
            aug_arg = eval_params.aug_fn_arg[1]

            # augment raw full sized image and label
            augment_image, augment_patch_labels, augment_seg, _, augment_time = \
                aug_fn(cur_full_image, cur_patch_labels, cur_seg, aug_arg)

            if eval_params.save_params.output_var is True:
                eval_output.aug_image = augment_image
                eval_output.aug_label = augment_patch_labels
                eval_output.aug_seg = augment_seg

            if eval_params.verbosity >= 2:
                print("Constructing patches...")

            # construct patches
            input_patches, input_labels, patch_time = \
                datacon.construct_patches_whole_image(augment_image, augment_patch_labels,
                                                             eval_params.patch_size)

            patch_imdb = image_db.ImageDatabase(images=input_patches, labels=input_labels)

            if eval_params.verbosity >= 2:
                print("Running network predictions...")

            # use a generator to supply data to model (predict_generator)
            # we have already previously augmented to image so need to augment the individual patches

            start_predict_time = time.time()

            import keras
            class CustomCallback(keras.callbacks.Callback):
                def __init__(self, gen):
                    keras.callbacks.Callback.__init__(self)
                    self.gen = gen

                def on_predict_begin(self, logs=None):
                    self.gen.batch_gen.batch_counter = 0
                    self.gen.batch_gen.full_counter = 0
                    self.gen.batch_gen.aug_counter = 0

            if not eval_params.ensemble:
                start_gen_time = time.time()
                gen = data_generator.DataGenerator(patch_imdb, eval_params.batch_size, aug_fn_args=[],
                                                   aug_mode='none', aug_probs=[], aug_fly=False, shuffle=False,
                                                   normalise=eval_params.normalise_input,
                                                   transpose=eval_params.transpose)
                end_gen_time = time.time()
                gen_time = end_gen_time - start_gen_time

                cust_callback = CustomCallback(gen)
                predicted_labels = eval_params.loaded_model.predict_generator(gen, verbose=eval_params.predict_verbosity, callbacks=[cust_callback])
                print(predicted_labels.shape)
            else:
                predicted_labels = []

                for i in range(len(eval_params.loaded_models)):
                    start_gen_time = time.time()
                    gen = data_generator.DataGenerator(patch_imdb, eval_params.batch_size, aug_fn_args=[],
                                                       aug_mode='none', aug_probs=[], aug_fly=False, shuffle=False,
                                                       normalise=eval_params.normalise_input,
                                                       transpose=eval_params.transpose)
                    end_gen_time = time.time()
                    gen_time = end_gen_time - start_gen_time

                    predicted_labels.append(eval_params.loaded_models[i].predict_generator(gen, verbose=eval_params.predict_verbosity))

            end_predict_time = time.time()
            predict_time = end_predict_time - start_predict_time

            if eval_params.verbosity >= 2:
                print("Converting predictions to boundary maps...")

            # convert predictions to usable probability maps
            start_convert_time = time.time()

            if eval_params.boundaries is True and eval_params.save_params.boundary_maps is True:

                if not eval_params.ensemble:

                    prob_maps = convert_predictions_to_maps_patch_based(predicted_labels, imdb.image_width,
                                                                        imdb.image_height)
                else:
                    prob_maps = []

                    for i in range(len(predicted_labels)):
                        prob_maps.append(np.expand_dims(convert_predictions_to_maps_patch_based(predicted_labels[i], imdb.image_width,
                                                                                 imdb.image_height), axis=0))

                    prob_maps = eval_helper.perform_ensemble_patch(prob_maps)
            else:
                prob_maps = None

            if eval_params.save_params.output_var is True:
                eval_output.boundary_maps = prob_maps

            end_convert_time = time.time()
            convert_time = end_convert_time - start_convert_time

            # save data to file
            if eval_params.save_params.disable is False:
                eval_helper.intermediate_save_patch_based(eval_params, imdb, cur_image_name, prob_maps, predict_time,
                                                          augment_time, gen_time, convert_time, patch_time, augment_image,
                                                          augment_patch_labels, augment_seg, cur_full_image,
                                                          cur_patch_labels, cur_seg)

        if eval_params.save_params.disable is False:
            status = eval_helper.get_complete_status(eval_params.save_foldername, cur_image_name,
                                                     boundaries=eval_params.boundaries)
        else:
            status = 'predict'

        if status == 'predict' and eval_params.boundaries is True and \
                (eval_params.eval_mode == 'both' or eval_params.eval_mode == 'gs'):
            aug_fn = eval_params.aug_fn_arg[0]
            aug_arg = eval_params.aug_fn_arg[1]

            # augment raw full sized image and label
            augment_image, augment_patch_labels, augment_seg, _, augment_time = \
                aug_fn(cur_full_image, cur_patch_labels, cur_seg, aug_arg)

            # load probability maps from previous step
            if eval_params.save_params.disable is False and eval_params.save_params.boundary_maps is True:
                prob_maps = eval_helper.load_dataset_extra(eval_params, cur_image_name, "boundary_maps")

            # PERFORM STEP 2: segment probability maps using graph search
            boundary_maps = get_boundary_maps_only(imdb, prob_maps)
            eval_helper.eval_second_step(eval_params, boundary_maps, augment_seg, cur_image_name, augment_image, augment_patch_labels, imdb,
                                         dices=None, eval_output=eval_output)
        elif eval_params.boundaries is False:
            if eval_params.save_params.disable is False and eval_params.save_params.attributes is True:
                eval_helper.save_final_attributes(eval_params, cur_image_name, graph_time=None)

        if eval_params.save_params.disable is False and eval_params.save_params.temp_extra is True:
            eval_helper.delete_loadsaveextra_file(eval_params, cur_image_name)

        if eval_params.verbosity >= 2:
            print("DONE image number: " + str(ind + 1) + " (" + cur_image_name + ")...")
            print("______________________________")

    return eval_outputs


def get_boundary_maps_only(imdb, prob_maps):
    # make a new numpy array of maps excluding any maps that are not for boundaries

    num_boundaries = len(imdb.get_boundary_names())
    boundary_maps = np.zeros((num_boundaries, prob_maps.shape[1], prob_maps.shape[2]))

    num_classes = prob_maps.shape[0]

    cur_boundary_ind = 0
    for class_ind in range(num_classes):
        if imdb.get_boundary_name(cur_boundary_ind) == imdb.get_patch_class_name(class_ind):
            boundary_maps[cur_boundary_ind] = prob_maps[class_ind]
            cur_boundary_ind += 1
        else:
            # this is not a boundary map
            pass

    return boundary_maps


def convert_predictions_to_maps_patch_based(predictions, img_width, img_height):
    num_maps = predictions.shape[1]

    prob_maps = np.zeros((num_maps, img_width, img_height))

    # convert prob maps into correct shape
    for row in range(img_height):
        for col in range(img_width):
            prob_maps[:, col, row] = predictions[row * img_width + col, :]

    prob_maps = eval_helper.convert_maps_uint8(prob_maps)   # convert maps to uint8 to save space

    return prob_maps
