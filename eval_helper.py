import graph_search
import numpy as np
import time
import evaluation_parameters as eparams
import augmentation as aug
import h5py
from keras.models import load_model
import eval_patch_based
import eval_semantic
import os
import plotting
from matplotlib import cm
import common
import custom_metrics
import custom_losses
import results_collation
import dataset_construction as datacon
import sys
import keras.backend as K
from keras.utils import to_categorical


def resource_path(relative):
    application_path = os.path.abspath(".")
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle, the pyInstaller bootloader
        # extends the sys module by a flag frozen=True and sets the app
        # path into variable _MEIPASS'.
        application_path = sys._MEIPASS
        return os.path.join(application_path, relative)
    else:
        return relative


def evaluate_network(imdb, model_filename, network_foldername, batch_size, save_params, gsgrad=1, eval_mode='both',
                     aug_fn_arg=(aug.no_aug, {}), use_config=True, type='fullsize', patch_size=None,
                     col_error_range=None, transpose=False, normalise_input=True,
                     comb_pred=False, verbosity=3, recalc_errors=False, boundaries=True, boundary_errors=True,
                     dice_errors=True,
                     loaded_model=None, graph_structure=None, trim_maps=False, trim_ref_ind=0, trim_window=(0, 0),
                     save_foldername=None, collate_results=True, flatten_image=False, flatten_ind=0,
                     flatten_poly=False, ensemble=False, loaded_models=None, network_foldernames=None,
                     model_filenames=None, binarize=True, binarize_after=True, vertical_graph_search=False, bg_ilm=True, bg_csi=False, flatten_pred_edges=False,
                     flat_marg=0, use_thresh=False, thresh=0.5):

    if not ensemble:
        if loaded_model is None:
            if eval_mode == 'gs':
                # don't need to load model if we are just doing the graph search
                loaded_model = None
            else:
                custom_objects = dict(list(custom_losses.custom_loss_objects.items()) +
                                      list(custom_metrics.custom_metric_objects.items()))

                loaded_model = load_model(resource_path(network_foldername + model_filename), custom_objects=custom_objects)
    else:
        if loaded_models is None:
            if eval_mode == 'gs':
                # don't need to load models if we are just doing the graph search
                loaded_models = None
            else:
                loaded_models = []

                custom_objects = dict(list(custom_losses.custom_loss_objects.items()) +
                                      list(custom_metrics.custom_metric_objects.items()))

                for i in range(len(network_foldernames)):
                    loaded_models.append(load_model(resource_path(network_foldernames[i] + model_filenames[i]), custom_objects=custom_objects))

    if col_error_range is None:
        col_error_range = range(imdb.image_width)

    if graph_structure is None:
        if vertical_graph_search is False:
            graph_structure = graph_search.create_graph_structure((imdb.image_width, imdb.image_height), max_grad=gsgrad)
        elif vertical_graph_search is True:
            graph_structure = graph_search.create_graph_structure_vertical((imdb.image_width, imdb.image_height),)
        elif vertical_graph_search is "ilm_vertical":
            graph_structure = [graph_search.create_graph_structure_vertical((imdb.image_width, imdb.image_height), ),
                               graph_search.create_graph_structure((imdb.image_width, imdb.image_height),
                                                                   max_grad=gsgrad)]

    if use_config:
        if not ensemble:
            loaded_config = h5py.File(resource_path(network_foldername + "config.hdf5"), 'r')
        else:
            loaded_config = h5py.File(resource_path(network_foldernames[0] + "config.hdf5"), 'r')

        type = np.char.decode(loaded_config.attrs["type"], 'utf-8')

        if type == 'patch':
            patch_size = loaded_config.attrs["patch_size"]
        else:
            patch_size = None

        loaded_config.close()

    eval_params = eparams.EvaluationParameters(loaded_model, model_filename, network_foldername, imdb.filename,
                                               batch_size, graph_structure, col_error_range,
                                               eval_mode=eval_mode, aug_fn_arg=aug_fn_arg, patch_size=patch_size,
                                               save_params=save_params,
                                               verbosity=verbosity, gsgrad=gsgrad, transpose=transpose,
                                               normalise_input=normalise_input, comb_pred=comb_pred,
                                               recalc_errors=recalc_errors, boundaries=boundaries,
                                               trim_maps=trim_maps, trim_ref_ind=trim_ref_ind, trim_window=trim_window, dice_errors=dice_errors,
                                               save_foldername=save_foldername, flatten_image=flatten_image,
                                               flatten_ind=flatten_ind, flatten_poly=flatten_poly, ensemble=ensemble, loaded_models=loaded_models,
                                               model_filenames=model_filenames, network_foldernames=network_foldernames, binarize=binarize, binarize_after=binarize_after, vertical_graph_search=vertical_graph_search,
                                               bg_ilm=bg_ilm, bg_csi=bg_csi, flatten_pred_edges=flatten_pred_edges, flat_marg=flat_marg, use_thresh=use_thresh, thresh=thresh)

    if save_params.disable is False:
        if not os.path.exists(eval_params.save_foldername):
            os.makedirs(eval_params.save_foldername)

    if save_params.disable is False:
        save_eval_config_file(eval_params, imdb)

    if type == 'patch':
        eval_outputs = eval_patch_based.evaluate_patch_based_network(eval_params, imdb)
    else:
        eval_outputs = eval_semantic.evaluate_semantic_network(eval_params, imdb)

    if save_params.disable is False:
        if collate_results is True:
            if eval_mode == 'network' or boundaries is False or boundary_errors is False:
                if type == 'patch':
                    inc_boundary_errors = False
                    inc_dice = False
                else:
                    inc_boundary_errors = False
                    if dice_errors is False:
                        inc_dice = False
                    else:
                        inc_dice = True
            else:
                if type == 'patch':
                    inc_boundary_errors = True
                    inc_dice = False
                else:
                    inc_boundary_errors = True
                    if dice_errors is False:
                        inc_dice = False
                    else:
                        inc_dice = True

            results_collation.calc_overall_dataset_errors(eval_params.save_foldername, inc_boundary_errors=inc_boundary_errors,
                                                          inc_dice=inc_dice)

    return [eval_params, eval_outputs]


def save_eval_config_file(eval_params, imdb):
    config_file = h5py.File(eval_params.save_foldername + "/config.hdf5", 'w')

    config_file.attrs['ensemble'] = eval_params.ensemble

    if not eval_params.ensemble:
        config_file.attrs["model_filename"] = np.array(eval_params.model_filename, dtype='S100')
    else:
        config_file.attrs["model_filenames"] = np.array(eval_params.model_filenames, dtype='S100')
        config_file.attrs["network_foldernames"] = np.array(eval_params.network_foldernames, dtype='S100')

    config_file.attrs["data_filename"] = np.array(imdb.filename, dtype='S100')

    aug_fn = eval_params.aug_fn_arg[0]
    aug_arg = eval_params.aug_fn_arg[1]

    config_file.attrs["aug"] = np.array(aug_fn.__name__, dtype='S100')

    for aug_param_key in aug_arg.keys():
        val = str(aug_arg[aug_param_key])

        config_file.attrs["aug_param: " + aug_param_key] = np.array(val, dtype='S100')

    if eval_params.patch_size is not None:
        config_file.attrs["patch_size"] = np.array(eval_params.patch_size)

    config_file.attrs["batch_size"] = np.array(eval_params.batch_size)
    config_file.attrs["normalise_input"] = np.array(eval_params.normalise_input)

    config_file.attrs["boundaries"] = np.array(eval_params.boundaries)

    if eval_params.boundaries is True:
        config_file.attrs["gsgrad"] = np.array(eval_params.gsgrad)
        config_file.attrs["error_col_inc_range"] = np.array((eval_params.col_error_range[0],
                                                             eval_params.col_error_range[-1]))

    config_file.close()


def eval_second_step(eval_params, prob_maps,
                              cur_seg, cur_image_name, cur_augment_image, cur_augment_label, imdb, dices, eval_output):
    if eval_params.verbosity >= 2:
        print("Running graph search, segmenting boundary maps...")

    truths = cur_seg
    start_graph_time = time.time()
    delineations, errors, trim_maps = graph_search.segment_maps(prob_maps, truths, eval_params)

    reconstructed_maps = datacon.create_area_mask(cur_augment_image, delineations)

    reconstructed_maps = to_categorical(reconstructed_maps, num_classes=imdb.num_classes)
    reconstructed_maps = np.expand_dims(reconstructed_maps, axis=0)

    if eval_params.dice_errors is True:
        [comb_area_map_recalc, reconstructed_maps] = perform_argmax(reconstructed_maps, ensemble=False)

        recalc_dices = calc_dice(eval_params, reconstructed_maps, np.expand_dims(cur_augment_label, axis=0))

        comb_area_map_recalc = np.squeeze(comb_area_map_recalc)

    if eval_params.flatten_image is True:
        [flattened_image, offsets, flatten_boundary] = datacon.flatten_image_boundary(cur_augment_image, delineations[eval_params.flatten_ind],
                                                         poly=eval_params.flatten_poly)
        if eval_params.save_params.output_var is True:
            eval_output.flattened_image = flattened_image
            eval_output.offsets = offsets
            eval_output.flatten_boundary = flatten_boundary

    if eval_params.save_params.output_var is True:
        eval_output.delineations = delineations
        eval_output.errors = errors
        eval_output.trim_maps = trim_maps

    end_graph_time = time.time()
    graph_time = end_graph_time - start_graph_time

    # TODO code to calculate dice overlap of areas
    overall_errors = graph_search.calculate_overall_errors(errors, eval_params.col_error_range)

    if eval_params.verbosity == 3 and truths is not None:
        print_error_summary(overall_errors, imdb, cur_image_name)

    [mean_abs_err, mean_err, abs_err_sd, err_sd] = overall_errors

    # FINAL SAVE
    if eval_params.save_params.disable is False:
        if eval_params.save_params.delineations is True:
            save_dataset(eval_params, cur_image_name, "delineations", 'uint16', delineations)
        if eval_params.save_params.errors is True and truths is not None:
            save_dataset(eval_params, cur_image_name, "errors", 'float64', errors)
            save_dataset(eval_params, cur_image_name, "mean_abs_err", 'float64', mean_abs_err)
            save_dataset(eval_params, cur_image_name, "mean_err", 'float64', mean_err)
            save_dataset(eval_params, cur_image_name, "abs_err_sd", 'float64', abs_err_sd)
            save_dataset(eval_params, cur_image_name, "err_sd", 'float64', err_sd)
            if dices is not None:
                save_dataset(eval_params, cur_image_name, "dices", 'float64', dices)
                save_dataset(eval_params, cur_image_name, "dices_recon", 'float64', np.squeeze(recalc_dices))
        if eval_params.save_params.comb_area_maps_recalc is True and eval_params.dice_errors is True:
            save_dataset_extra(eval_params, cur_image_name, "comb_area_map_recalc", 'uint8', comb_area_map_recalc)

            if eval_params.save_params.pngimages is True:

                plotting.save_image_plot(comb_area_map_recalc, get_loadsave_path(eval_params.save_foldername, cur_image_name) +
                                         "/comb_area_map_recalc.png",
                                         cmap=plotting.colors.ListedColormap(plotting.region_colours,
                                                                             N=len(np.squeeze(reconstructed_maps))))

        if eval_params.save_params.seg_plot is True:
            if eval_params.save_params.pngimages is True:
                if truths is not None:
                    plotting.save_segmentation_plot(cur_augment_image, cm.gray, get_loadsave_path(eval_params.save_foldername,
                                                                                                  cur_image_name)
                                                    + "/" + "seg_plot.png",
                                                    cur_seg, delineations, column_range=eval_params.col_error_range)
                    plotting.save_segmentation_plot(cur_augment_image, cm.gray, get_loadsave_path(eval_params.save_foldername,
                                                                                                  cur_image_name)
                                                    + "/" + "truth_plot.png",
                                                    cur_seg, predictions=None, column_range=eval_params.col_error_range)

                plotting.save_segmentation_plot(cur_augment_image, cm.gray, get_loadsave_path(eval_params.save_foldername,
                                                                                              cur_image_name)
                                                + "/" + "delin_plot.png",
                                                delineations, predictions=None, column_range=eval_params.col_error_range)

        if eval_params.save_params.indiv_seg_plot is True:
            if eval_params.save_params.pngimages is True:
                for i in range(delineations.shape[0]):
                    if truths is not None:
                        plotting.save_segmentation_plot(cur_augment_image, cm.gray, get_loadsave_path(eval_params.save_foldername,
                                                                                                      cur_image_name)
                                                        + "/" + "seg_plot_" + str(i) + ".png",
                                                        np.expand_dims(cur_seg[i], axis=0), np.expand_dims(delineations[i], axis=0), column_range=eval_params.col_error_range, color="#ffe100", linewidth=2.0)
                        plotting.save_segmentation_plot(cur_augment_image, cm.gray, get_loadsave_path(eval_params.save_foldername,
                                                                                                      cur_image_name)
                                                        + "/" + "truth_plot_" + str(i) + ".png",
                                                        np.expand_dims(cur_seg[i], axis=0), predictions=None, column_range=eval_params.col_error_range, color="#ffe100", linewidth=2.0)

                    plotting.save_segmentation_plot(cur_augment_image, cm.gray, get_loadsave_path(eval_params.save_foldername,
                                                                                                  cur_image_name)
                                                    + "/" + "delin_plot_" + str(i) + ".png",
                                                    np.expand_dims(delineations[i], axis=0), predictions=None, column_range=eval_params.col_error_range, color="#ffe100", linewidth=2.0)

        if eval_params.save_params.pair_seg_plot is True:
            if eval_params.save_params.pngimages is True:
                for i in range(delineations.shape[0] - 1):
                    if truths is not None:
                        plotting.save_segmentation_plot(cur_augment_image, cm.gray, get_loadsave_path(eval_params.save_foldername,
                                                                                                      cur_image_name)
                                                        + "/" + "seg_plot_pair" + str(i) + ".png",
                                                        cur_seg[i:i+2], np.expand_dims(delineations[i], axis=0), column_range=eval_params.col_error_range, color="#ffe100", linewidth=2.0)
                        plotting.save_segmentation_plot(cur_augment_image, cm.gray, get_loadsave_path(eval_params.save_foldername,
                                                                                                      cur_image_name)
                                                        + "/" + "truth_plot_pair" + str(i) + ".png",
                                                        cur_seg[i:i+2], predictions=None, column_range=eval_params.col_error_range, color="#ffe100", linewidth=2.0)

                    plotting.save_segmentation_plot(cur_augment_image, cm.gray, get_loadsave_path(eval_params.save_foldername,
                                                                                                  cur_image_name)
                                                    + "/" + "delin_plot_pair" + str(i) + ".png",
                                                    delineations[i:i+2], predictions=None, column_range=eval_params.col_error_range, color="#ffe100", linewidth=2.0)

        if eval_params.save_params.ret_seg_plot is True:
            if eval_params.save_params.pngimages is True:
                if truths is not None:
                    plotting.save_segmentation_plot(cur_augment_image, cm.gray, get_loadsave_path(eval_params.save_foldername,
                                                                                                  cur_image_name)
                                                    + "/" + "seg_plot_ret.png",
                                                    np.concatenate([np.expand_dims(cur_seg[0], axis=0), np.expand_dims(cur_seg[6], axis=0)], axis=0), np.expand_dims(delineations[i], axis=0), column_range=eval_params.col_error_range, color="#ffe100", linewidth=2.0)
                    plotting.save_segmentation_plot(cur_augment_image, cm.gray, get_loadsave_path(eval_params.save_foldername,
                                                                                                  cur_image_name)
                                                    + "/" + "truth_plot_ret.png",
                                                    np.concatenate([np.expand_dims(cur_seg[0], axis=0),
                                                                   np.expand_dims(cur_seg[6], axis=0)], axis=0), predictions=None, column_range=eval_params.col_error_range, color="#ffe100", linewidth=2.0)

                plotting.save_segmentation_plot(cur_augment_image, cm.gray, get_loadsave_path(eval_params.save_foldername,
                                                                                              cur_image_name)
                                                + "/" + "delin_plot_ret.png",
                                                np.concatenate([np.expand_dims(delineations[0], axis=0),
                                                               np.expand_dims(delineations[6], axis=0)], axis=0), predictions=None, column_range=eval_params.col_error_range, color="#ffe100", linewidth=2.0)


        if eval_params.save_params.attributes is True:
            save_final_attributes(eval_params, cur_image_name, graph_time)
        if eval_params.save_params.error_plot is True:
            # TODO implement error profile plots
            pass

        if eval_params.save_params.flatten_image is True and eval_params.flatten_image is True:
            save_dataset_extra(eval_params, cur_image_name, 'flatten_boundary', 'uint16', flatten_boundary)
            save_dataset_extra(eval_params, cur_image_name, 'flatten_image', 'uint8', flattened_image)
            save_dataset_extra(eval_params, cur_image_name, "flatten_offsets", 'uint16', offsets)
            if eval_params.save_params.pngimages is True:
                plotting.save_image_plot(flattened_image, get_loadsave_path(eval_params.save_foldername, cur_image_name) +
                                         "/flattened_image.png", cmap=cm.gray, vmin=0, vmax=255)
                plotting.save_segmentation_plot(cur_augment_image, cm.gray, get_loadsave_path(eval_params.save_foldername,
                                                                                              cur_image_name)
                                                + "/" + "flatten_boundary_plot.png",
                                                np.expand_dims(flatten_boundary, axis=0), predictions=None,
                                                column_range=eval_params.col_error_range)

        if eval_params.boundaries is True and eval_params.save_params.boundary_maps is True and eval_params.trim_maps is True:
            save_dataset_extra(eval_params, cur_image_name, "trim_maps", 'uint8', trim_maps)
            if eval_params.save_params.pngimages is True:
                for map_ind in range(len(trim_maps)):
                    boundary_name = imdb.get_boundary_name(map_ind)
                    plotting.save_image_plot(trim_maps[map_ind], get_loadsave_path(eval_params.save_foldername,
                                                                                   cur_image_name) +
                                             "/trim_map_" + boundary_name + ".png", cmap=cm.Blues)

    return eval_output


def print_error_summary(overall_errors, imdb, cur_image_name):
    num_boundaries = overall_errors[0].shape[0]

    # overall errors: list of four numpy arrays: [mean abs error, mean error, abs error sd, error sd]
    print("\n")
    print("Error summary for image: " + cur_image_name)
    print("_" * 92)
    print("BOUNDARY".center(30) + "|" + "Mean absolute error [px]".center(30) + "|" + "Mean error [px]".center(30))
    print("_" * 92)
    for boundary_ind in range(num_boundaries):
        boundary_name = imdb.get_boundary_name(boundary_ind)
        mae = overall_errors[0][boundary_ind]
        me = overall_errors[1][boundary_ind]
        ae_sd = overall_errors[2][boundary_ind]
        e_sd = overall_errors[3][boundary_ind]
        first_col_str = "{:.2f} ({:.2f})".format(mae, ae_sd)
        second_col_str = "{:.2f} ({:.2f})".format(me, e_sd)
        print(boundary_name.center(30) + "|" + first_col_str.center(30) + "|" + second_col_str.center(30))
    print("\n")


def all_maps_save(eval_params, imdb, boundary_maps, predict_time, augment_time, gen_time, dices):
    if eval_params.save_params.boundary_maps is True:
        save_dataset_all_images(eval_params, "boundary_maps", 'uint8', boundary_maps)

    if eval_params.save_params.attributes is True:
        save_intermediate_attributes_allimages(eval_params, imdb, predict_time, augment_time, gen_time)

    save_filename = eval_params.save_foldername + "/results.hdf5"
    save_file = h5py.File(save_filename, 'w')

    save_file['dices'] = dices
    save_file['mean_dices'] = np.mean(dices, axis=0)
    save_file['sd_dices'] = np.std(dices, axis=0)

    save_file.close()


def intermediate_save_semantic(eval_params, imdb, cur_image_name, prob_maps, predict_time, augment_time, gen_time,
                      augment_image, augment_labels, augment_segs, cur_raw_image, cur_labels, cur_seg, area_maps,
                      comb_area_map, dices, convert_time, activations, layer_outputs):
    if eval_params.save_params.area_maps is True:
        save_dataset_extra(eval_params, cur_image_name, "area_maps", 'uint8', area_maps)

        if eval_params.save_params.pngimages is True:
            for map_ind in range(len(area_maps)):
                plotting.save_image_plot(area_maps[map_ind], get_loadsave_path(eval_params.save_foldername,
                                                                               cur_image_name) +
                                         "/area_map_" + imdb.get_fullsize_class_name(map_ind) + ".png", cmap=cm.Blues)
    if eval_params.save_params.comb_area_maps is True:
        save_dataset_extra(eval_params, cur_image_name, "comb_area_map", 'uint8', comb_area_map)

        if eval_params.save_params.pngimages is True:
            plotting.save_image_plot(comb_area_map, get_loadsave_path(eval_params.save_foldername, cur_image_name) +
                                     "/comb_area_map.png", cmap=plotting.colors.ListedColormap(plotting.region_colours,
                                                                                               N=len(area_maps)))
        if eval_params.save_params.crop_map is True:
            plotting.save_image_plot_crop(comb_area_map,
                                          get_loadsave_path(eval_params.save_foldername, cur_image_name) +
                                          "/comb_area_map_crop.png",
                                          cmap=plotting.colors.ListedColormap(plotting.region_colours,
                                                                              N=len(area_maps)),
                                          crop_bounds=eval_params.save_params.crop_bounds)
    if eval_params.boundaries is True and eval_params.save_params.boundary_maps is True:
        save_dataset_extra(eval_params, cur_image_name, "boundary_maps", 'uint8', prob_maps)
        if eval_params.save_params.pngimages is True and eval_params.save_params.indivboundarypngs:
            for map_ind in range(len(prob_maps)):
                boundary_name = imdb.get_boundary_name(map_ind)
                plotting.save_image_plot(prob_maps[map_ind], get_loadsave_path(eval_params.save_foldername,
                                                                               cur_image_name) +
                                         "/boundary_map_" + boundary_name + ".png", cmap=cm.Blues)
    if eval_params.save_params.raw_image is True:
        save_dataset_extra(eval_params, cur_image_name, "raw_image", 'uint8', cur_raw_image)
        if eval_params.save_params.pngimages is True:
            if cur_raw_image.shape[2] == 3:
                plotting.save_image_plot(cur_raw_image, get_loadsave_path(eval_params.save_foldername, cur_image_name)
                                         + "/raw_image.png", cmap=None, vmin=0, vmax=255)
            else:
                plotting.save_image_plot(cur_raw_image, get_loadsave_path(eval_params.save_foldername, cur_image_name)
                                         + "/raw_image.png", cmap=cm.gray, vmin=0, vmax=255)
            if eval_params.save_params.crop_map is True:
                plotting.save_image_plot_crop(cur_raw_image, get_loadsave_path(eval_params.save_foldername, cur_image_name) +
                                         "/raw_image_crop.png", cmap=cm.gray,
                                              crop_bounds=eval_params.save_params.crop_bounds, vmin=0, vmax=255)
    if eval_params.save_params.raw_labels is True:
        save_dataset_extra(eval_params, cur_image_name, "raw_labels", 'uint8', cur_labels)
        if eval_params.save_params.pngimages is True:
            plotting.save_image_plot(np.argmax(cur_labels, axis=2), get_loadsave_path(eval_params.save_foldername, cur_image_name)
                                     + "/comb_raw_label.png", cmap=plotting.colors.ListedColormap(plotting.region_colours,
                                                                                               N=len(area_maps)))
    if eval_params.save_params.raw_segs is True:
        save_dataset_extra(eval_params, cur_image_name, "raw_segs", 'uint16', cur_seg)
    if eval_params.save_params.aug_image is True:
        save_dataset_extra(eval_params, cur_image_name, "augment_image", 'uint8', augment_image)
        if eval_params.save_params.pngimages is True:
            plotting.save_image_plot(augment_image, get_loadsave_path(eval_params.save_foldername, cur_image_name) +
                                     "/augment_image.png", cmap=cm.gray, vmin=0, vmax=255)
            if eval_params.save_params.crop_map is True:
                plotting.save_image_plot_crop(augment_image, get_loadsave_path(eval_params.save_foldername, cur_image_name) +
                                         "/augment_image_crop.png", cmap=cm.gray,
                                              crop_bounds=eval_params.save_params.crop_bounds, vmin=0, vmax=255)
    if eval_params.save_params.aug_labels is True:
        save_dataset_extra(eval_params, cur_image_name, "augment_labels", 'uint8', augment_labels)
    if eval_params.save_params.aug_segs is True:
        save_dataset_extra(eval_params, cur_image_name, "augment_segs", 'uint16', augment_segs)
    if eval_params.save_params.boundary_names is True:
        save_dataset_extra(eval_params, cur_image_name, "boundary_names", 'S100',
                     imdb.get_boundary_names())
    if eval_params.save_params.area_names is True:
        save_dataset_extra(eval_params, cur_image_name, "area_names", 'S100', imdb.get_area_names())
    if eval_params.save_params.patch_class_names is True:
        save_dataset_extra(eval_params, cur_image_name, "patch_class_names", 'S100', imdb.get_patch_class_names())
    if eval_params.save_params.fullsize_class_names is True:
        save_dataset_extra(eval_params, cur_image_name, "fullsize_class_names", 'S100', imdb.get_fullsize_class_names())
    if eval_params.save_params.errors is True:
        if dices is not None:
            save_dataset(eval_params, cur_image_name, "dices", 'float64', np.squeeze(dices))
    if eval_params.save_params.attributes is True:
        save_intermediate_attributes_semantic(eval_params, imdb, cur_image_name, predict_time, augment_time, gen_time,
                                              convert_time)
    if eval_params.save_params.activations is True:
        for i in range(len(activations)):
            layer_name = layer_outputs[i].name.split("/")[0]

            if eval_params.save_params.pngimages is True:
                dir_name = get_loadsave_path(eval_params.save_foldername, cur_image_name) + "/activations/" + layer_name
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)

                for j in range(activations[i].shape[3]):
                    plotting.save_image_plot(activations[i][0, :, :, j], filename=dir_name + "/" + str(j), cmap=cm.viridis)

            bin_act = np.array(activations[i][0, :, :, :])
            bin_act[bin_act != 0] = 1

            save_dataset_extra(eval_params, cur_image_name, "activations/" + layer_name, 'float32', activations[i][0, :, :, :])
            save_dataset_extra(eval_params, cur_image_name, "activations_bin/" + layer_name, 'float32', bin_act)


def intermediate_save_patch_based(eval_params, imdb, cur_image_name, prob_maps, predict_time, augment_time,
                                  gen_time, convert_time, patch_time,
                                  augment_image, augment_labels, augment_segs, cur_raw_image, cur_labels, cur_seg):
    if eval_params.save_params.boundary_maps is True:
        save_dataset_extra(eval_params, cur_image_name, "boundary_maps", 'uint8', prob_maps)
        if eval_params.save_params.pngimages is True:
            for map_ind in range(len(prob_maps)):
                boundary_name = imdb.get_patch_class_name(map_ind)
                plotting.save_image_plot(prob_maps[map_ind], get_loadsave_path(eval_params.save_foldername,
                                                                               cur_image_name) +
                                         "/boundary_map_" + boundary_name + ".png", cmap=cm.Blues)
    if eval_params.save_params.raw_image is True:
        save_dataset_extra(eval_params, cur_image_name, "raw_image", 'uint8', cur_raw_image)
        if eval_params.save_params.pngimages is True:
            plotting.save_image_plot(cur_raw_image, get_loadsave_path(eval_params.save_foldername, cur_image_name)
                                     + "/raw_image.png", cmap=cm.gray)
    if eval_params.save_params.raw_labels is True:
        save_dataset_extra(eval_params, cur_image_name, "raw_labels", 'uint8', cur_labels)
    if eval_params.save_params.raw_segs is True:
        save_dataset_extra(eval_params, cur_image_name, "raw_segs", 'uint16', cur_seg)
    if eval_params.save_params.aug_image is True:
        save_dataset_extra(eval_params, cur_image_name, "augment_image", 'uint8', augment_image)
        if eval_params.save_params.pngimages is True:
            plotting.save_image_plot(augment_image, get_loadsave_path(eval_params.save_foldername, cur_image_name) +
                                     "/augment_image.png", cmap=cm.gray)
    if eval_params.save_params.aug_labels is True:
        save_dataset_extra(eval_params, cur_image_name, "augment_labels", 'uint8', augment_labels)
    if eval_params.save_params.aug_segs is True:
        save_dataset_extra(eval_params, cur_image_name, "augment_segs", 'uint16', augment_segs)
    if eval_params.save_params.boundary_names is True:
        save_dataset_extra(eval_params, cur_image_name, "boundary_names", 'S100',
                     imdb.get_boundary_names())
    if eval_params.save_params.area_names is True:
        save_dataset_extra(eval_params, cur_image_name, "area_names", 'S100', imdb.get_area_names())
    if eval_params.save_params.patch_class_names is True:
        save_dataset_extra(eval_params, cur_image_name, "patch_class_names", 'S100', imdb.get_patch_class_names())
    if eval_params.save_params.attributes is True:
        save_intermediate_attributes_patch_based(eval_params, imdb, cur_image_name, predict_time, augment_time,
                                                 gen_time, convert_time, patch_time)


def delete_loadsaveextra_file(eval_params, name):
    if eval_params.verbosity >= 2:
        print("Deleting extra evaluations file...")

    loadsaveextra_path = get_loadsaveextra_path(eval_params.save_foldername, name)

    if os.path.isfile(get_loadsaveextra_filename(loadsaveextra_path)):
        os.remove(get_loadsaveextra_filename(loadsaveextra_path))


def delete_allimages_file(eval_params):
    if eval_params.verbosity >= 2:
        print("Deleting extra evaluations file...")

    allimages_path = get_allimages_path(eval_params.save_foldername)

    if os.path.isfile(get_allimages_filename(allimages_path)):
        os.remove(get_allimages_filename(allimages_path))


def save_dataset(eval_params, name, dset_name, datatype, dataset):
    if eval_params.verbosity >= 2:
        print("Saving dataset (" + dset_name + ") to file...")

    loadsave_file = open_append_loadsave_file(eval_params.save_foldername, name)

    dataset = np.array(dataset, dtype=datatype)
    dset = loadsave_file.require_dataset(dset_name, dataset.shape, dtype=datatype)
    dset[:] = dataset

    loadsave_file.close()


def save_dataset_extra(eval_params, name, dset_name, datatype, dataset):
    if eval_params.verbosity >= 2:
        print("Saving dataset (" + dset_name + ") to file...")

    loadsave_file = open_append_loadsaveextra_file(eval_params.save_foldername, name)

    dataset = np.array(dataset, dtype=datatype)
    dset = loadsave_file.require_dataset(dset_name, dataset.shape, dtype=datatype)
    dset[:] = dataset

    loadsave_file.close()


def save_dataset_all_images(eval_params, dset_name, datatype, dataset):
    if eval_params.verbosity >= 2:
        print("Saving dataset (" + dset_name + ") to file...")

    loadsave_file = open_allimages_file(eval_params.save_foldername, 'a')

    dataset = np.array(dataset, dtype=datatype)
    dset = loadsave_file.require_dataset(dset_name, dataset.shape, dtype=datatype)
    dset[:] = dataset

    loadsave_file.close()


def load_dataset(eval_params, name, dset_name):
    if eval_params.verbosity >= 2:
        print("Loading dataset (" + dset_name + ") from file...")

    loadsave_file = open_read_loadsave_file(eval_params.save_foldername, name)

    dset = loadsave_file[dset_name][:]

    loadsave_file.close()

    return dset


def load_dataset_extra(eval_params, name, dset_name):
    if eval_params.verbosity >= 2:
        print("Loading dataset (" + dset_name + ") from file...")

    loadsaveextra_file = open_read_loadsaveextra_file(eval_params.save_foldername, name)

    dset = loadsaveextra_file[dset_name][:]

    loadsaveextra_file.close()

    return dset


def load_dataset_all_images_nonram(eval_params, dset_name):
    if eval_params.verbosity >= 2:
        print("Loading dataset (" + dset_name + ") from file...")

    allimages_file = open_allimages_file(eval_params.save_foldername, 'r')

    dset = allimages_file[dset_name]

    return [dset, allimages_file]


def load_dataset_results_nonram(eval_params, dset_name):
    if eval_params.verbosity >= 2:
        print("Loading dataset (" + dset_name + ") from file...")

    results_file = open_results_file(eval_params.save_foldername, 'r')

    dset = results_file[dset_name]

    return [dset, results_file]


def save_intermediate_attributes_semantic(eval_params, imdb, name, predict_time, augment_time, generator_time,
                                          convert_time):
    if eval_params.verbosity >= 2:
        print("Saving intermediate attributes...")

    loadsave_file = open_append_loadsave_file(eval_params.save_foldername, name)

    loadsave_file.attrs["aug_desc"] = np.array(eval_params.aug_desc, dtype='S100')
    if not eval_params.ensemble:
        loadsave_file.attrs["model_filename"] = np.array(eval_params.model_filename, dtype='S100')
        loadsave_file.attrs["network_foldername"] = np.array(eval_params.network_foldername, dtype='S100')
    else:
        loadsave_file.attrs["model_filenames"] = np.array(eval_params.model_filenames, dtype='S100')
        loadsave_file.attrs["network_foldernames"] = np.array(eval_params.network_foldernames, dtype='S100')
    loadsave_file.attrs["data_filename"] = np.array(imdb.filename, dtype='S100')
    loadsave_file.attrs["predict_time"] = np.array(predict_time)
    loadsave_file.attrs["augment_time"] = np.array(augment_time)
    loadsave_file.attrs["generator_time"] = np.array(generator_time)
    loadsave_file.attrs["intermediate_timestamp"] = np.array(common.get_timestamp(), dtype='S100')
    if eval_params.save_params.boundary_maps is True:
        loadsave_file.attrs["complete_predict"] = True
    loadsave_file.attrs["convert_time"] = convert_time
    loadsave_file.attrs["ensemble"] = eval_params.ensemble

    loadsave_file.close()


def save_intermediate_attributes_patch_based(eval_params, imdb, name, predict_time, augment_time, generator_time,
                                          convert_time, patch_time=None):
    if eval_params.verbosity >= 2:
        print("Saving intermediate attributes...")

    loadsave_file = open_append_loadsave_file(eval_params.save_foldername, name)

    loadsave_file.attrs["aug_desc"] = np.array(eval_params.aug_desc, dtype='S100')
    loadsave_file.attrs["model_filename"] = np.array(eval_params.model_filename, dtype='S100')
    loadsave_file.attrs["network_foldername"] = np.array(eval_params.network_foldername, dtype='S100')
    loadsave_file.attrs["data_filename"] = np.array(imdb.filename, dtype='S100')
    loadsave_file.attrs["patch_size"] = np.array(eval_params.patch_size)
    loadsave_file.attrs["predict_time"] = np.array(predict_time)
    loadsave_file.attrs["augment_time"] = np.array(augment_time)
    loadsave_file.attrs["convert_time"] = np.array(convert_time)
    loadsave_file.attrs["patch_time"] = np.array(patch_time)
    loadsave_file.attrs["generator_time"] = np.array(generator_time)
    loadsave_file.attrs["intermediate_timestamp"] = np.array(common.get_timestamp(), dtype='S100')
    if eval_params.save_params.boundary_maps is True:
        loadsave_file.attrs["complete_predict"] = True

    loadsave_file.close()


def save_final_attributes(eval_params, name, graph_time):
    if eval_params.verbosity >= 2:
        print("Saving final attributes...")

    loadsave_file = open_append_loadsave_file(eval_params.save_foldername, name)

    loadsave_file.attrs["final_timestamp"] = np.array(common.get_timestamp(), dtype='S100')

    if eval_params.boundaries is True:
        loadsave_file.attrs["graph_time"] = np.array(graph_time)
        loadsave_file.attrs["error_col_bounds"] = np.array((eval_params.col_error_range[0],
                                                            eval_params.col_error_range[-1]))
        loadsave_file.attrs["gsgrad"] = np.array(eval_params.gsgrad)
        loadsave_file.attrs["complete_graph"] = True

    loadsave_file.close()


def save_initial_attributes(eval_params, name):
    if eval_params.verbosity >= 2:
        print("Saving initial attributes...")

    loadsave_file = open_append_loadsave_file(eval_params.save_foldername, name)

    loadsave_file.attrs["complete_predict"] = False
    loadsave_file.attrs["initial_timestamp"] = np.array(common.get_timestamp(), dtype='S100')

    if eval_params.boundaries is True:
        loadsave_file.attrs["complete_graph"] = False

    loadsave_file.close()


def save_initial_attributes_all_images(eval_params):
    if eval_params.verbosity >= 2:
        print("Saving initial attributes for all images file...")

    loadsave_file = open_allimages_file(eval_params.save_foldername, 'a')

    loadsave_file.attrs["complete_predict"] = False
    loadsave_file.attrs["initial_timestamp"] = np.array(common.get_timestamp(), dtype='S100')

    loadsave_file.close()


def save_intermediate_attributes_allimages(eval_params, imdb, predict_time, augment_time, generator_time):
    if eval_params.verbosity >= 2:
        print("Saving intermediate attributes...")

    loadsave_file = open_allimages_file(eval_params.save_foldername, 'a')

    loadsave_file.attrs["aug_desc"] = np.array(eval_params.aug_desc, dtype='S100')
    if not eval_params.ensemble:
        loadsave_file.attrs["model_filename"] = np.array(eval_params.model_filename, dtype='S100')
        loadsave_file.attrs["network_foldername"] = np.array(eval_params.network_foldername, dtype='S100')
    else:
        loadsave_file.attrs["model_filenames"] = np.array(eval_params.model_filenames, dtype='S100')
        loadsave_file.attrs["network_foldernames"] = np.array(eval_params.network_foldernames, dtype='S100')
    loadsave_file.attrs["data_filename"] = np.array(imdb.filename, dtype='S100')
    loadsave_file.attrs["predict_time"] = np.array(predict_time)
    loadsave_file.attrs["augment_time"] = np.array(augment_time)
    loadsave_file.attrs["generator_time"] = np.array(generator_time)
    loadsave_file.attrs["intermediate_timestamp"] = np.array(common.get_timestamp(), dtype='S100')
    loadsave_file.attrs["ensemble"] = eval_params.ensemble

    if eval_params.save_params.boundary_maps is True:
        loadsave_file.attrs["complete_predict"] = True

    loadsave_file.close()


def check_exists(save_foldername, name):
    loadsave_path = get_loadsave_path(save_foldername, name)

    if not os.path.exists(loadsave_path):
        os.makedirs(loadsave_path)

    loadsave_filename = get_loadsave_filename(loadsave_path)

    if os.path.isfile(loadsave_filename):
        return True
    else:
        return False


def check_allimages_exists(save_foldername):
    allimages_path = get_allimages_path(save_foldername)

    if not os.path.exists(allimages_path):
        os.makedirs(allimages_path)

    allimages_filename = get_allimages_filename(allimages_path)

    if os.path.isfile(allimages_filename):
        return True
    else:
        return False


def get_complete_status_allimages(save_foldername):
    loadsave_file = open_allimages_file(save_foldername, 'r')

    if loadsave_file.attrs["complete_predict"] == True:
        status = 'predict'
    else:
        status = 'none'

    loadsave_file.close()
    return status


def get_complete_status(save_foldername, name, boundaries):
    loadsave_file = open_read_loadsave_file(save_foldername, name)

    if boundaries:
        if loadsave_file.attrs["complete_graph"] == True:
            status = 'graph'
        elif loadsave_file.attrs["complete_predict"] == True:
            status = 'predict'
        else:
            status = 'none'
    else:
        if loadsave_file.attrs["complete_predict"] == True:
            status = 'predict'
        else:
            status = 'none'

    loadsave_file.close()
    return status


def open_append_loadsave_file(save_foldername, name):
    loadsave_path = get_loadsave_path(save_foldername, name)

    if not os.path.exists(loadsave_path):
        os.makedirs(loadsave_path)

    loadsave_file = h5py.File(get_loadsave_filename(loadsave_path), 'a')

    return loadsave_file


def open_append_loadsaveextra_file(save_foldername, name):
    loadsaveextra_path = get_loadsaveextra_path(save_foldername, name)

    if not os.path.exists(loadsaveextra_path):
        os.makedirs(loadsaveextra_path)

    loadsaveextra_file = h5py.File(get_loadsaveextra_filename(loadsaveextra_path), 'a')

    return loadsaveextra_file


def open_allimages_file(save_foldername, mode):
    all_images_path = get_allimages_path(save_foldername)

    if mode == 'a':
        if not os.path.exists(all_images_path):
            os.makedirs(all_images_path)

    loadsave_file = h5py.File(get_allimages_filename(all_images_path), mode)

    return loadsave_file


def open_results_file(save_foldername, mode):
    results_path = get_results_path(save_foldername)

    if mode == 'a':
        if not os.path.exists(results_path):
            os.makedirs(results_path)

    loadsave_file = h5py.File(get_results_filename(results_path), mode)

    return loadsave_file


def open_read_loadsave_file(save_foldername, name):
    loadsave_path = get_loadsave_path(save_foldername, name)
    loadsave_file = h5py.File(get_loadsave_filename(loadsave_path), 'r')

    return loadsave_file


def open_read_loadsaveextra_file(save_foldername, name):
    loadsaveextra_path = get_loadsaveextra_path(save_foldername, name)
    loadsaveextra_file = h5py.File(get_loadsaveextra_filename(loadsaveextra_path), 'r')

    return loadsaveextra_file


def get_loadsave_path(save_foldername, name):
    return save_foldername + "/" + name


def get_loadsaveextra_path(save_foldername, name):
    return save_foldername + "/" + name


def get_loadsave_filename(loadsave_path):
    return loadsave_path + "/evaluations.hdf5"


def get_loadsaveextra_filename(loadsave_path):
    return loadsave_path + "/evaluations_extra.hdf5"


def get_allimages_path(save_foldername):
    return save_foldername


def get_allimages_filename(allimages_path):
    return allimages_path + "/all_images.hdf5"


def get_results_path(save_foldername):
    return save_foldername


def get_results_filename(allimages_path):
    return allimages_path + "/results.hdf5"


def convert_maps_uint8(prob_maps):
    prob_maps *= 255
    prob_maps = prob_maps.astype('uint8')

    return prob_maps


def convert_maps_float64(prob_maps):
    prob_maps = prob_maps.astype('float64')
    prob_maps /= 255

    return prob_maps


def soft_dice_numpy(y_pred, y_true, eps=1e-7):
    '''
    c is number of classes
    :param y_pred: b x c x X x Y( x Z...) network output, must sum to 1 over c channel (such as after softmax)
    :param y_true: b x c x X x Y( x Z...) one hot encoding of ground truth
    :param eps:
    :return:
    '''

    axes = tuple(range(2, len(y_pred.shape)))
    intersect = np.sum(y_pred * y_true, axis=axes)
    denom = np.sum(y_pred + y_true, axis=axes)

    for i in range(intersect.shape[1]):
        # if there is no region for a class to predict, there shouldn't be a penalty for correctly predicting empty
        if intersect[0, i] == 0 and denom[0, i] == 0:
            # set to 1
            # intersect[0, i] = 0.5
            # denom[0, i] = 1 - eps

            # OR

            # set to NaN
            intersect[0, i] = np.nan

    class_dices = 2. * intersect / (denom + eps)

    intersect_total = np.sum(np.sum(y_pred * y_true, axis=axes))
    denom_total = np.sum(np.sum(y_pred + y_true, axis=axes))

    overall_dice = np.array([np.array([2. * intersect_total / (denom_total + eps)])])

    return np.concatenate((class_dices, overall_dice), axis=1)


def soft_dice_numpy_multiple(y_pred, y_true, eps=1e-7):
    '''
    c is number of classes
    :param y_pred: b x c x X x Y( x Z...) network output, must sum to 1 over c channel (such as after softmax)
    :param y_true: b x c x X x Y( x Z...) one hot encoding of ground truth
    :param eps:
    :return:
    '''

    axes = tuple(range(2, len(y_pred.shape)))
    intersect = np.sum(y_pred * y_true, axis=axes)
    denom = np.sum(y_pred + y_true, axis=axes)

    print(intersect.shape)

    for j in range(intersect.shape[0]):
        for i in range(intersect.shape[1]):
            # if there is no region for a class to predict, there shouldn't be a penalty for correctly predicting empty
            if intersect[j, i] == 0 and denom[j, i] == 0:
                # set to 1
                # intersect[0, i] = 0.5
                # denom[0, i] = 1 - eps

                # OR

                # set to NaN
                intersect[j, i] = np.nan

    class_dices = 2. * intersect / (denom + eps)

    return class_dices


def calc_dice(eval_params, predictions, labels):
    if eval_params is not None:
        low = eval_params.col_error_range[0]
        high = eval_params.col_error_range[-1] + 1
    else:
        low = 0
        high = predictions.shape[2]

    if K.image_data_format() == 'channels_last':
        dices = soft_dice_numpy(predictions[:, :, low:high, :],
                                            np.transpose(labels, axes=(0, 3, 1, 2))[:, :, low:high, :])
    else:
        dices = soft_dice_numpy(predictions[:, :, low:high, :], labels[:, :, low:high, :])

    return dices


def perform_argmax(predictions, ensemble, bin=True):
    if not ensemble:
        if K.image_data_format() == 'channels_last':
            pass
        else:
            predictions = np.transpose(predictions, (0, 2, 3, 1))

        num_maps = predictions.shape[3]

        if bin:
            argmax_pred = np.argmax(predictions, axis=3)

            categorical_pred = to_categorical(argmax_pred, num_maps)
            categorical_pred = np.transpose(categorical_pred, axes=(0, 3, 1, 2))
        else:
            argmax_pred = np.argmax(predictions, axis=3)
            categorical_pred = np.transpose(predictions, axes=(0, 3, 1, 2))
    else:
        argmax_pred = []
        categorical_pred = []

        for i in range(len(predictions)):
            if K.image_data_format() == 'channels_last':
                pass
            else:
                predictions[i] = np.transpose(predictions[i], (0, 2, 3, 1))

            num_maps = predictions[i].shape[3]

            if bin:
                argmax_pred.append(np.argmax(predictions[i], axis=3))

                categorical_pred.append(to_categorical(argmax_pred[i], num_maps))
                categorical_pred[i] = np.transpose(categorical_pred[i], axes=(0, 3, 1, 2))
            else:
                argmax_pred.append(np.argmax(predictions[i], axis=3))

                categorical_pred.append(np.transpose(predictions[i], axes=(0, 3, 1, 2)))

    return [argmax_pred, categorical_pred]


def perform_ensemble(area_maps_sep):
    area_maps_ensemble = np.concatenate(area_maps_sep, axis=0)

    area_maps_ensemble = np.sum(area_maps_ensemble, axis=0)

    area_maps_ensemble /= len(area_maps_sep)

    area_maps_ensemble = np.expand_dims(area_maps_ensemble, axis=0)

    comb_area_maps_ensemble = np.argmax(area_maps_ensemble, axis=1)

    return [comb_area_maps_ensemble, area_maps_ensemble]


def perform_ensemble_patch(boundary_maps_sep):
    boundary_maps_ensemble = np.concatenate(boundary_maps_sep, axis=0)

    boundary_maps_ensemble = np.sum(boundary_maps_ensemble, axis=0)

    boundary_maps_ensemble = convert_maps_float64(boundary_maps_ensemble)

    boundary_maps_ensemble /= len(boundary_maps_sep)

    boundary_maps_ensemble = convert_maps_uint8(boundary_maps_ensemble)

    return boundary_maps_ensemble


def calc_areas(y_pred, y_true):
    areas = []
    area_diffs = []
    area_abs_diffs = []
    for i in range(y_pred.shape[0]):
        pred_area_size = np.count_nonzero(y_pred[i, :, :, 1])
        true_area_size = np.count_nonzero(y_true[i, :, :, 1])

        areas.append([pred_area_size, true_area_size])
        area_diffs.append(pred_area_size - true_area_size)
        area_abs_diffs.append(np.abs(pred_area_size - true_area_size))

    return [areas, area_diffs, area_abs_diffs]
