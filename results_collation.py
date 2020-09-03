import numpy as np
import os
import h5py


def calc_overall_dataset_errors(folder_path, inc_boundary_errors=True, inc_dice=True, inc_recon_dice=False):
    dir_list = os.listdir(folder_path)

    errors = None
    dices = None
    dices_recon = None
    obj_names = []

    for obj_name in dir_list:
        if os.path.isdir(folder_path + '/' + obj_name):
            filename = folder_path + '/' + obj_name + '/evaluations.hdf5'

            if os.path.exists(filename):
                file = h5py.File(filename, 'r')

                obj_names.append(obj_name)

                if inc_boundary_errors is True:
                    error_col_bounds = file.attrs["error_col_bounds"]
                    file_errors = file["errors"][:, error_col_bounds[0]:error_col_bounds[1] + 1]

                    if errors is None:
                        errors = np.expand_dims(file_errors, 0)
                    else:
                        errors = np.concatenate((errors, np.expand_dims(file_errors, 0)), 0)

                if inc_dice is True:
                    file_dices = file["dices"][:]

                    if dices is None:
                        dices = np.expand_dims(file_dices, 0)
                    else:
                        dices = np.concatenate((dices, np.expand_dims(file_dices, 0)), 0)

                if inc_recon_dice is True:
                    file_dices_recon = file["dices_recon"][:]

                    if dices_recon is None:
                        dices_recon = np.expand_dims(file_dices_recon, 0)
                    else:
                        dices_recon = np.concatenate((dices_recon, np.expand_dims(file_dices_recon, 0)), 0)
            else:
                # no evaluations file in this folder
                pass

    save_filename = folder_path + "/results.hdf5"
    save_file = h5py.File(save_filename, 'w')

    save_textfilename = folder_path + "/results.csv"
    save_textfile = open(save_textfilename, 'w')

    save_file['image_names'] = np.array(obj_names, dtype='S100')

    if inc_boundary_errors is True:
        mean_abs_errors_cols = np.nanmean(np.abs(errors), axis=0)
        mean_abs_errors_samples = np.nanmean(np.abs(errors), axis=2)
        sd_abs_errors_samples = np.nanstd(np.abs(errors), axis=2)
        mean_abs_errors = np.nanmean(mean_abs_errors_samples, axis=0)
        sd_abs_errors = np.nanstd(mean_abs_errors_samples, axis=0)

        median_abs_errors = np.nanmedian(mean_abs_errors_samples, axis=0)

        mean_errors_cols = np.nanmean(errors, axis=0)
        mean_errors_samples = np.nanmean(errors, axis=2)
        mean_errors = np.nanmean(mean_errors_samples, axis=0)
        sd_errors = np.nanstd(mean_errors_samples, axis=0)

        median_errors = np.nanmedian(mean_errors_samples, axis=0)

        save_file['mean_abs_errors_cols'] = mean_abs_errors_cols
        save_file['mean_abs_errors_samples'] = mean_abs_errors_samples
        save_file['mean_abs_errors'] = mean_abs_errors
        save_file['sd_abs_errors'] = sd_abs_errors
        save_file['median_abs_errors'] = median_abs_errors
        save_file['sd_abs_errors_samples'] = sd_abs_errors_samples

        save_file['mean_errors_cols'] = mean_errors_cols
        save_file['mean_errors_samples'] = mean_errors_samples
        save_file['mean_errors'] = mean_errors
        save_file['sd_errors'] = sd_errors
        save_file['median_errors'] = median_errors

        save_file['errors'] = errors

        save_textfile.write("Mean abs errors,")
        write_array_to_csvfile(save_textfile, mean_abs_errors)

        save_textfile.write("Mean errors,")
        write_array_to_csvfile(save_textfile, mean_errors)

        save_textfile.write("Median absolute errors,")
        write_array_to_csvfile(save_textfile, median_abs_errors)

        save_textfile.write("SD abs errors,")
        write_array_to_csvfile(save_textfile, sd_abs_errors)

        save_textfile.write("SD errors,")
        write_array_to_csvfile(save_textfile, sd_errors)

    if inc_dice is True:
        save_file['dices'] = dices

        mean_dices = np.nanmean(dices, axis=0)
        sd_dices = np.nanstd(dices, axis=0)

        save_file['mean_dices'] = mean_dices
        save_file['sd_dices'] = sd_dices

        save_textfile.write("Mean dices,")
        write_array_to_csvfile(save_textfile, mean_dices)

        save_textfile.write("SD dices,")
        write_array_to_csvfile(save_textfile, sd_dices)

    if inc_recon_dice is True:
        save_file['dices_recon'] = dices_recon

        mean_dices_recon = np.mean(dices_recon, axis=0)
        sd_dices_recon = np.std(dices_recon, axis=0)

        save_file['mean_dices_recon'] = mean_dices_recon
        save_file['sd_dices_recon'] = sd_dices_recon

        save_textfile.write("Mean dices recon,")
        write_array_to_csvfile(save_textfile, mean_dices_recon)

        save_textfile.write("SD dices recon,")
        write_array_to_csvfile(save_textfile, sd_dices_recon)

    save_file.close()
    save_textfile.close()


def write_array_to_csvfile(csv_file, array):
    for i in range(len(array)):
        csv_file.write(str(array[i]))
        if i != len(array) - 1:
            csv_file.write(",")

    csv_file.write('\n')

