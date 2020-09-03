import augmentation as aug
import save_parameters as sparams


class EvaluationParameters:
    """Parameters for evaluation of trained network.
        _________

        model_filename: filename of Keras model to be loaded
        _________

        data_filename: filename of .hdf5 or .h5 file to be used to load images, masks, segs and names.
        Data files should contain 5 datasets.

        'images' with shape: (number of images, width, height) (dtype = 'uint8')
        'mask_labels' with shape: (number of images, width, height) (dtype = 'uint8')
        'patch_labels' with shape: (number of images, width, height) (dtype = 'uint8')
        'segs' with shape: (number of images, number of boundaries, width) (dtype = 'uint16')
        'image_names' with shape: (number of images,) (dtype = 'S' - fixed length strings)
        'boundary_names' with shape: (number of boundaries,) (dtype = 'S' - fixed length strings)
        'area_names' with shape: (number of boundaries + 1,) (dtype = 'S' - fixed length strings)
        _________

        batch_size: size of batch to be used (number of images through network at once)
        _________

        aug_fn_args: 2 tuple includes: function used to augment each image,
         tuple of arguments to be passed to the augmentation function. Default: (None, None), do not use augmentation.
        _________

        graph_structure: graph neighbour structure to be used for evaluations
        _________

        eval_mode

        both: predict using network and construct probability maps AND segment maps with graph search
        network: predict using network and construct probability maps ONLY
        gs: segment maps with graph search ONLY
        _________

        save_filename

        file/folder used to load and save predicted maps, delineations, errors and other information.
        Usage depends on mode and output_type.

        when output_type = 'file':

        when mode is:
        both: will save prob maps, delineations and errors to file
        network: will save probs maps to file
        gs: will load prob maps from file and save delineations and errors to file

        savefiles contain a number of datasets for each image and information associated with each:

        'predictions' with shape (number of boundaries, width) (dtype = 'uint16')
        'errors' with shape (number of boundaries, width) (dtype = 'int16')     (SIGNED INTEGER FOR NEGATIVE VALUES)
        'boundary_maps' with shape (number of boundaries, width, height)  (dtype = 'uint8')
        'area_maps' with shape (number of boundaries + 1, width, height) (dtype = 'uint8') (semantic segmentation ONLY)
        'datafile_name' string with name of the datafile used for evaluation
        'model_name' string with name of the model used for evaluation
        'augmentation' string with a description of the augmentation function used for evaluation
        'patch_size' string with the size of patches used for evaluation
        'error_range' string with the column range used for error calculations
        _________

        patch_size: size of patches to use with shape: (patch width, patch height)
        _________

        col_error_range: range of columns to calculate errors. Default: None calculates error for all columns.
        _________

        save_params: parameters using to determine what to save from the evaluation
        _________

        verbosity: level of verbosity to use when printing output to the console

        0: no additional output
        1: network evaluation progress
        2  network evaluation and output progress
        3: network evaluation and output progress and results
        _________

        binar_boundary_maps: whether to binarize to boundary probability maps or not
        _________

        """
    def __init__(self, loaded_model, model_filename, network_foldername, dataname,
                 batch_size, graph_structure, col_error_range,
                 eval_mode='both', aug_fn_arg=(aug.no_aug, {}), patch_size=None, save_params=sparams.SaveParameters(),
                 transpose=False, normalise_input=True,
                 verbosity=3, gsgrad=1, comb_pred=False, recalc_errors=False, boundaries=True,
                 trim_maps=False, trim_ref_ind=0, trim_window=(0, 0), dice_errors=True,
                 save_foldername=None, flatten_image=False, flatten_ind=0, flatten_poly=False, ensemble=False,
                 loaded_models=None, model_filenames=None, network_foldernames=None, binarize=True, binarize_after=False, vertical_graph_search=False, bg_ilm=True, bg_csi=False, flatten_pred_edges=False,
                 flat_marg=0, use_thresh=False, thresh=0.5):
        self.loaded_model = loaded_model
        self.loaded_models = loaded_models
        self.model_filename = model_filename
        self.model_filenames = model_filenames
        self.network_foldername = network_foldername
        self.network_foldernames = network_foldernames

        self.ensemble = ensemble
        self.binarize = binarize
        self.binarize_after = binarize_after

        self.dataname = dataname
        self.batch_size = batch_size
        self.eval_mode = eval_mode
        self.aug_fn_arg = aug_fn_arg
        self.patch_size = patch_size
        self.col_error_range = col_error_range
        self.save_params = save_params
        self.transpose = transpose
        self.normalise_input = normalise_input
        self.verbosity = verbosity
        self.comb_pred = comb_pred
        self.recalc_errors = recalc_errors
        self.boundaries = boundaries
        self.trim_maps = trim_maps
        self.trim_ref_ind = trim_ref_ind
        self.trim_window = trim_window
        self.dice_errors = dice_errors
        self.flatten_image = flatten_image
        self.flatten_ind = flatten_ind
        self.flatten_poly = flatten_poly

        self.graph_structure = graph_structure
        self.vertical_graph_search = vertical_graph_search
        self.flatten_pred_edges = flatten_pred_edges
        self.flat_marg = flat_marg

        self.bg_ilm=bg_ilm
        self.bg_csi=bg_csi

        self.use_thresh=use_thresh
        self.thresh=thresh

        self.aug_fn = aug_fn_arg[0]
        self.aug_arg = aug_fn_arg[1]
        self.aug_desc = self.aug_fn(None, None, None, self.aug_arg, desc_only=True)
        self.gsgrad = gsgrad

        if save_foldername is None:
            if boundaries is True:
                self.save_foldername = self.network_foldername + "/" + self.aug_desc + "_" + self.dataname
            else:
                self.save_foldername = self.network_foldername + "/" + self.aug_desc + "_" + \
                                       self.dataname
        else:
            self.save_foldername = save_foldername

        if self.verbosity >= 1:
            self.predict_verbosity = 1
        else:
            self.predict_verbosity = 0


