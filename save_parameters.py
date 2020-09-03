class SaveParameters:
    """Parameters for saving results for evaluation parameters.
        _________

        prob_maps: whether to save the boundary probability maps
        _________

        delineations: whether to save the boundary location predictions
        _________

        errors: whether to save the errors between predictions and truths
        _________

        seg_plot: whether to save a plot of the raw image overlaid with truths and predictions line markings
        _________

        area_maps: (semantic only) - whether to save the predicted area maps
        _________

        attributes: whether to save various attributes associated with the evaluation
        _________

        raw_image: whether to save the raw image
        _________

        raw_segs: whether to save the raw segmentations (true boundary positions)
        _________

        raw_maps: whether to save the raw area maps
        _________

        error_plot: whether to save a plot of the error of each boundary for each column
        _________

        aug_image: whether to save the augmented image
        _________

        aug_labels: whether to save the augmented labels (save patch labels or masks depending on the situation)
        _________

        aug_segs: whether to save the augmented boundary segmentations (truths)
        _________

        boundary_names: whether to save the boundary names
        _________

        area_names: whether to save the area names
        _________

        """

    def __init__(self, boundary_maps=True, temp_extra=True, delineations=True, errors=True, seg_plot=False, indiv_seg_plot=False, pair_seg_plot=False, ret_seg_plot=False,
                 area_maps=False, comb_area_maps=False, comb_area_maps_recalc=False, attributes=True, raw_image=False, aug_image=False, aug_labels=False, aug_segs=False,
                 raw_segs=False, raw_labels=False, error_plot=False, boundary_names=False, area_names=False,
                 pngimages=False, indivboundarypngs=False, patch_class_names=False, fullsize_class_names=False,
                 crop_map=False, crop_bounds=None, activations=False, act_layers=None, flatten_image=False, flatten_ind=0, flatten_poly=False,
                 disable=False, output_var=False):
        self.boundary_maps = boundary_maps
        self.delineations = delineations
        self.errors = errors
        self.seg_plot = seg_plot
        self.area_maps = area_maps
        self.comb_area_maps = comb_area_maps
        self.comb_area_maps_recalc = comb_area_maps_recalc
        self.attributes = attributes
        self.raw_image = raw_image
        self.aug_image = aug_image
        self.raw_segs = raw_segs
        self.raw_labels = raw_labels
        self.error_plot = error_plot
        self.aug_labels = aug_labels
        self.aug_segs = aug_segs
        self.boundary_names = boundary_names
        self.area_names = area_names
        self.patch_class_names = patch_class_names
        self.fullsize_class_names = fullsize_class_names
        self.pngimages = pngimages
        self.indivboundarypngs = indivboundarypngs
        self.temp_extra = temp_extra
        self.crop_map = crop_map
        self.crop_bounds = crop_bounds
        self.activations = activations
        self.flatten_image = flatten_image
        self.flatten_ind = flatten_ind
        self.flatten_poly = flatten_poly
        self.disable = disable
        self.output_var = output_var
        self.act_layers = act_layers
        self.indiv_seg_plot = indiv_seg_plot
        self.pair_seg_plot = pair_seg_plot
        self.ret_seg_plot = ret_seg_plot
