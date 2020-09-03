class EvaluationOutput:
    raw_image = None
    raw_label = None
    image_name = None
    raw_seg = None
    aug_image = None
    aug_label = None
    aug_seg = None
    comb_area_map = None
    area_maps = None
    boundary_maps = None
    delineations = None
    errors = None
    trim_maps = None
    flattened_image = None
    offsets = None
    flatten_boundary = None

    def __init__(self):
        pass


