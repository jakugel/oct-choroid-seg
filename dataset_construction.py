import h5py
import time
import numpy as np
import datetime
import parameters

from keras import backend as K

# all datasets have common trailing indices
# patch images: (, width, height, channels)
# full size images: (, width, height, channels)
# patch labels: (, 1)
# full size labels: (, width, height, 1)
# but may have zero or more uncommon leading indices:
# for example:
# participants and scans:
# patch images: (participants, scans per participant, patches per scan, width, height, channels)
# full size images: (participants, scans per participant, width, height, channels)
# patch labels: (participants, scans per participant, patches per scan, 1)
# full size labels: (participants, scans per participant, width, height, 1)
# single leading ind
# patch images: (total patches, width, height, channels)
# full size images: (total scans, width, height, channels)
# patch labels: (total patches, 1)
# full size labels: (total scans, width, height, 1)


def construct_dataset(images, labels, segs, write_filename, trainvaltest, boundary_names, area_names, patch_class_names,
                      fullsize_class_names, image_names, start_construct_time, patches, patch_labels, patch_col_range,
                      patch_size, num_boundaries, num_areas, num_channels, dim_ordering, dim_names, alt_output=parameters.DATA_LOCATION,
                      bg_mode='single', bg_margin=0):
    images = np.array(images, dtype='uint8')

    if labels is not None:
        labels = np.array(labels, dtype='uint8')

    if patches is True:
        labels = np.expand_dims(labels, axis=-1)

        patch_width = patch_size[0]
        patch_height = patch_size[1]

        multi_bg_str = "_" + bg_mode
        if bg_margin != 0:
            bg_margin_str = "_" + str(bg_margin) + "marg"
        else:
            bg_margin_str = ""

        filename = alt_output + write_filename + "_" + str(patch_width) + "x" + str(patch_height) + "patches_" + trainvaltest + multi_bg_str + bg_margin_str + ".hdf5"

        save_file = h5py.File(filename, 'w')

        if bg_mode == 'three':
            save_file.attrs["num_bgs"] = 3
        elif bg_mode == 'one':
            save_file.attrs["num_bgs"] = 1
        elif bg_mode == 'all':
            save_file.attrs["num_bgs"] = num_boundaries + 1
        elif bg_mode == 'extra':
            save_file.attrs["num_bgs"] = num_boundaries * 2 + 1

        save_file.attrs["image_width"] = patch_width
        save_file.attrs["image_height"] = patch_height

        save_file.attrs["patch_col_inc_bounds"] = np.array([patch_col_range[0], patch_col_range[-1]])

        save_file.attrs["type"] = np.array("patch", dtype='S100')
    else:
        if dim_ordering == 'channels_last':

            if len(images.shape) < 4:
                images = np.expand_dims(images, axis=-1)

            if labels is not None:
                labels = np.expand_dims(labels, axis=-1)
            if patch_labels is not None:
                patch_labels = np.expand_dims(patch_labels, axis=-1)
        elif dim_ordering == 'channels_first':
            if len(images.shape) < 4:
                images = np.expand_dims(images, axis=-3)

            if labels is not None:
                labels = np.expand_dims(labels, axis=-3)
            if patch_labels is not None:
                patch_labels = np.expand_dims(patch_labels, axis=-3)

        if patch_labels is not None:
            multi_bg_str = "_" + bg_mode
        else:
            multi_bg_str = ""

        filename = alt_output + write_filename + "_fullsize_" + trainvaltest + multi_bg_str + ".hdf5"
        save_file = h5py.File(filename, 'w')

        save_file.attrs["image_width"] = images.shape[-3]
        save_file.attrs["image_height"] = images.shape[-2]

        if patch_labels is not None:
            patch_labels_dset = save_file.create_dataset("patch_labels", patch_labels.shape, dtype='uint8')
            patch_labels_dset[:] = patch_labels

        save_file.attrs["type"] = np.array("fullsize", dtype='S100')

        if segs is not None:
            seg_dset = save_file.create_dataset("segs", segs.shape, dtype='uint16')
            seg_dset[:] = segs

        if fullsize_class_names is not None:
            fullsize_class_names_dset = save_file.create_dataset("fullsize_class_names", fullsize_class_names.shape,
                                                                 dtype='S100')
            fullsize_class_names_dset[:] = fullsize_class_names

    save_file.attrs["num_channels"] = num_channels
    save_file.attrs["dim_ordering"] = np.array(dim_ordering, dtype='S100')
    save_file.attrs["dim_names"] = np.array(dim_names, dtype='S100')

    if boundary_names is not None:
        boundary_names_dset = save_file.create_dataset("boundary_names", boundary_names.shape, dtype='S100')
        boundary_names_dset[:] = boundary_names

    if area_names is not None:
        area_names_dset = save_file.create_dataset("area_names", area_names.shape, dtype='S100')
        area_names_dset[:] = area_names

    if patch_class_names is not None:
        patch_class_names_dset = save_file.create_dataset("patch_class_names", patch_class_names.shape, dtype='S100')
        patch_class_names_dset[:] = patch_class_names

    if image_names is not None:
        image_names_dset = save_file.create_dataset("image_names", image_names.shape, dtype='S100')
        image_names_dset[:] = image_names

    save_file.attrs["name"] = np.array(write_filename, dtype='S100')

    save_file.attrs["num_boundaries"] = num_boundaries
    save_file.attrs["num_areas"] = num_areas

    save_file.attrs["set"] = np.array(trainvaltest, dtype='S100')

    image_dset = save_file.create_dataset("images", images.shape, dtype='uint8')
    image_dset[:] = images

    if labels is not None:
        label_dset = save_file.create_dataset("labels", labels.shape, dtype='uint8')
        label_dset[:] = labels

    end_construct_time = time.time()
    save_file.attrs["construct_time"] = end_construct_time - start_construct_time

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H:%M:%S")

    save_file.attrs["timestamp"] = np.array(timestamp, dtype='S100')

    save_file.close()

    return filename


def create_all_patch_labels(images, segs, bg_mode='single', bg_margin=0, bg_splits=None):
    all_patch_labels = []

    for i in range(images.shape[0]):
        patch_labels = create_patch_labels(images[i], segs[i], bg_mode=bg_mode, bg_margin=bg_margin, bg_splits=bg_splits)
        all_patch_labels.append(patch_labels)

    all_patch_labels = np.array(all_patch_labels)

    return all_patch_labels


def create_patch_labels(image, segs, bg_mode='single', bg_margin=0, bg_splits=None):
    image_width = image.shape[0]
    image_height = image.shape[1]
    num_boundaries = len(segs)

    patch_labels = np.zeros((image_width, image_height))

    if bg_mode == 'single':
        class_label = 1

        for boundary_ind in range(num_boundaries):
            for col in range(image_width):
                seg_val = segs[boundary_ind, col]
                if not np.isnan(seg_val) and not seg_val == 0:
                    patch_labels[col, seg_val] = class_label

            class_label += 1
    elif bg_mode == 'extra':
        class_label = 0

        for boundary_ind in range(num_boundaries):
            # class_label = 2

            for col in range(image_width):
                seg_val = segs[boundary_ind, col]
                if not np.isnan(seg_val) and not seg_val == 0:
                    patch_labels[col, seg_val] = class_label

            class_label += 1

        for boundary_ind in range(num_boundaries):
            # class_label = 0
            # adjacent backgrounds
            for col in range(image_width):
                seg_val = segs[boundary_ind, col]
                if not np.isnan(seg_val) and not seg_val == 0:
                    for i in range(1, bg_margin + 1):
                        patch_labels[col, seg_val - i] = class_label
                        patch_labels[col, seg_val + i] = class_label

            class_label += 1

        for layer_ind in range(num_boundaries + 1):
            # class_label = 1
            # layer backgrounds
            for col in range(image_width):
                if layer_ind == 0 and not np.isnan(segs[layer_ind, col]) and segs[layer_ind, col] != 0:
                    patch_labels[col, :segs[layer_ind, col] - bg_margin] = class_label
                elif layer_ind == num_boundaries and not np.isnan(segs[layer_ind - 1, col]) and segs[layer_ind - 1, col] != 0:
                    patch_labels[col, segs[layer_ind - 1, col] + bg_margin:] = class_label
                elif not np.isnan(segs[layer_ind - 1, col]) and segs[layer_ind - 1, col] != 0 and not np.isnan(segs[layer_ind, col]) and segs[layer_ind, col] != 0:
                    patch_labels[col, segs[layer_ind - 1, col] + bg_margin:segs[layer_ind, col] - bg_margin] = class_label

            class_label += 1

    return patch_labels


def construct_patches_whole_image(image, patch_labels, patch_size):
    """Construct patches (image + class label) for a given image, labels and patch size. Patches
    are created centred at all pixels within the image. The image is symmetrically zero padded before patches are
        constructed.
        _________

        image: image to construct patches for. Shape: (width, height).
        _________

        patch_labels: array with class labels for each patch corresponding to the image.
        Each entry in this array corresponds exactly with a pixel in the image. Shape: (width, height)
        _________

        patch_size: tuple with patch shape of the form (width, height)
        _________

        Returns: Numpy array of patches and corresponding labels:

        patches: shape (image width * image height, patch width, patch height)

        labels: shape (image width * image height,)

        Coordinates of patch at index i may be found by:

        (col, row) = (i % image width, i / image width)
        _________
        """
    start_patch_time = time.time()

    patch_width = patch_size[0]
    patch_height = patch_size[1]

    img_width = image.shape[0]
    img_height = image.shape[1]

    image = pad_patch_image(image, patch_size)

    patches = np.zeros((img_width * img_height, patch_width, patch_height, 1), dtype='uint8')
    labels = np.zeros((img_width * img_height, 1), dtype='uint8')

    for row in range(img_height):
        for col in range(img_width):
            patch = construct_patch(image, col, row, patch_size)

            patches[row * img_width + col, :, :] = patch
            labels[row * img_width + col] = patch_labels[col, row]

    end_patch_time = time.time()
    patch_time = end_patch_time - start_patch_time

    return patches, labels, patch_time


def construct_patch(image, x, y, patch_size):
    """Constructs a patch for the specified image centred at column x and row y with specified patch size
        _________

        image: image to construct a patch for. Shape: (width, height)
        _________

        x, y: column and row respectively of top left pixel of the patch
        _________

        patch_size: size of the patch with shape: (width, height)
        _________

        Returns: patch with shape (patch width, patch height)
        _________
        """

    patch_width = patch_size[0]
    patch_height = patch_size[1]

    start_y = y
    end_y = (y + patch_height)
    start_x = x
    end_x = (x + patch_width)
    patch = image[start_x:end_x, start_y:end_y]

    return patch


def sample_all_training_patches(images, segs, col_range, patch_size, bg_mode='single', bg_margin=0, bg_splits=None):
    all_patches = []
    all_labels = []

    for i in range(images.shape[0]):
        image_patches, image_patch_labels = sample_training_patches(images[i], segs[i], col_range, patch_size, bg_mode=bg_mode, bg_margin=bg_margin, bg_splits=bg_splits)

        for j in range(len(image_patches)):
            all_patches.append(image_patches[j])
            all_labels.append(image_patch_labels[j])

    all_patches = np.array(all_patches)
    all_labels = np.array(all_labels)

    return all_patches, all_labels


def sample_training_patches(image, segs, col_range, patch_size, bg_mode='single', bg_margin=0, bg_splits=None):
    # multi_bg = False --> sample background patches across whole image for 1 class
    # multi_bg = True --> sample background patches for 3 classes (above top boundary), between top and bottom boundary
    # and below bottom boundary

    num_boundaries = len(segs)

    image_width = image.shape[0]
    image_height = image.shape[1]

    patches = []
    labels = []

    # patches = np.zeros((image_width, num_classes, patch_width, patch_height))
    # labels = np.zeros((image_width, num_classes))
    # invalid = np.zeros((image_width, num_classes))

    image = pad_patch_image(image, patch_size)

    for col in range(image_width):
        if bg_mode == 'single':
            class_label = 1
        elif bg_mode == 'three':
            class_label = 3
        elif bg_mode == 'all':
            class_label = num_boundaries + 1
        elif bg_mode == 'extra':
            class_label = num_boundaries * 2 + 1
        elif bg_mode == 'super':
            class_label = num_boundaries + sum(bg_splits)

        for boundary_ind in range(num_boundaries):
            seg_val = segs[boundary_ind, col]
            if col in col_range:
                patches.append(construct_patch(image, col, seg_val, patch_size))
                labels.append(class_label)

            class_label += 1

        if col in col_range:
            if bg_mode == 'single':
                bg_ind = choose_bg_ind(col, segs, 0, image_height)
                patches.append(construct_patch(image, col, bg_ind, patch_size))
                labels.append(0)
            elif bg_mode == 'three':
                bg_ind_upper = choose_bg_ind(col, segs, 0, segs[0, col] - bg_margin)
                patches.append(construct_patch(image, col, bg_ind_upper, patch_size))
                labels.append(0)

                bg_ind_mid = choose_bg_ind(col, segs, segs[0, col] - bg_margin, segs[-1, col] + bg_margin)
                patches.append(construct_patch(image, col, bg_ind_mid, patch_size))
                labels.append(1)

                bg_ind_lower = choose_bg_ind(col, segs, segs[-1, col] + bg_margin, image_height)
                patches.append(construct_patch(image, col, bg_ind_lower, patch_size))
                labels.append(2)
            elif bg_mode == 'all':
                for i in range(num_boundaries + 1):
                    if i == 0:
                        bg_ind = choose_bg_ind(col, segs, 0, segs[i, col])
                    elif i == num_boundaries:
                        bg_ind = choose_bg_ind(col, segs, segs[-1, col] + 1, image_height)
                    else:
                        bg_ind = choose_bg_ind(col, segs, segs[i - 1, col] + 1, segs[i, col])

                    patches.append(construct_patch(image, col, bg_ind, patch_size))
                    labels.append(i)
            elif bg_mode == 'extra':
                for i in range(num_boundaries):
                    bg_ind_1 = choose_bg_ind(col, segs, segs[i, col] - bg_margin, segs[i, col])
                    bg_ind_2 = choose_bg_ind(col, segs, segs[i, col] + 1, segs[i, col] + bg_margin)

                    bg_ind_props = [bg_ind_1, bg_ind_2]

                    bg_ind = np.random.choice(bg_ind_props)

                    patches.append(construct_patch(image, col, bg_ind, patch_size))
                    labels.append(i)

                for i in range(num_boundaries + 1):
                    if i == 0:
                        bg_ind = choose_bg_ind(col, segs, 0, segs[i, col] - bg_margin)
                    elif i == num_boundaries:
                        bg_ind = choose_bg_ind(col, segs, segs[-1, col] + bg_margin, image_height)
                    else:
                        bg_ind = choose_bg_ind(col, segs, segs[i - 1, col] + bg_margin, segs[i, col] - bg_margin)

                    patches.append(construct_patch(image, col, bg_ind, patch_size))
                    labels.append(num_boundaries + i)
            elif bg_mode == 'super':
                for i in range(num_boundaries):
                    bg_ind_1 = choose_bg_ind(col, segs, segs[i, col] - bg_margin, segs[i, col])
                    bg_ind_2 = choose_bg_ind(col, segs, segs[i, col] + 1, segs[i, col] + bg_margin)

                    bg_ind_props = [bg_ind_1, bg_ind_2]

                    bg_ind = np.random.choice(bg_ind_props)

                    patches.append(construct_patch(image, col, bg_ind, patch_size))
                    labels.append(i)

                for i in range(num_boundaries + 1):
                    if i == 0:
                        total_height = segs[i, col] - bg_margin
                        split_step = int(total_height / bg_splits[i])
                        for j in range(bg_splits[i]):
                            bg_ind = int(choose_bg_ind(col, segs, split_step * j, split_step * (j + 1)))
                            patches.append(construct_patch(image, col, bg_ind, patch_size))
                            labels.append(num_boundaries + sum(bg_splits[:i]) + j)
                    elif i == num_boundaries:
                        total_height = image_height - (segs[-1, col] + bg_margin)
                        split_step = int(total_height / bg_splits[i])
                        for j in range(bg_splits[i]):
                            bg_ind = int(choose_bg_ind(col, segs, (segs[-1, col] + bg_margin) + split_step * j, (segs[-1, col] + bg_margin) + split_step * (j + 1)))
                            patches.append(construct_patch(image, col, bg_ind, patch_size))
                            labels.append(num_boundaries + sum(bg_splits[:i]) + j)
                    else:
                        total_height = (segs[i, col] - bg_margin) - (segs[i - 1, col] + bg_margin)
                        split_step = int(total_height / bg_splits[i])
                        for j in range(bg_splits[i]):
                            bg_ind = int(choose_bg_ind(col, segs, (segs[i - 1, col] + bg_margin) + split_step * j, (segs[i - 1, col] + bg_margin) + split_step * (j + 1)))
                            patches.append(construct_patch(image, col, bg_ind, patch_size))
                            labels.append(num_boundaries + sum(bg_splits[:i]) + j)

    return patches, labels


def choose_bg_ind(col, segs, bg_ind_min, bg_ind_max):
    num_boundaries = len(segs)

    invalids = []

    for boundary_ind in range(num_boundaries):
        seg_val = segs[boundary_ind, col]
        invalids.append(seg_val)

    valid_bg = False
    bg_ind = -1

    while not valid_bg:
        if bg_ind_max - bg_ind_min > 0:
            bg_ind = bg_ind_min + np.random.randint(bg_ind_max - bg_ind_min)
        else:
            bg_ind = bg_ind_min

        if bg_ind == bg_ind_min or bg_ind not in invalids:
            valid_bg = True

    return bg_ind


def pad_patch_image(image, patch_size):
    patch_width = patch_size[0]
    patch_height = patch_size[1]

    if len(image.shape) == 3:
        pad_image = np.pad(image, (
            (int(np.ceil(patch_width / 2.0)), int(np.ceil(patch_width / 2.0))),
            (int(np.ceil(patch_height / 2.0)), int(np.ceil(patch_height / 2.0))), (0, 0)), 'constant')
    elif len(image.shape) == 2:
        pad_image = np.pad(image, (
            (int(np.ceil(patch_width / 2.0)), int(np.ceil(patch_width / 2.0))),
            (int(np.ceil(patch_height / 2.0)), int(np.ceil(patch_height / 2.0)))), 'constant')

    return pad_image


def create_all_area_masks(images, segs):
    all_masks = []

    for i in range(images.shape[0]):
        masks = create_area_mask(images[i], segs[i])
        all_masks.append(masks)

    all_masks = np.array(all_masks)

    return all_masks


# note that each area does not include the pixels of the boundary of which it ends, the boundaries belong to the
# first pixel of their corresponding regions (in a top to bottom sense)
def create_area_mask(image, segs):
    if image.ndim == 3:
        if K.image_data_format() == 'channels_last':
            mask_shape = image.shape[:-1]

        elif K.image_data_format() == 'channels_first':
            mask_shape = image.shape[1:]
    else:
        mask_shape = image.shape

    mask = np.zeros(mask_shape, dtype='uint8')
    image_width = mask_shape[0]
    image_height = mask_shape[1]

    if image.ndim == 3:
        if K.image_data_format() == 'channels_last':
            mask = np.expand_dims(mask, axis=-1)
        elif K.image_data_format() == 'channels_first':
            mask = np.expand_dims(mask, axis=0)

    segs = np.array(segs)

    for col in range(image_width):
        for seg_ind in range(len(segs)):
            seg = segs[seg_ind, col]
            if np.isnan(seg) or seg == 0:
                # can't use this segmentation as is
                # try and replace it
                found_rep = False
                for rep_ind in range(seg_ind + 1, len(segs)):
                    rep_seg = segs[rep_ind, col]
                    if not np.isnan(rep_seg) and not rep_seg == 0:
                        found_rep = True
                        segs[seg_ind, col] = rep_seg
                        break

                if found_rep is False:
                    segs[seg_ind, col] = image_height

        for seg_ind in range(len(segs)):
            cur_seg = segs[seg_ind, col]

            if seg_ind == 0:
                # first boundary
                mask[col, 0:cur_seg] = seg_ind
            else:
                # intermediate boundaries
                prev_seg = segs[seg_ind - 1, col]
                mask[col, prev_seg:cur_seg] = seg_ind

        # final boundaries
        mask[col, segs[len(segs) - 1, col]:] = len(segs)

    return mask


def mask_optic_nerve(mask, seg, onh):
    onh = np.squeeze(onh)
    seg = np.squeeze(seg)

    print(seg.shape)

    for x in range(onh[0], onh[1]):
        mask[x, :seg[0][x]] = 0
        mask[x, seg[0][x]:] = np.max(mask)

    return mask


def flatten_image_boundary(image, boundary, poly=False):
    image = np.array(image)
    num_cols = boundary.shape[0]

    offsets = []
    if poly is True:
        poly_coef = np.polyfit(np.arange(num_cols), boundary, deg=2)

        new_boundary = []

        for i in range(num_cols):
            new_boundary.append(poly_coef[0] * i ** 2 + poly_coef[1] * i + poly_coef[2])

        b_max = np.max(new_boundary)

        new_boundary = np.array(new_boundary)

        for i in range(num_cols):
            boundary_pos = new_boundary[i]
            offset = int(b_max - boundary_pos)
            offsets.append(offset)
            image[i, :, :] = np.roll(image[i, :, :], shift=offset, axis=0)

        flatten_boundary = new_boundary
    else:
        b_max = np.max(boundary)

        for i in range(num_cols):
            boundary_pos = boundary[i]
            offset = b_max - boundary_pos
            offsets.append(offset)
            image[i, :, :] = np.roll(image[i, :, :], shift=offset, axis=0)

        flatten_boundary = boundary

    return [image, np.asarray(offsets), np.asarray(flatten_boundary)]


def roll_image_offset(image, offset):
    image = np.array(image)
    num_cols = offset.shape[0]
    for i in range(num_cols):
        shift = offset[i]
        image[i, :] = np.roll(image[i, :], shift=shift, axis=0)

    return image

