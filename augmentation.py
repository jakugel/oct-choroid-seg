import numpy as np
import time
from keras import backend as K


def augment_dataset(images, masks, segs, aug_fn_arg):
    start_augment_time = time.time()

    num_images = len(images)
    aug_fn = aug_fn_arg[0]
    aug_arg = aug_fn_arg[1]

    augmented_images = np.zeros_like(images)
    augmented_masks = np.zeros_like(masks)
    augmented_segs = np.zeros_like(segs)

    for i in range(num_images):
        image = images[i, :, :]
        mask = masks[i, :, :]
        seg = segs[i, :, :]
        augmented_images[i, :, :], augmented_masks[i, :, :], augmented_segs[i, :, :], _, _ \
            = aug_fn(image, mask, seg, aug_arg)

    aug_desc = aug_fn(None, None, None, aug_arg, True)

    end_augment_time = time.time()
    total_aug_time = end_augment_time - start_augment_time

    return [augmented_images, augmented_masks, augmented_segs, aug_desc, total_aug_time]


def no_aug(image, mask, seg, aug_args, desc_only=False, sample_ind=None, set=None):
    desc = "no aug"
    if desc_only is False:
        return image, mask, seg, desc, 0
    else:
        return desc


def flip_aug(image, mask, seg, aug_args, desc_only=False, sample_ind=None, set=None):
    start_augment_time = time.time()

    flip_type = aug_args['flip_type']

    if flip_type == 'up-down':
        axis = 1
    elif flip_type == 'left-right':
        axis = 0

    aug_desc = "flip aug: " + flip_type

    if desc_only is False:
        aug_image = np.flip(image, axis=axis)
        if mask is not None:
            aug_mask = np.flip(mask, axis=axis)
        else:
            aug_mask = None
        if seg is not None:
            aug_seg = np.flip(seg, axis=axis)
        else:
            aug_seg = None

        end_augment_time = time.time()
        augment_time = end_augment_time - start_augment_time

        return aug_image, aug_mask, aug_seg, aug_desc, augment_time
    else:
        return aug_desc


def normalize(x):
    x = np.asarray(x)
    return (x - x.min()) / (np.ptp(x))
