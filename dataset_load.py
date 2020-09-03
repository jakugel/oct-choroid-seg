import h5py
import numpy as np
import image_database as image_db
import keras.backend as K
import keras
import parameters


def load_dataset_file_ram(filename, dim_inds=None, image_range=None, padding=None, reps_channels=None, alt_output=parameters, imagenet=False, categ_labels=True):
    open_filename = alt_output + filename
    loadfile = h5py.File(open_filename, 'r')

    type = loadfile.attrs['type'].decode('utf-8')
    set = loadfile.attrs['set'].decode('utf-8')
    dim_ordering = loadfile.attrs["dim_ordering"]
    dim_names = np.char.decode(loadfile.attrs["dim_names"], 'utf-8')

    if "boundary_names" in loadfile.keys():
        boundary_names = np.char.decode(loadfile["boundary_names"][:], 'utf-8')
    else:
        boundary_names = None

    area_names = np.char.decode(loadfile["area_names"][:], 'utf-8')

    if "patch_class_names" in loadfile.keys():
        patch_class_names = np.char.decode(loadfile["patch_class_names"][:], 'utf-8')
    else:
        patch_class_names = None

    name = loadfile.attrs["name"].decode('utf-8')

    if "labels" in loadfile.keys():
        labels = loadfile["labels"][:]
    else:
        labels = None

    images = loadfile["images"][:]

    if "image_names" in loadfile.keys():
        image_names = np.char.decode(loadfile["image_names"][:], 'utf-8')
    else:
        image_names = None

    # load images
    if dim_inds is not None:
        dim_slices = []
        for dim_ind in dim_inds:
            if dim_ind is None:
                dim_slices.append(slice(None))
            elif len(dim_ind) == 2:
                dim_slices.append(slice(dim_ind[0], dim_ind[1]))
            elif len(dim_ind) == 3:
                dim_slices.append(slice(dim_ind[0], dim_ind[1], dim_ind[2]))

        images = images[tuple(dim_slices)]
        labels = labels[tuple(dim_slices)]
        image_names = image_names[tuple(dim_slices)]        # these don't work as intended

    imgs_shape = images.shape
    # input images shape (..., width, height, channels)
    images = np.reshape(images, newshape=(int(np.prod(imgs_shape[:-3])), imgs_shape[-3], imgs_shape[-2],
                                          imgs_shape[-1]))

    if image_names is not None:
        image_names = np.ndarray.flatten(image_names)

    # resulting images shape (num_images, width, height, channels) or (num_images, channels, width, height)

    # adjust images to dim ordering
    if K.image_dim_ordering() == 'tf' and dim_ordering == 'channels_first':
        images = np.transpose(images, axes=(0, 2, 3, 1))

    elif K.image_dim_ordering() == 'th' and dim_ordering == 'channels_last':
        images = np.transpose(images, axes=(0, 3, 1, 2))

    if padding is not None:
        images = np.pad(images, padding, mode='constant')

    if reps_channels is not None:
        if dim_ordering == 'channels_first':
            images = np.repeat(images, reps_channels, axis=1)
        elif dim_ordering == 'channels_last':
            images = np.repeat(images, reps_channels, axis=3)

    if type == 'fullsize':
        if dim_inds is not None:
            if "patch_labels" in loadfile.keys():
                patch_labels = loadfile["patch_labels"][tuple(dim_slices)]
            else:
                patch_labels = None

            if "segs" in loadfile.keys():
                segs = loadfile["segs"][tuple(dim_slices)]
            else:
                segs = None
        else:
            if "patch_labels" in loadfile.keys():
                patch_labels = loadfile["patch_labels"][:]
            else:
                patch_labels = None

            if "segs" in loadfile.keys():
                segs = loadfile["segs"][:]
            else:
                segs = None

        if segs is not None:
            segs_shape = segs.shape
            segs = np.reshape(segs, newshape=(int(np.prod(segs_shape[:-2])), segs_shape[-2], segs_shape[-1]))

        if labels is not None:
            lbls_shape = labels.shape
            labels = np.reshape(labels, newshape=(np.prod(imgs_shape[:-3]), lbls_shape[-3],
                                                  lbls_shape[-2], lbls_shape[-1]))
        if patch_labels is not None:
            patch_lbls_shape = patch_labels.shape
            patch_labels = np.reshape(patch_labels, newshape=(np.prod(imgs_shape[:-3]), patch_lbls_shape[-3],
                                                              patch_lbls_shape[-2], patch_lbls_shape[-1]))
        # adjust labels to dim ordering
        if K.image_dim_ordering() == 'tf' and dim_ordering == 'channels_first':
            if labels is not None:
                labels = np.transpose(labels, axes=(0, 2, 3, 1))
            if patch_labels is not None:
                patch_labels = np.transpose(patch_labels, axes=(0, 2, 3, 1))

        elif K.image_dim_ordering() == 'th' and dim_ordering == 'channels_last':
            if labels is not None:
                labels = np.transpose(labels, axes=(0, 3, 1, 2))
            if patch_labels is not None:
                patch_labels = np.transpose(patch_labels, axes=(0, 3, 1, 2))

        if padding is not None:
            if labels is not None:
                labels = np.pad(labels, padding, mode='constant')
            if patch_labels is not None:
                patch_labels = np.pad(patch_labels, padding, mode='constant')
            if segs is not None:
                segs[segs != 0] = segs[segs != 0] + padding[2][0]
                segs = np.pad(segs, ((0, 0), (0, 0), (padding[1][0], padding[1][1])), mode='constant')

        fullsize_class_names = np.char.decode(loadfile["fullsize_class_names"][:], 'utf-8')

        num_classes = len(fullsize_class_names)
        if labels is not None and categ_labels is True:
            labels = keras.utils.to_categorical(labels, num_classes=num_classes)
    elif type == 'patch':
        if labels is not None:
            lbls_shape = labels.shape
            labels = np.reshape(labels, newshape=(np.prod(lbls_shape[:-1]), lbls_shape[-1]))

        patch_labels = None
        segs = None
        fullsize_class_names = None

        if patch_class_names is not None:
            num_classes = len(patch_class_names)

            if labels is not None:
                labels = keras.utils.to_categorical(labels, num_classes=num_classes)

    imdb = image_db.ImageDatabase(images, labels, patch_labels=patch_labels, segs=segs, image_names=image_names,
                                  boundary_names=boundary_names, area_names=area_names,
                                  patch_class_names=patch_class_names, fullsize_class_names=fullsize_class_names,
                                  filename=filename, name=name, num_classes=num_classes,
                                  dim_inds=dim_inds, image_range=image_range, dim_names=dim_names, mode_type=type, set=set,
                                  padding=padding, ram_load=1, reps_channels=reps_channels)

    loadfile.close()

    return imdb


def load_dataset_file_disk(filename, image_range=None, padding=None, reps_channels=None, alt_output=parameters.DATA_LOCATION, imagenet=False):
    open_filename = alt_output + filename
    loadfile = h5py.File(open_filename, 'r')

    type = loadfile.attrs['type'].decode('utf-8')
    set = loadfile.attrs['set'].decode('utf-8')
    dim_ordering = np.char.decode(loadfile.attrs["dim_ordering"], 'utf-8')
    dim_names = np.char.decode(loadfile.attrs["dim_names"], 'utf-8')

    if "boundary_names" in loadfile.keys():
        boundary_names = np.char.decode(loadfile["boundary_names"][:], 'utf-8')
    else:
        boundary_names = None

    area_names = np.char.decode(loadfile["area_names"][:], 'utf-8')

    if "patch_class_names" in loadfile.keys():
        patch_class_names = np.char.decode(loadfile["patch_class_names"][:], 'utf-8')
    else:
        patch_class_names = None

    name = loadfile.attrs["name"].decode('utf-8')

    # don't load into memory yet
    if "labels" in loadfile.keys():
        labels = False
    else:
        labels = None

    # don't load into memory yet
    images = False

    if "image_names" in loadfile.keys():
        image_names = np.char.decode(loadfile["image_names"][:], 'utf-8')
    else:
        image_names = None

    # dim inds ignored if not loading into memory
    dim_inds = None


    # no reshaping will be performed, file must be in an existing suitable format

    # resulting images shape (num_images, width, height, channels) or (num_images, channels, width, height)

    # padding and dim ordering performed on the fly in image database

    if type == 'fullsize':
        # don't load in yet

        if "patch_labels" in loadfile.keys():
            # load and slice as required later
            patch_labels = False
        else:
            patch_labels = None

        if "segs" in loadfile.keys():
            segs = False
        else:
            segs = None

    # no reshaping will be performed, file must be in correct format already

    #     padding and dim ordering performed on the fly in image database

        fullsize_class_names = np.char.decode(loadfile["fullsize_class_names"][:], 'utf-8')

        num_classes = len(fullsize_class_names)

        # one hot encoding will have to be performed within generator
    elif type == 'patch':
        # no reshaping will be performed, file must already exist in correct format (image, width, height, channels)

        patch_labels = None
        segs = None
        fullsize_class_names = None

        if patch_class_names is not None:
            num_classes = len(patch_class_names)
        else:
            num_classes = None

            # one hot encoding will have to occur within generator

    imdb = image_db.ImageDatabase(images, labels, patch_labels=patch_labels, segs=segs, image_names=image_names,
                                  boundary_names=boundary_names, area_names=area_names,
                                  patch_class_names=patch_class_names, fullsize_class_names=fullsize_class_names,
                                  filename=filename, name=name, num_classes=num_classes,
                                  dim_inds=dim_inds, image_range=image_range, dim_names=dim_names, mode_type=type, set=set,
                                  padding=padding, dim_ordering=dim_ordering, ram_load=0, reps_channels=reps_channels,
                                  imagenet=imagenet, open_filename=open_filename)

    loadfile.close()

    return imdb


def load_dataset_file(filename, ram_load=1, dim_inds=None, image_range=None, padding=None, reps_channels=None, alt_output=parameters.DATA_LOCATION, imagenet=False):
    if ram_load == 1:
        imdb = load_dataset_file_ram(filename, dim_inds=dim_inds, image_range=image_range, padding=padding, reps_channels=reps_channels, alt_output=alt_output, imagenet=imagenet)
    else:
        imdb = load_dataset_file_disk(filename, image_range=image_range, padding=padding, reps_channels=reps_channels, alt_output=alt_output, imagenet=imagenet)

    return imdb


def combine_patch_imdbs(imdb1, imdb2):
    new_images = np.concatenate((imdb1.images, imdb2.images), axis=0)
    new_labels = np.concatenate((imdb1.labels, imdb2.labels), axis=0)
    new_imdb = image_db.ImageDatabase(new_images, new_labels, patch_labels=None, segs=None, image_names=None,
                                  boundary_names=imdb1.boundary_names, area_names=imdb1.area_names,
                                  patch_class_names=imdb1.patch_class_names, fullsize_class_names=imdb1.fullsize_class_names,
                                  filename=imdb1.filename + "_comb", name=imdb1.name + "_comb", num_classes=imdb1.num_classes,
                                  dim_inds=None, image_range=None, dim_names=imdb1.dim_names, mode_type=imdb1.type,
                                  padding=imdb1.padding)

    return new_imdb


def combine_patch_imdbs_disk(imdb1, imdb2):
    new_imdb = image_db.ImageDatabase(None, None, patch_labels=None, segs=None, image_names=None,
                                  boundary_names=imdb1.boundary_names, area_names=imdb1.area_names,
                                  patch_class_names=imdb1.patch_class_names, fullsize_class_names=imdb1.fullsize_class_names,
                                  filename=imdb1.filename + "_comb", name=imdb1.name + "_comb", num_classes=imdb1.num_classes,
                                  dim_inds=None, image_range=None, dim_names=imdb1.dim_names, mode_type=imdb1.type,
                                  padding=imdb1.padding, ram_load=0, imdb1=imdb1, imdb2=imdb2)

    return new_imdb


def combine_semantic_imdbs(imdb1, imdb2):
    new_images = np.concatenate((imdb1.images, imdb2.images), axis=0)
    new_labels = np.concatenate((imdb1.labels, imdb2.labels), axis=0)
    new_patch_labels = imdb2.patch_labels   #### HACKED. TODO: FIX
    new_imdb = image_db.ImageDatabase(new_images, new_labels, patch_labels=new_patch_labels, segs=None, image_names=None,
                                  boundary_names=imdb1.boundary_names, area_names=imdb1.area_names,
                                  patch_class_names=imdb1.patch_class_names, fullsize_class_names=imdb1.fullsize_class_names,
                                  filename=imdb1.filename + "_" + imdb2.filename + "_comb", name=imdb1.name + "_" + imdb2.name + "_comb", num_classes=imdb1.num_classes,
                                  dim_inds=None, image_range=None, dim_names=imdb1.dim_names, mode_type=imdb1.type,
                                  padding=imdb1.padding)

    return new_imdb
