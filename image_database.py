import numpy as np

import keras.backend as K
import keras
import h5py

from keras.applications import imagenet_utils


class ImageDatabase:
    def __init__(self, images, labels, patch_labels=None, segs=None, image_names=None, boundary_names=None,
                 area_names=None, patch_class_names=None, fullsize_class_names=None,
                 image_range=None, filename=None, name=None, num_classes=None,
                 dim_inds=None, dim_names=None, mode_type=None, set=None, padding=None, dim_ordering=None, ram_load=1, reps_channels=None,
                 imagenet=False, open_filename=None, imdb1=None, imdb2=None):

        self.images = images
        self.labels = labels
        self.patch_labels = patch_labels
        self.image_names = image_names
        self.boundary_names = boundary_names
        self.area_names = area_names
        self.patch_class_names = patch_class_names
        self.fullsize_class_names = fullsize_class_names
        self.segs = segs
        if filename is not None:
            self.filename = filename.split("/")[-1]
        else:
            self.filename = filename
        self.name = name
        self.num_classes = num_classes
        self.ram_load = ram_load

        self.open_filename = open_filename

        self.reps_channels = reps_channels

        self.imdb1 = imdb1
        self.imdb2 = imdb2

        if self.images is None:
            self.num_images = self.imdb1.num_images + self.imdb2.num_images
            self.image_width = self.imdb1.image_width
            self.image_height = self.imdb1.image_height

            if self.reps_channels is not None:
                self.num_channels = self.imdb1.num_channels * self.reps_channels
            else:
                self.num_channels = self.imdb1.num_channels

            self.labels_shape = self.imdb1.labels_shape
        else:
            if ram_load == 0:
                loadfile = h5py.File(self.open_filename, 'r')
                fileimages = loadfile["images"]
                self.num_images = fileimages.shape[0]
                self.image_width = fileimages.shape[1]
                self.image_height = fileimages.shape[2]

                if self.reps_channels is not None:
                    self.num_channels = fileimages.shape[3] * self.reps_channels
                else:
                    self.num_channels = fileimages.shape[3]

                filelabels = loadfile["labels"]

                if self.labels is not None:
                    self.labels_shape = list(filelabels.shape)
                loadfile.close()
            else:
                self.num_images = images.shape[0]
                self.image_width = images.shape[1]
                self.image_height = images.shape[2]
                self.num_channels = images.shape[3]

                if self.labels is not None:
                    self.labels_shape = list(self.labels.shape)

        if ram_load == 0 and padding is not None:
            self.image_width = self.image_width + padding[1][0] + padding[1][1]
            self.image_height = self.image_height + padding[2][0] + padding[2][1]

        self.dim_inds = dim_inds
        self.dim_names = dim_names
        self.type = mode_type
        self.set = set
        self.padding = padding

        if self.dim_inds is not None:
            self.ind_name = ""
            dim_count = 0
            for dim_name in dim_names:
                if self.dim_inds[dim_count] is not None:
                    self.ind_name += "_" + dim_name + ": " + str(self.dim_inds[dim_count])
        else:
            self.ind_name = "_allinds"

        self.image_range = image_range

        if self.image_range is None:
            self.image_range = range(self.num_images)

        self.start_image_ind = self.image_range[0]
        self.end_image_ind = self.image_range[-1] + 1
        self.num_images = self.end_image_ind - self.start_image_ind

        self.dim_ordering = dim_ordering

        if self.labels is not None:
            if mode_type == 'fullsize':
                if self.dim_ordering == 'channels_last':
                    self.labels_shape[3] = self.num_classes
                elif self.dim_ordering == 'channels_first':
                    self.labels_shape[1] = self.num_classes

                if self.ram_load == 0 and self.padding is not None:
                    if self.dim_ordering == 'channels_last':
                        self.labels_shape[1] = self.labels_shape[1] + self.padding[1][0] + self.padding[1][1]
                        self.labels_shape[2] = self.labels_shape[2] + self.padding[2][0] + self.padding[2][1]
                    elif self.dim_ordering == 'channels_last':
                        self.labels_shape[2] = self.labels_shape[2] + self.padding[2][0] + self.padding[2][1]
                        self.labels_shape[3] = self.labels_shape[3] + self.padding[3][0] + self.padding[3][1]
            elif mode_type == 'patch':
                print(self.labels_shape)
                self.labels_shape[1] = self.num_classes

            self.labels_shape = tuple(self.labels_shape)

        self.imagenet = imagenet

        if self.ram_load == 1 and self.imagenet is True:
            self.images = imagenet_utils.preprocess_input(self.images, data_format=None, mode='tf')

    def get_image_names_range(self):
        if self.image_names is not None:
            return self.image_names[self.start_image_ind:self.end_image_ind]
        else:
            return None

    def get_images_subrange_disk(self, a, b):
        loadfile = h5py.File(self.open_filename, 'r')
        fileimages = loadfile["images"]

        if b is not None and a is not None:
            images = fileimages[self.start_image_ind + a:self.start_image_ind + a + b]
        elif a is not None:
            images = fileimages[self.start_image_ind + a:self.end_image_ind]
        elif b is not None:
            images = fileimages[self.start_image_ind:self.end_image_ind + b]

        if K.image_data_format() == 'channels_last' and self.dim_ordering == 'channels_first':
            images = np.transpose(images, axes=(0, 2, 3, 1))

        elif K.image_data_format() == 'channels_first' and self.dim_ordering == 'channels_last':
            images = np.transpose(images, axes=(0, 3, 1, 2))

        if self.padding is not None:
            images = np.pad(images, self.padding, mode='constant')

        if self.reps_channels is not None:
            if self.dim_ordering == 'channels_first':
                images = np.repeat(images, self.reps_channels, axis=1)
            elif self.dim_ordering == 'channels_last':
                images = np.repeat(images, self.reps_channels, axis=3)

        if self.imagenet is True:
            images = imagenet_utils.preprocess_input(images, data_format=None, mode='tf')

        loadfile.close()

        return images

    def get_labels_subrange_disk(self, a, b):
        loadfile = h5py.File(self.open_filename, 'r')
        filelabels = loadfile["labels"]

        if b is not None and a is not None:
            labels = filelabels[self.start_image_ind + a:self.start_image_ind + a + b]
        elif a is not None:
            labels = filelabels[self.start_image_ind + a:self.end_image_ind]
        elif b is not None:
            labels = filelabels[self.start_image_ind:self.start_image_ind + b]

        if self.type == "fullsize":
            if K.image_data_format() == 'channels_last' and self.dim_ordering == 'channels_first':
                labels = np.transpose(labels, axes=(0, 2, 3, 1))

            elif K.image_data_format() == 'channels_first' and self.dim_ordering == 'channels_last':
                labels = np.transpose(labels, axes=(0, 3, 1, 2))

            if self.padding is not None:
                labels = np.pad(labels, self.padding, mode='constant')

        labels = keras.utils.to_categorical(labels, num_classes=self.num_classes)

        loadfile.close()

        return labels

    def get_images_range(self):
        if self.images is None:
            num_images1 = self.imdb1.num_images
            num_images2 = self.imdb2.num_images

            if self.start_image_ind < num_images1 and self.end_image_ind < num_images1:
                images = self.imdb1.get_images_subrange_disk()[self.start_image_ind:self.end_image_ind]
            elif self.start_image_ind < num_images1 and self.end_image_ind >= num_images1:
                images = np.concatenate(self.imdb1.get_images_subrange_disk(self.start_image_ind, None), self.imdb2.get_images_subrange_disk(None, self.end_image_ind - num_images1), axis=0)
            else:
                images = self.imdb2.get_images_subrange_disk(self.start_image_ind - num_images1, self.end_image_ind - num_images1)

            if K.image_data_format() == 'channels_last' and self.dim_ordering == 'channels_first':
                images = np.transpose(images, axes=(0, 2, 3, 1))

            elif K.image_data_format() == 'channels_first' and self.dim_ordering == 'channels_last':
                images = np.transpose(images, axes=(0, 3, 1, 2))

            if self.padding is not None:
                images = np.pad(images, self.padding, mode='constant')

            if self.reps_channels is not None:
                if self.dim_ordering == 'channels_first':
                    images = np.repeat(images, self.reps_channels, axis=1)
                elif self.dim_ordering == 'channels_last':
                    images = np.repeat(images, self.reps_channels, axis=3)

            if self.imagenet is True:
                images = imagenet_utils.preprocess_input(images, data_format=None, mode='tf')

            return images

        else:
            if self.ram_load == 1:
                return self.images[self.start_image_ind:self.end_image_ind]
            else:
                loadfile = h5py.File(self.open_filename, 'r')
                fileimages = loadfile["images"]

                images = fileimages[self.start_image_ind:self.end_image_ind]

                if K.image_data_format() == 'channels_last' and self.dim_ordering == 'channels_first':
                    images = np.transpose(images, axes=(0, 2, 3, 1))

                elif K.image_data_format() == 'channels_first' and self.dim_ordering == 'channels_last':
                    images = np.transpose(images, axes=(0, 3, 1, 2))

                if self.padding is not None:
                    images = np.pad(images, self.padding, mode='constant')

                if self.reps_channels is not None:
                    if self.dim_ordering == 'channels_first':
                        images = np.repeat(images, self.reps_channels, axis=1)
                    elif self.dim_ordering == 'channels_last':
                        images = np.repeat(images, self.reps_channels, axis=3)

                if self.imagenet is True:
                    images = imagenet_utils.preprocess_input(images, data_format=None, mode='tf')

                loadfile.close()

                return images

    def get_labels_range(self):
        if self.labels is not None:
            if self.images is None:
                num_labels1 = self.imdb1.num_images
                num_labels2 = self.imdb2.num_images
                if self.start_image_ind < num_labels1 and self.end_image_ind < num_labels1:
                    labels = self.imdb1.get_labels_subrange_disk(self.start_image_ind, self.end_image_ind)
                elif self.start_image_ind < num_labels1 and self.end_image_ind >= num_labels1:
                    labels = np.concatenate(self.imdb1.get_labels_subrange_disk(self.start_image_ind, None),
                                            self.imdb2.get_labels_subrange_disk(None, self.end_image_ind - num_labels1), axis=0)
                else:
                    labels = self.imdb2.get_labels_subrange_disk(self.start_image_ind - num_labels1, self.end_image_ind - num_labels1)

                if self.type == "fullsize":
                    if K.image_data_format() == 'channels_last' and self.dim_ordering == 'channels_first':
                        labels = np.transpose(labels, axes=(0, 2, 3, 1))

                    elif K.image_data_format() == 'channels_first' and self.dim_ordering == 'channels_last':
                        labels = np.transpose(labels, axes=(0, 3, 1, 2))

                    if self.padding is not None:
                        labels = np.pad(labels, self.padding, mode='constant')

                labels = keras.utils.to_categorical(labels, num_classes=self.num_classes)

                return labels
            else:
                if self.ram_load == 1:
                    return self.labels[self.start_image_ind:self.end_image_ind]
                else:
                    loadfile = h5py.File(self.open_filename, 'r')
                    filelabels = loadfile["labels"]

                    labels = filelabels[self.start_image_ind:self.end_image_ind]
                    if self.type == "fullsize":
                        if K.image_data_format() == 'channels_last' and self.dim_ordering == 'channels_first':
                                labels = np.transpose(labels, axes=(0, 2, 3, 1))

                        elif K.image_data_format() == 'channels_first' and self.dim_ordering == 'channels_last':
                                labels = np.transpose(labels, axes=(0, 3, 1, 2))

                        if self.padding is not None:
                            labels = np.pad(labels, self.padding, mode='constant')

                    labels = keras.utils.to_categorical(labels, num_classes=self.num_classes)

                    loadfile.close()

                    return labels
        else:
            return None

    def get_patch_labels_range(self):
        if self.patch_labels is not None:
            if self.ram_load == 1:
                return self.patch_labels[self.start_image_ind:self.end_image_ind]
            else:
                loadfile = h5py.File(self.open_filename, 'r')
                filepatchlabels = loadfile["patch_labels"]

                patch_labels = filepatchlabels[self.start_image_ind:self.end_image_ind]
                if K.image_data_format() == 'channels_last' and self.dim_ordering == 'channels_first':
                        patch_labels = np.transpose(patch_labels, axes=(0, 2, 3, 1))
                elif K.image_data_format() == 'channels_first' and self.dim_ordering == 'channels_last':
                        patch_labels = np.transpose(patch_labels, axes=(0, 3, 1, 2))

                if self.padding is not None:
                    patch_labels = np.pad(patch_labels, self.padding, mode='constant')

                loadfile.close()

                return patch_labels
        else:
            return None

    def get_segs_range(self):
        if self.segs is not None:
            if self.ram_load == 1:
                return self.segs[self.start_image_ind:self.end_image_ind]
            else:
                loadfile = h5py.File(self.open_filename, 'r')
                filesegs = loadfile["segs"]

                segs = filesegs[self.start_image_ind:self.end_image_ind]

                if self.padding is not None:
                    segs[segs != 0] = segs[segs != 0] + self.padding[2][0]

                loadfile.close()

                return segs
        else:
            return None

    def get_image_range(self, ind):
        if self.images is None:
            num_images1 = self.imdb1.num_images
            num_images2 = self.imdb2.num_images
            if ind + self.start_image_ind < num_images1:
                image = self.imdb1.get_image_range(ind + self.start_image_ind)
            else:
                image = self.imdb2.get_image_range(ind + self.start_image_ind - num_images1)

            if K.image_data_format() == 'channels_last' and self.dim_ordering == 'channels_first':
                image = np.transpose(image, axes=(1, 2, 0))

            elif K.image_data_format() == 'channels_first' and self.dim_ordering == 'channels_last':
                image = np.transpose(image, axes=(2, 0, 1))

            if self.padding is not None:
                image = np.pad(image, self.padding[1:], mode='constant')

            if self.reps_channels is not None:
                if self.dim_ordering == 'channels_first':
                    image = np.repeat(image, self.reps_channels, axis=1)
                elif self.dim_ordering == 'channels_last':
                    image = np.repeat(image, self.reps_channels, axis=3)

            if self.imagenet is True:
                image = imagenet_utils.preprocess_input(image, data_format=None, mode='tf')

            return image

        else:
            if self.ram_load == 1:
                return self.images[ind + self.start_image_ind]
            else:
                loadfile = h5py.File(self.open_filename, 'r')
                fileimages = loadfile["images"]

                image = fileimages[ind + self.start_image_ind]

                if K.image_data_format() == 'channels_last' and self.dim_ordering == 'channels_first':
                    image = np.transpose(image, axes=(1, 2, 0))

                elif K.image_data_format() == 'channels_first' and self.dim_ordering == 'channels_last':
                    image = np.transpose(image, axes=(2, 0, 1))

                if self.padding is not None:
                    image = np.pad(image, self.padding[1:], mode='constant')

                if self.reps_channels is not None:
                    if self.dim_ordering == 'channels_first':
                        image = np.repeat(image, self.reps_channels, axis=1)
                    elif self.dim_ordering == 'channels_last':
                        image = np.repeat(image, self.reps_channels, axis=3)

                if self.imagenet is True:
                    image = imagenet_utils.preprocess_input(image, data_format=self.dim_ordering, mode='tf')

                loadfile.close()

                return image

    def get_patch_label(self, ind):
        if self.patch_labels is not None:
            if self.ram_load == 1:
                return self.patch_labels[ind]
            else:
                loadfile = h5py.File(self.open_filename, 'r')
                filepatchlabels = loadfile["patch_labels"]

                patch_label = filepatchlabels[ind]
                if K.image_data_format() == 'channels_last' and self.dim_ordering == 'channels_first':
                        patch_label = np.transpose(patch_label, axes=(1, 2, 0))
                elif K.image_data_format() == 'channels_first' and self.dim_ordering == 'channels_last':
                        patch_label = np.transpose(patch_label, axes=(2, 0, 1))

                if self.padding is not None:
                    patch_label = np.pad(patch_label, self.padding[1:], mode='constant')

                loadfile.close()

                return patch_label
        else:
            return None

    def get_patch_label_range(self, ind):
        if self.patch_labels is not None:
            if self.ram_load == 1:
                return self.patch_labels[ind + self.start_image_ind]
            else:
                loadfile = h5py.File(self.open_filename, 'r')
                filepatchlabels = loadfile["patch_labels"]

                patch_label = filepatchlabels[ind + self.start_image_ind]
                if K.image_data_format() == 'channels_last' and self.dim_ordering == 'channels_first':
                    patch_label = np.transpose(patch_label, axes=(1, 2, 0))
                elif K.image_data_format() == 'channels_first' and self.dim_ordering == 'channels_last':
                    patch_label = np.transpose(patch_label, axes=(2, 0, 1))

                if self.padding is not None:
                    patch_label = np.pad(patch_label, self.padding[1:], mode='constant')

                loadfile.close()

                return patch_label
        else:
            return None

    def get_image(self, ind):
        if self.images is None:
            num_images1 = self.imdb1.num_images
            num_images2 = self.imdb2.num_images
            if ind < num_images1:
                image = self.imdb1.get_image(ind)
            else:
                image = self.imdb2.get_image(ind - num_images1)

            if K.image_data_format() == 'channels_last' and self.dim_ordering == 'channels_first':
                image = np.transpose(image, axes=(1, 2, 0))

            elif K.image_data_format() == 'channels_first' and self.dim_ordering == 'channels_last':
                image = np.transpose(image, axes=(2, 0, 1))

            if self.padding is not None:
                image = np.pad(image, self.padding[1:], mode='constant')

            if self.reps_channels is not None:
                if self.dim_ordering == 'channels_first':
                    image = np.repeat(image, self.reps_channels, axis=1)
                elif self.dim_ordering == 'channels_last':
                    image = np.repeat(image, self.reps_channels, axis=3)

            if self.imagenet is True:
                image = imagenet_utils.preprocess_input(image, data_format=None, mode='tf')

            return image
        else:
            if self.ram_load == 1:
                return self.images[ind]
            else:
                loadfile = h5py.File(self.open_filename, 'r')
                fileimages = loadfile["images"]

                image = fileimages[ind]

                if K.image_data_format() == 'channels_last' and self.dim_ordering == 'channels_first':
                    image = np.transpose(image, axes=(1, 2, 0))

                elif K.image_data_format() == 'channels_first' and self.dim_ordering == 'channels_last':
                    image = np.transpose(image, axes=(2, 0, 1))

                if self.padding is not None:
                    print(image.shape)
                    print(self.padding[1:])
                    image = np.pad(image, self.padding[1:], mode='constant')

                if self.reps_channels is not None:
                    if self.dim_ordering == 'channels_first':
                        image = np.repeat(image, self.reps_channels, axis=0)
                    elif self.dim_ordering == 'channels_last':
                        image = np.repeat(image, self.reps_channels, axis=2)

                if self.imagenet is True:
                    image = imagenet_utils.preprocess_input(image, data_format=None, mode='tf')

                loadfile.close()

                return image

    def get_label(self, ind):
        if self.labels is not None:
            if self.images is None:
                num_labels1 = self.imdb1.num_images
                num_labels2 = self.imdb2.num_images
                if ind < num_labels1:
                    label = self.imdb1.get_label(ind)
                else:
                    label = self.imdb2.get_label(ind - num_labels1)

                if self.type == "fullsize":
                    if K.image_data_format() == 'channels_last' and self.dim_ordering == 'channels_first':
                        label = np.transpose(label, axes=(1, 2, 0))
                    elif K.image_data_format() == 'channels_first' and self.dim_ordering == 'channels_last':
                        label = np.transpose(label, axes=(2, 0, 1))

                    if self.padding is not None:
                        label = np.pad(label, self.padding[1:], mode='constant')

                label = keras.utils.to_categorical(label, num_classes=self.num_classes)

                return label
            else:
                if self.ram_load == 1:
                    return self.labels[ind]
                else:
                    loadfile = h5py.File(self.open_filename, 'r')
                    filelabels = loadfile["labels"]

                    label = filelabels[ind]
                    if self.type == "fullsize":
                        if K.image_data_format() == 'channels_last' and self.dim_ordering == 'channels_first':
                                label = np.transpose(label, axes=(1, 2, 0))
                        elif K.image_data_format() == 'channels_first' and self.dim_ordering == 'channels_last':
                                label = np.transpose(label, axes=(2, 0, 1))

                        if self.padding is not None:
                            label = np.pad(label, self.padding[1:], mode='constant')

                    label = keras.utils.to_categorical(label, num_classes=self.num_classes)

                    loadfile.close()

                    return label
        else:
            return None

    def get_label_range(self, ind):
        if self.labels is not None:
            if self.images is None:
                num_labels1 = self.imdb1.num_images
                num_labels2 = self.imdb2.num_images
                if ind + self.image_range < num_labels1:
                    label = self.imdb1.get_label_range(ind + self.image_range)
                else:
                    label = self.imdb2.get_label_range(ind + self.image_range - num_labels1)

                if self.type == "fullsize":
                    if K.image_data_format() == 'channels_last' and self.dim_ordering == 'channels_first':
                        label = np.transpose(label, axes=(1, 2, 0))
                    elif K.image_data_format() == 'channels_first' and self.dim_ordering == 'channels_last':
                        label = np.transpose(label, axes=(2, 0, 1))

                    if self.padding is not None:
                        label = np.pad(label, self.padding[1:], mode='constant')

                label = keras.utils.to_categorical(label, num_classes=self.num_classes)

                return label
            else:
                if self.ram_load == 1:
                    return self.labels[ind + self.start_image_ind]
                else:
                    loadfile = h5py.File(self.open_filename, 'r')
                    filelabels = loadfile["labels"]

                    label = filelabels[ind + self.start_image_ind]
                    if self.type == "fullsize":
                        if K.image_data_format() == 'channels_last' and self.dim_ordering == 'channels_first':
                            label = np.transpose(label, axes=(1, 2, 0))
                        elif K.image_data_format() == 'channels_first' and self.dim_ordering == 'channels_last':
                            label = np.transpose(label, axes=(2, 0, 1))

                        if self.padding is not None:
                            label = np.pad(label, self.padding[1:], mode='constant')

                    label = keras.utils.to_categorical(label, num_classes=self.num_classes)

                    loadfile.close()

                    return label
        else:
            return None

    def get_seg(self, ind):
        if self.segs is not None:
            if self.ram_load == 1:
                return self.segs[ind]
            else:
                loadfile = h5py.File(self.open_filename, 'r')
                filesegs = loadfile["segs"]

                seg = filesegs[ind]

                if self.padding is not None:
                    seg[seg != 0] = seg[seg != 0] + self.padding[2][0]

                loadfile.close()

                return seg
        else:
            return None

    def get_seg_range(self, ind):
        if self.segs is not None:
            if self.ram_load == 1:
                return self.segs[ind + self.start_image_ind]
            else:
                loadfile = h5py.File(self.open_filename, 'r')
                filesegs = loadfile["segs"]

                seg = filesegs[ind + self.start_image_ind]

                if self.padding is not None:
                    seg[seg != 0] = seg[seg != 0] + self.padding[2][0]

                loadfile.close()

                return seg
        else:
            return None

    def get_image_name(self, ind):
        if self.image_names is not None:
            return self.image_names[ind]
        else:
            return None

    def get_image_name_range(self, ind):
        if self.image_names is not None:
            return self.image_names[ind + self.start_image_ind]
        else:
            return None

    def get_boundary_names(self):
        if self.boundary_names is not None:
            return self.boundary_names[:]
        else:
            return None

    def get_boundary_name(self, ind):
        if self.boundary_names is not None:
            return self.boundary_names[ind]
        else:
            return None

    def get_area_names(self):
        if self.area_names is not None:
            return self.area_names[:]
        else:
            return None

    def get_area_name(self, ind):
        if self.area_names is not None:
            return self.area_names[ind]
        else:
            return None

    def get_patch_class_names(self):
        if self.patch_class_names is not None:
            return self.patch_class_names[:]
        else:
            return None

    def get_patch_class_name(self, ind):
        if self.patch_class_names is not None:
            return self.patch_class_names[ind]
        else:
            return None

    def get_fullsize_class_names(self):
        if self.fullsize_class_names is not None:
            return self.fullsize_class_names[:]
        else:
            return None

    def get_fullsize_class_name(self, ind):
        if self.fullsize_class_names is not None:
            return self.fullsize_class_names[ind]
        else:
            return None
