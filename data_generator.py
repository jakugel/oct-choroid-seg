import numpy as np
import keras
from math import floor


class BatchGenerator:
    """Class to generate batches of images and their corresponding label to be used for fit_generator (training)
    or predict_generator (evaluation)
        _________

        images: array or .hdf5 dataset of all images to be used. Shape: (number of images, width, height)
        _________

        labels: array or .hdf5 dataset of all labels to be used. Shape: (number of images, width, height)
        _________

        batch_size: size of the batch for neural network to process
        _________

        aug_fn_args: tuple of two-tuples containing augmentation function and argument pairs
        _________

        aug_mode: mode to use for augmentation

        none: no augmentations -> will just use what is in the images and labels arrays as is
        one: for each image, one augmentation will be picked from the list of possible augmentation functions
            chosen based on probabilities in aug_probs.
        all: for each image, all augmentations will be performed creating a new separate image for each

        note that for patch mode: augs are applied to the full size images before being broken into patches
        _________

        aug_probs: probabilities used for selecting augmentations in 'one' mode. Should be values between 0 and 1
        which add to 1.
        _________

        aug_fly: whether or not to perform all augmentations at the very start or to perform them each time the
        image is required.
        _________

        shuffle: whether or not to shuffle the order of the images at the start as well as at the end of each epoch
        _________

        """
    def __init__(self, imdb, batch_size, aug_fn_args, aug_mode, aug_probs, aug_fly, shuffle=True, transpose=False,
                 normalise=True, ram_load=1):

        self.shuffle = shuffle              # whether to shuffle the order that images are iterated
        self.transpose = transpose          # whether to swap rows and columns of batches
        self.normalise = normalise
        self.batch_counter = 0              # number of batches generated in the current epoch
        self.batch_size = batch_size        # number of samples in a batch
        self.full_counter = 0               # used to track which full size image we are up to
        self.aug_counter = 0                # used to track which augmentation index we are up to (for aug_mode='all')

        self.imdb = imdb

        self.aug_fn_args = aug_fn_args
        self.aug_mode = aug_mode
        self.aug_probs = aug_probs
        self.aug_fly = aug_fly

        self.ram_load = ram_load

        self.total_full_images = self.imdb.num_images

        self.total_raw_samples = self.total_full_images        # total raw samples (w/out augs)

        self.labels_shape = self.imdb.labels_shape

        if self.aug_mode == 'none':
            self.total_samples = self.total_raw_samples
            self.total_augs = 0
        if self.aug_mode == 'all':
            # want to combine all augmentations
            self.total_augs = len(self.aug_fn_args)
            self.total_samples = self.total_raw_samples * self.total_augs   # total samples (including augmentations)
        elif aug_mode == 'one':
            self.total_augs = len(self.aug_fn_args)
            self.total_samples = self.total_raw_samples

        # create shape to be used to create the batch labels array

        self.batch_labels_shape = list(self.labels_shape)
        self.batch_labels_shape[0] = self.batch_size
        self.batch_labels_shape = tuple(self.batch_labels_shape)

        if self.aug_fly is False and self.aug_mode != 'none':
            if self.ram_load == 0:
                print("Incompatible parameter selection: ")
                exit(1)

            # don't augment on the fly so generate samples now
            self.aug_images, self.aug_labels = self.setup_augnofly_data()

        self.sample_shuffle = np.arange(self.total_full_images)

        self.num_batches = int(floor(1.0 * self.total_samples / self.batch_size))
        self.handle_epoch_end()

    def setup_augnofly_data(self):
        """Setup augmented data to be used when aug_fly=False.
            _________

            Returns:

            (1) array of images is created with
            shape: (total full images, total number of augs, image width, image height, num_channels).

            (2) array of labels is created with
            shape: (total full images, total number of augs, image width, image height, num_channels). (semantic)
            or
            shape: (total full images, total number of augs). (patch based imdb)
            _________
            """

        aug_labels_shape = list(self.labels_shape)
        aug_labels_shape[0] = self.total_full_images
        aug_labels_shape.insert(1, self.total_augs)
        aug_labels_shape = tuple(aug_labels_shape)

        aug_images = np.zeros((self.total_full_images, self.total_augs, self.imdb.image_width,
                               self.imdb.image_height, self.imdb.num_channels), dtype='uint8')
        aug_labels = np.zeros(aug_labels_shape, dtype='uint8')

        for i in range(self.total_full_images):
            for j in range(self.total_augs):
                aug_fn = self.aug_fn_args[j][0]
                aug_arg = self.aug_fn_args[j][1]
                image = self.imdb.get_image(i)
                label = self.imdb.get_label(i)
                aug_images[i, j], aug_labels[i, j], _, _, _ = aug_fn(image, label, None, aug_arg, sample_ind=i, set=self.imdb.set)

        return aug_images, aug_labels

    def get_aug_fly(self, sample_ind):
        """Get next sample where augmentation needs to be generated on the fly.
            _________

            Returns:

            aug_image: next sample. shape: (image width, image height)

            aug_label: next label. shape: (image width, image height) (semantic) or shape: (1,) (patch based)
            _________
            """
        raw_image = self.imdb.get_image(sample_ind)
        raw_label = self.imdb.get_label(sample_ind)
        raw_seg = self.imdb.get_seg(sample_ind)

        if self.aug_mode == 'all':
            # perform each augmentation (current augmentation indicated by aug_ind)
            aug_fn_arg = self.aug_fn_args[self.aug_counter]

            aug_fn = aug_fn_arg[0]
            aug_arg = aug_fn_arg[1]
            aug_image, aug_label, _, _, _ = aug_fn(raw_image, raw_label, raw_seg, aug_arg, sample_ind=sample_ind, set=self.imdb.set)  # apply augmentation

            self.aug_counter += 1  # move to the next augmentation ready for next time
            if self.aug_counter == self.total_augs:
                self.aug_counter = 0  # reset the aug_ind, we are done with them all for this particular image
                self.full_counter += 1  # move to the next full image as we have no more augs to do for the current
        elif self.aug_mode == 'one':
            # choose single augmentation for replacement based on probabilities
            aug_fn_arg_ind = np.random.choice(np.arange(self.total_augs), p=self.aug_probs)

            aug_fn_arg = self.aug_fn_args[aug_fn_arg_ind]

            aug_fn = aug_fn_arg[0]
            aug_arg = aug_fn_arg[1]

            aug_image, aug_label, _, _, _ = aug_fn(raw_image, raw_label, raw_seg, aug_arg, sample_ind=sample_ind, set=self.imdb.set)  # apply augmentation

            self.full_counter += 1  # just the single random augmentation so move to the next raw image
        else:
            # no augmentation: just use the raw image and label as is
            aug_image = raw_image
            aug_label = raw_label
            self.full_counter += 1   # move to the next image

        return aug_image, aug_label

    def get_aug_nofly(self, sample_ind):
        """Get next sample from pre-constructed augmentation data.
            _________

            Returns:

            aug_image: next sample. shape: (image width, image height)

            aug_label: next label. shape: (image width, image height) (semantic) or shape: (1,) (patch based)
            _________
            """
        raw_image = self.imdb.get_image(sample_ind)
        raw_label = self.imdb.get_label(sample_ind)

        if self.aug_mode == 'all':
            # all augmentations are used
            aug_image = self.aug_images[sample_ind, self.aug_counter]
            aug_label = self.aug_labels[sample_ind, self.aug_counter]

            self.aug_counter += 1
            if self.aug_counter == self.total_augs:
                self.aug_counter = 0
                self.full_counter += 1
        elif self.aug_mode == 'one':
            # just one random augmentation is used
            aug_ind_choice = np.random.choice(np.arange(self.total_augs), p=self.aug_probs)

            aug_image = self.aug_images[sample_ind, aug_ind_choice]
            aug_label = self.aug_labels[sample_ind, aug_ind_choice]

            self.full_counter += 1
        else:
            # no augmentation: just use raw image
            aug_image = raw_image
            aug_label = raw_label
            self.full_counter += 1

        return aug_image, aug_label

    def get_batch_list(self):
        """Generate next batch of data
            _________

            Returns: [batch_images, batch_labels]

            batch_images: set of images. shape: (batch_size, image width, image height)

            batch_labels: set of labels. shape: (batch_size, image width, image height) (semantic)
            or shape: (batch_size,) (patch based)
            _________
            """
        batch_images = np.zeros((self.batch_size, self.imdb.image_width, self.imdb.image_height,
                                 self.imdb.num_channels), dtype='float32')

        batch_labels = np.zeros(self.batch_labels_shape)

        cur_sample_counter = 0    # sample we are up to in the current batch

        while cur_sample_counter < self.batch_size:
            # store images and labels here

            full_sample_ind = self.sample_shuffle[self.full_counter]

            if self.aug_fly is True:
                # need to perform the augmentations on the fly as they haven't been done already
                batch_images[cur_sample_counter], batch_labels[cur_sample_counter] = \
                    self.get_aug_fly(full_sample_ind)
            elif self.aug_fly is False:
                # augmentations have been done beforehand and are stored, so just retrieve the appropriate one
                batch_image, batch_label = self.get_aug_nofly(full_sample_ind)
                batch_images[cur_sample_counter], batch_labels[cur_sample_counter] = batch_image, batch_label

            cur_sample_counter += 1

            if self.full_counter == self.total_full_images:
                self.full_counter = 0

        # end of the batch
        # we have done another batch
        self.batch_counter += 1

        if self.batch_counter == self.num_batches:
            self.batch_counter = 0

        # normalise batch images before passing to network
        if self.normalise is True:
            batch_images /= 255

        if self.transpose is True:
            batch_images = np.transpose(batch_images, axes=(0, 2, 1, 3))
            if len(batch_labels.shape) == 4:
                # labels are masks
                batch_labels = np.transpose(batch_labels, axes=(0, 2, 1, 3))


        return [batch_images, batch_labels]

    def handle_epoch_end(self):
        """Handle the end of the epoch by resetting the augmentation index, and the counters for number of batches
            and number of images. If shuffle is enabled, shuffle the order of the raw images for the next epoch.
            _________

            Returns:

            aug_image: next sample. shape: (image width, image height)

            aug_label: next label. shape: (image width, image height) (semantic) or shape: (1,) (patch based)
            _________
            """
        self.batch_counter = 0
        self.full_counter = 0
        self.aug_counter = 0

        if self.shuffle:
            np.random.seed()
            x = np.arange(self.total_raw_samples)
            s = np.arange(x.shape[0])
            np.random.shuffle(s)
            self.sample_shuffle = self.sample_shuffle[s]


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras: see BatchGenerator for parameter details"""
    def __init__(self, imdb, batch_size, aug_fn_args, aug_mode, aug_probs, aug_fly, shuffle=False, transpose=False,
                 normalise=True, ram_load=1):
        self.batch_gen = BatchGenerator(imdb=imdb, batch_size=batch_size,
                                        aug_fn_args=aug_fn_args, aug_mode=aug_mode,
                                        aug_probs=aug_probs, aug_fly=aug_fly, shuffle=shuffle, transpose=transpose,
                                        normalise=normalise, ram_load=ram_load)


    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.batch_gen.num_batches

    def __getitem__(self, index):
        """Generate one batch of data"""

        # Generate data
        X, y = self.__data_generation()

        return X, y

    def on_epoch_end(self):
        """Runs when a training epoch ends"""
        self.batch_gen.handle_epoch_end()

    def __data_generation(self):
        """Generates data to be used for a batch"""
        [X, y] = self.batch_gen.get_batch_list()

        return X, y
