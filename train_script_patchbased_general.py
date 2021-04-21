import training
import training_parameters as tparams
import keras.optimizers
import image_database as imdb
import patch_based_network_models as patch_models
import dataset_construction
from keras.layers.cudnn_recurrent import CuDNNLSTM, CuDNNGRU
import h5py

keras.backend.set_image_data_format('channels_last')

INPUT_CHANNELS = 1
DATASET_NAME = "myexampledata"     # can choose a name if desired
DATASET_FILE = h5py.File("example_data.hdf5", 'r')
PATCH_SIZE = (32, 32)       # modify depending on desired patch size

# images numpy array should be of the shape: (number of images, image width, image height, 1)
# segs numpy array should be of the shape: (number of images, number of boundaries, image width)


# fill in this function to load your data for the training set with format/shape given above
def load_training_data():
    # FILL IN THIS FUNCTION TO LOAD YOUR DATA
    train_images = DATASET_FILE['train_images'][:]
    train_segs = DATASET_FILE['train_segs'][:]

    return train_images, train_segs

# fill in this function to load your data for the validation set with format/shape given above
def load_validation_data():
    # FILL IN THIS FUNCTION TO LOAD YOUR DATA
    val_images = DATASET_FILE['val_images'][:]
    val_segs = DATASET_FILE['val_segs'][:]

    return val_images, val_segs


train_images, train_segs = load_training_data()
val_images, val_segs = load_validation_data()

# if you would like to generate patches differently, comment out these lines and write a replacement function
train_patches, train_patch_labels = dataset_construction.sample_all_training_patches(train_images, train_segs, range(0, train_images.shape[1]), PATCH_SIZE)
val_patches, val_patch_labels = dataset_construction.sample_all_training_patches(val_images, val_segs, range(0, val_images.shape[1]), PATCH_SIZE)

NUM_CLASSES = train_segs.shape[1] + 1

train_patch_labels = keras.utils.to_categorical(train_patch_labels, num_classes=NUM_CLASSES)
val_patch_labels = keras.utils.to_categorical(val_patch_labels, num_classes=NUM_CLASSES)

train_patch_imdb = imdb.ImageDatabase(images=train_patches, labels=train_patch_labels, name=DATASET_NAME, filename=DATASET_NAME, mode_type='patch', num_classes=NUM_CLASSES)
val_patch_imdb = imdb.ImageDatabase(images=val_patches, labels=val_patch_labels, name=DATASET_NAME, filename=DATASET_NAME, mode_type='patch', num_classes=NUM_CLASSES)

# patch-based models from the "Automatic choroidal segmentation in OCT images using supervised deep learning methods" paper
model_cifar = patch_models.cifar_cnn(NUM_CLASSES, train_patches.shape[1], train_patches.shape[2])
model_complex = patch_models.complex_cnn(NUM_CLASSES, train_patches.shape[1], train_patches.shape[2])
model_rnn = patch_models.rnn_stack(4, ('ver', 'hor', 'ver', 'hor'), (True, True, True, True),
                               (CuDNNGRU, CuDNNGRU, CuDNNGRU, CuDNNGRU), (0.25, 0.25, 0.25, 0.25), (1, 1, 2, 2), (1, 1, 2, 2),
                               (16, 16, 16, 16), False, 0, INPUT_CHANNELS, train_patches.shape[1], train_patches.shape[2],
                               NUM_CLASSES)

opt_con = keras.optimizers.Adam
opt_params = {}     # default params
loss = keras.losses.categorical_crossentropy
metric = keras.metrics.categorical_accuracy
epochs = 100
batch_size = 1024

train_params = tparams.TrainingParams(model_complex, opt_con, opt_params, loss, metric, epochs, batch_size, model_save_best=True)

training.train_network(train_patch_imdb, val_patch_imdb, train_params)
