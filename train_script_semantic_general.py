import training
import training_parameters as tparams
import keras.optimizers
import image_database as imdb
import semantic_network_models as sem_models
import custom_losses
import custom_metrics
import dataset_construction
from keras.utils import to_categorical
import augmentation as aug
import h5py

keras.backend.set_image_data_format('channels_last')

INPUT_CHANNELS = 1
DATASET_NAME = "exampledata"     # can choose a name if desired
DATASET_FILE = h5py.File("example_data.hdf5", 'r')

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

train_labels = dataset_construction.create_all_area_masks(train_images, train_segs)
val_labels = dataset_construction.create_all_area_masks(val_images, val_segs)

NUM_CLASSES = train_segs.shape[1] + 1

train_labels = to_categorical(train_labels, NUM_CLASSES)
val_labels = to_categorical(val_labels, NUM_CLASSES)

train_imdb = imdb.ImageDatabase(images=train_images, labels=train_labels, name=DATASET_NAME, filename=DATASET_NAME, mode_type='fullsize', num_classes=NUM_CLASSES)
val_imdb = imdb.ImageDatabase(images=val_images, labels=val_labels, name=DATASET_NAME, filename=DATASET_NAME, mode_type='fullsize', num_classes=NUM_CLASSES)

# models from the "Automatic choroidal segmentation in OCT images using supervised deep learning methods" paper (currently excluding RNN bottleneck and Combined)
model_residual = sem_models.resnet(8, 4, 2, 1, (3, 3), (2, 2), input_channels=INPUT_CHANNELS, output_channels=NUM_CLASSES)
model_standard = sem_models.unet(8, 4, 2, (3, 3), (2, 2), input_channels=INPUT_CHANNELS, output_channels=NUM_CLASSES)
model_sSE = sem_models.unet(8, 4, 2, (3, 3), (2, 2), input_channels=INPUT_CHANNELS, output_channels=NUM_CLASSES, se='sSE')
model_cSE = sem_models.unet(8, 4, 2, (3, 3), (2, 2), input_channels=INPUT_CHANNELS, output_channels=NUM_CLASSES, se='cSE')
model_scSE = sem_models.unet(8, 4, 2, (3, 3), (2, 2), input_channels=INPUT_CHANNELS, output_channels=NUM_CLASSES, se='scSE')

opt_con = keras.optimizers.Adam
opt_params = {}     # default params
loss = custom_losses.dice_loss
metric = custom_metrics.dice_coef
epochs = 10000
batch_size = 3

aug_fn_args = [(aug.no_aug, {}), (aug.flip_aug, {'flip_type': 'left-right'})]

aug_mode = 'one'
aug_probs = (0.5, 0.5)
aug_val = False
aug_fly = True

train_params = tparams.TrainingParams(model_standard, opt_con, opt_params, loss, metric, epochs, batch_size, model_save_best=True, aug_fn_args=aug_fn_args, aug_mode=aug_mode,
                                      aug_probs=aug_probs, aug_val=aug_val, aug_fly=aug_fly)

training.train_network(train_imdb, val_imdb, train_params)
