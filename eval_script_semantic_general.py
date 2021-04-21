import eval_helper
import save_parameters
import augmentation as aug
import custom_losses
import custom_metrics
from keras.models import load_model
import image_database as imdb
import dataset_construction
import parameters
from keras.utils import to_categorical
import h5py

TEST_DATA_NAME = "mytestdata"   # can choose a name if desired
DATASET_FILE = h5py.File("example_data.hdf5", 'r')

# segs are true boundary positions for each image

# images numpy array should be of the shape: (number of images, image width, image height, 1)
# segs array should be of the shape: (number of images, NUM_CLASSES - 1, width)
# image names array should be a list of strings with length = number of images

# if there are segs (boundary position truths) return them as the second argument, otherwise pass None

# for each image by passing the image and its correspondings segs as arguments
def load_testing_data():
    # FILL IN THIS FUNCTION TO LOAD YOUR DATA
    test_images = DATASET_FILE['test_images'][:]
    test_segs = DATASET_FILE['test_segs'][:]
    test_image_names = ['image_1', 'image_2', 'image_3']

    return test_images, test_segs, test_image_names


test_images, test_segs, test_image_names = load_testing_data()

test_labels = dataset_construction.create_all_area_masks(test_images, test_segs)

NUM_CLASSES = test_segs.shape[1] + 1

test_labels = to_categorical(test_labels, NUM_CLASSES)

# boundary names should be a list of strings with length = NUM_CLASSES - 1
# class names should be a list of strings with length = NUM_CLASSES
AREA_NAMES = ["area_" + str(i) for i in range(NUM_CLASSES)]
BOUNDARY_NAMES = ["boundary_" + str(i) for i in range(NUM_CLASSES - 1)]
BATCH_SIZE = 1  # DO NOT MODIFY
GSGRAD = 1
CUSTOM_OBJECTS = dict(list(custom_losses.custom_loss_objects.items()) +
                      list(custom_metrics.custom_metric_objects.items()))

eval_imdb = imdb.ImageDatabase(images=test_images, labels=test_labels, segs=test_segs, image_names=test_image_names,
                               boundary_names=BOUNDARY_NAMES, area_names=AREA_NAMES,
                               fullsize_class_names=AREA_NAMES, num_classes=NUM_CLASSES, name=TEST_DATA_NAME, filename=TEST_DATA_NAME, mode_type='fullsize')

network_folder = parameters.RESULTS_LOCATION + "\\2021-04-21 12_28_48 U-net exampledata\\" # name of network folder for which to evaluate model
model_name = "model_epoch635.hdf5"   # name of model file inside network folder to evaluate

loaded_model = load_model(network_folder + "/" + model_name, custom_objects=CUSTOM_OBJECTS)

aug_fn_arg = (aug.no_aug, {})

eval_helper.evaluate_network(eval_imdb, model_name, network_folder,
                             BATCH_SIZE, save_parameters.SaveParameters(pngimages=True, raw_image=True, raw_labels=True, temp_extra=True, boundary_maps=True, area_maps=True, comb_area_maps=True, seg_plot=True),
                             gsgrad=GSGRAD, aug_fn_arg=aug_fn_arg, eval_mode='both', boundaries=True, boundary_errors=True, dice_errors=True, col_error_range=None, normalise_input=True, transpose=False)

