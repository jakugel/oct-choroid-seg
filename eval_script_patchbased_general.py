import eval_helper
import save_parameters
import augmentation as aug
import custom_losses
import custom_metrics
from keras.models import load_model
import image_database as imdb
import dataset_construction
import parameters
import h5py

TEST_DATA_NAME = "myexampledata"   # can choose a name if desired
DATASET_FILE = h5py.File("example_data.hdf5", 'r')

# segs are true boundary positions for each image

# images numpy array should be of the shape: (number of images, image width, image height, 1)
# segs array should be of the shape: (number of images, number of boundaries, width)
# image names array should be a list of strings with length = number of images

def load_testing_data():
    # FILL IN THIS FUNCTION TO LOAD YOUR DATA
    test_images = DATASET_FILE['test_images'][:]
    test_segs = DATASET_FILE['test_segs'][:]
    test_image_names = ['image_1', 'image_2', 'image_3']
    return test_images, test_segs, test_image_names


test_images, test_segs, test_image_names = load_testing_data()
test_patch_labels = dataset_construction.create_all_patch_labels(test_images, test_segs)

NUM_CLASSES = test_segs.shape[1] + 1                 # update for required number of classes

# boundary names should be a list of strings with length = NUM_CLASSES - 1
# class names should be a list of strings with length = NUM_CLASSES
AREA_NAMES = ["area_" + str(i) for i in range(NUM_CLASSES)]
BOUNDARY_NAMES = ["boundary_" + str(i) for i in range(NUM_CLASSES - 1)]
PATCH_CLASS_NAMES = ["BG"]
for i in range(len(BOUNDARY_NAMES)):
    PATCH_CLASS_NAMES.append(BOUNDARY_NAMES[i])

GSGRAD = 1
CUSTOM_OBJECTS = dict(list(custom_losses.custom_loss_objects.items()) +
                      list(custom_metrics.custom_metric_objects.items()))

eval_imdb = imdb.ImageDatabase(images=test_images, labels=None, patch_labels=test_patch_labels, segs=test_segs, image_names=test_image_names,
                               boundary_names=BOUNDARY_NAMES, area_names=AREA_NAMES,
                               fullsize_class_names=AREA_NAMES, patch_class_names=PATCH_CLASS_NAMES, num_classes=NUM_CLASSES, name=TEST_DATA_NAME, filename=TEST_DATA_NAME, mode_type='fullsize')

batch_size = 992    # CURRENTLY THIS NEEDS TO BE CHOSEN AS A VALUE WHICH IS A FACTOR OF THE AREA (IN PIXELS) OF THE FULL IMAGE (i.e. 992 is a factor of a 761856 (1536x496) pixel image [992 x 768 = 761856])
network_folder = parameters.RESULTS_LOCATION + "\\2021-04-21 14_35_20 Complex CNN 32x32 myexampledata\\" # name of network folder for which to evaluate model
model_name = "model_epoch06.hdf5"   # name of model file inside network folder to evaluate

loaded_model = load_model(network_folder + "/" + model_name, custom_objects=CUSTOM_OBJECTS)

aug_fn_arg = (aug.no_aug, {})

eval_helper.evaluate_network(eval_imdb, model_name, network_folder,
                             batch_size, save_parameters.SaveParameters(pngimages=True, raw_image=True, temp_extra=True, boundary_maps=True, area_maps=True, comb_area_maps=True, seg_plot=True),
                             gsgrad=GSGRAD, aug_fn_arg=aug_fn_arg, eval_mode='both', boundaries=True, boundary_errors=True, dice_errors=False, col_error_range=None, normalise_input=True, transpose=False)

