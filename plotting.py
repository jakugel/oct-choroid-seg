from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np

from keras import backend as K

predict_colours = ['#4285f4', '#db4437', '#f4b400', '#0f9d58', '#ff6d00', '#46bdc6', '#ab30c4', '#fde8ff']
truth_colours = ['#2b5790', '#7a261e', '#9b7200', '#085630', '#8e3d00', '#26686d', '#5f1a6d', '#f266ff']
region_colours = ['#fde8ff', '#4285f4', '#db4437', '#f4b400', '#0f9d58', '#ff6d00', '#46bdc6', '#ab30c4', '#0e0d5e']
region_cmap = colors.ListedColormap(region_colours)


def save_cur_trainval_plot(acc_name, loss_name, network_name,
                           num_epochs, epoch, train_accs, val_accs, train_losses, val_losses, filename):
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=False, sharey=False)
    f.set_size_inches(15, 15)
    # f.tight_layout()
    ax1.grid()
    ax2.grid()
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set(ylabel=loss_name, xlim=(1, num_epochs))
    ax1.set(ylabel=acc_name)

    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.set(ylabel=loss_name, xlim=(1, num_epochs))
    plt.xlabel("Epoch")

    best_val_acc = np.max(val_accs) * 100
    best_train_acc = np.max(train_accs) * 100
    best_val_loss = np.min(val_losses)
    best_train_loss = np.min(train_losses)

    best_val_acc_ep = np.argmax(val_accs) + 1
    best_train_acc_ep = np.argmax(train_accs) + 1
    best_val_loss_ep = np.argmin(val_losses) + 1
    best_train_loss_ep = np.argmin(train_losses) + 1

    train_acc_str = 'Best training ' + acc_name + ": {:.2f} at epoch {:d}".format(best_train_acc, best_train_acc_ep)
    val_acc_str = 'Best validation ' + acc_name + ": {:.2f} at epoch {:d}".format(best_val_acc, best_val_acc_ep)
    train_loss_str = 'Best training ' + loss_name + ": {:.4f} at epoch {:d}".format(best_train_loss,
                                                                                      best_train_loss_ep)
    val_loss_str = 'Best validation ' + loss_name + ": {:.4f} at epoch {:d}".format(best_val_loss, best_val_loss_ep)

    f.suptitle('Network: ' + network_name + '\n\n' + train_acc_str + " | " + val_acc_str + '\n\n' +
               train_loss_str + " | " + val_loss_str, fontsize=14, fontweight='bold')

    ax1.plot(list(range(1, epoch + 2)), train_accs[:epoch + 1], color='#4286f4')
    ax1.plot(list(range(1, epoch + 2)), val_accs[:epoch + 1], color='#b20e0e')
    ax1.legend(["Train Acc", "Val Acc"])

    ax1.plot(list(range(1, epoch + 2)), train_accs[:epoch + 1], '.', color='#4286f4')
    ax1.plot(list(range(1, epoch + 2)), val_accs[:epoch + 1], '.', color='#b20e0e')

    ax2.plot(list(range(1, epoch + 2)), train_losses[:epoch + 1], color='#4286f4')
    ax2.plot(list(range(1, epoch + 2)), val_losses[:epoch + 1], color='#b20e0e')
    ax2.legend(["Train Loss", "Val Loss"])

    ax2.plot(list(range(1, epoch + 2)), train_losses[:epoch + 1], '.', color='#4286f4')
    ax2.plot(list(range(1, epoch + 2)), val_losses[:epoch + 1], '.', color='#b20e0e')

    try:
        plt.savefig(filename)
    except Exception:
        pass

    plt.close()


def setup_image_plot(image, cmap, vmin=None, vmax=None):
    if image.ndim == 3:
        if K.image_data_format() == 'channels_last':
            image_width, image_height = image.shape[:-1]
            if image.shape[2] == 1:
                image = image[:, :, 0]
        elif K.image_data_format == 'channels_first':
            image_width, image_height = image.shape[1:]
            if image.shape[0] == 1:
                image = image[0, :, :]

    elif image.ndim == 2:
        image_width, image_height = image.shape

    fig = plt.figure(num=None, figsize=(image_width / 100, image_height / 100), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    if cmap is None:
        plt.imshow(np.transpose(image, (1, 0, 2)), vmin=vmin, vmax=vmax)
    else:
        plt.imshow(np.transpose(image), cmap=cmap, vmin=vmin, vmax=vmax)


def save_image_plot(image, filename, cmap, vmin=None, vmax=None):
    setup_image_plot(image, cmap, vmin, vmax)

    plt.savefig(filename)
    plt.close()


def save_image_plot_crop(image, filename, cmap, crop_bounds, vmin=None, vmax=None):
    image = np.array(image[crop_bounds[0][0]:crop_bounds[0][1], crop_bounds[1][0]:crop_bounds[1][1]])
    setup_image_plot(image, cmap, vmin, vmax)

    plt.savefig(filename)
    plt.close()


def save_segmentation_plot(image, image_cmap, filename, truths, predictions, column_range=None, linewidth=4.0, color=None):
    setup_image_plot(image, image_cmap, vmin=0, vmax=255)

    if truths is not None:
        num_boundaries = truths.shape[0]
        if column_range is None:
            column_range = range(0, truths.shape[1])
    else:
        num_boundaries = predictions.shape[0]
        if column_range is None:
            column_range = range(0, predictions.shape[1])

    for boundary_ind in range(num_boundaries):
        if truths is not None:
            truths = truths.astype('float64')
            truths[truths == 0] = np.nan
            if color is None:
                plt.plot(column_range, truths[boundary_ind, column_range[0]:column_range[-1] + 1], linewidth=linewidth,
                         color=truth_colours[boundary_ind])
            else:
                plt.plot(column_range, truths[boundary_ind, column_range[0]:column_range[-1] + 1], linewidth=linewidth,
                         color=color)

    for boundary_ind in range(num_boundaries):
        if predictions is not None:
            predictions = predictions.astype('float64')
            predictions[predictions == 0] = np.nan

            if color is None:
                plt.plot(column_range, predictions[boundary_ind, column_range[0]:column_range[-1] + 1], linestyle=':',
                         linewidth=linewidth, color=predict_colours[boundary_ind])
            else:
                plt.plot(column_range, predictions[boundary_ind, column_range[0]:column_range[-1] + 1], linestyle=':',
                         linewidth=linewidth, color=color)

    plt.savefig(filename)
    plt.close()

