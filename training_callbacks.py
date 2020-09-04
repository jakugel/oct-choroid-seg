import keras
import time
import os
import h5py


class SaveEpochInfo(keras.callbacks.Callback):
    def __init__(self, save_folder, train_params, train_imdb):
        super(SaveEpochInfo, self).__init__()
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.epoch_times = []
        self.start_epoch_time = -1
        self.start_time = -1
        self.train_time = -1
        self.f = None
        self.ax1 = None
        self.ax2 = None
        if type(train_params.metric) is not str:
            self.acc_name = train_params.metric.__name__
        else:
            self.acc_name = train_params.metric

        if type(train_params.loss) is not str:
            self.loss_name = train_params.loss.__name__
        else:
            self.loss_name = train_params.loss

        self.network_name = train_params.network_model[1]
        self.save_folder = save_folder
        self.plotpath = save_folder + '/performance_plot.png'
        self.num_epochs = train_params.epochs

    def on_train_begin(self, logs=None):
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.epoch_times = []
        self.start_time = time.time()

    def on_train_end(self, logs=None):
        self.train_time = time.time() - self.start_time

    def on_epoch_begin(self, epoch, logs=None):
        self.start_epoch_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.train_losses.append(logs.get('loss'))
        self.train_accs.append(logs.get(self.acc_name))
        self.val_losses.append(logs.get('val_loss'))
        self.val_accs.append(logs.get('val_' + self.acc_name))
        self.epoch_times.append(time.time() - self.start_epoch_time)

        stats_epoch_file = h5py.File(self.save_folder + "/stats_epoch{:02d}.hdf5".format(epoch + 1), 'w')
        stats_epoch_file["train_acc"] = self.train_accs
        stats_epoch_file["val_acc"] = self.val_accs
        stats_epoch_file["train_loss"] = self.train_losses
        stats_epoch_file["val_loss"] = self.val_losses
        stats_epoch_file["epoch_time"] = self.epoch_times
        stats_epoch_file.close()

        prev_stats_file = self.save_folder + "/stats_epoch{:02d}.hdf5".format(epoch)

        if os.path.isfile(prev_stats_file):
            try:
                os.remove(prev_stats_file)
            except Exception:
                pass
