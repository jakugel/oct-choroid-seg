import augmentation as aug


class TrainingParams:
    """Parameters for training a network.

        """
    def __init__(self, network_model, opt_con, opt_params, loss, metric, epochs, batch_size,
                 aug_fn_args=((aug.no_aug, {}),), aug_mode='none', aug_probs=(), aug_fly=False, aug_val=True,
                 shuffle=True, model_save_best=True, model_save_monitor=('val_acc', 'max'), normalise=True,
                 use_gen=True, use_tensorboard=False, class_weight=None):
        self.network_model = network_model
        self.opt_con = opt_con
        self.opt_params = opt_params
        self.loss = loss
        self.metric = metric
        self.epochs = epochs
        self.batch_size = batch_size
        self.aug_fn_args = aug_fn_args
        self.aug_mode = aug_mode
        self.aug_probs = aug_probs
        self.aug_fly = aug_fly
        self.aug_val = aug_val
        self.shuffle = shuffle
        self.normalise = normalise
        self.model_save_best = model_save_best
        self.model_save_monitor = model_save_monitor
        self.use_gen = use_gen
        self.use_tensorboard = use_tensorboard
        self.class_weight = class_weight

        if type(loss) is not str:
            self.loss_name = self.loss.__name__
        else:
            self.loss_name = self.loss

        if type(metric) is not str:
            self.metric_name = self.metric.__name__
        else:
            self.metric_name = self.metric

        if self.model_save_monitor[0] == 'val_acc':
            self.model_save_monitor = ['val_' + self.metric_name, model_save_monitor[1]]
