from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.models import Sequential
from keras.layers.cudnn_recurrent import CuDNNLSTM, CuDNNGRU
from keras.layers import Lambda, Bidirectional, Dropout, BatchNormalization, Input, Flatten, Dense, Activation
from keras.backend import int_shape
from keras.models import Model

from keras import backend as K


def cifar_cnn(output_channels, img_width, img_height):
    # initialize model
    model = Sequential()
    if K.image_data_format() == 'channels_last':
        model.add(ZeroPadding2D(padding=((2, 2), (2, 2)), input_shape=(img_width, img_height, 1)))
    else:
        model.add(ZeroPadding2D(padding=((2, 2), (2, 2)), input_shape=(1, img_width, img_height)))

    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation=None))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Activation("relu"))

    model.add(ZeroPadding2D(padding=((2, 2), (2, 2))))
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(ZeroPadding2D(padding=((2, 2), (2, 2))))
    model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    # model.build()
    # model.summary()

    # model(Input((img_width, img_height, 1)))

    # cur_shape = model.get_output_shape_at(-1)
    # cur_shape = model.layers[-1]._keras_shape

    if K.image_data_format() == 'channels_last':
        model.add(Conv2D(64, kernel_size=(int(img_width / 8), int(img_height / 8)), activation='relu'))
    else:
        model.add(Conv2D(64, kernel_size=(int(img_width / 8), int(img_height / 8)), activation='relu'))

    model.add(Flatten())
    model.add(Dense(output_channels, activation='softmax'))

    model_name = "Cifar CNN " + str(img_width) + "x" + str(img_height) + " " + str(output_channels) + "class"

    model_name_short = "Cifar CNN " + str(img_width) + "x" + str(img_height)

    return [model, model_name, model_name_short]


def complex_cnn(output_channels, img_width, img_height):
    model = Sequential()
    if K.image_data_format() == 'channels_last':
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu',
                         input_shape=(img_width, img_height, 1)))
    else:
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu',
                         input_shape=(1, img_width, img_height)))

    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation=None))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation=None))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # model.build()
    # model.summary()
    model(Input((img_width, img_height, 1)))

    cur_shape = model.get_output_shape_at(-1)

    if K.image_data_format() == 'channels_last':
        model.add(Conv2D(128, kernel_size=(cur_shape[1], cur_shape[2]), activation=None))
    else:
        model.add(Conv2D(128, kernel_size=(cur_shape[2], cur_shape[3]), activation=None))

    model.add(Flatten())
    model.add(Dense(output_channels, activation='softmax'))

    model_name = "Complex CNN " + str(img_width) + "x" + str(img_height) + " " + str(output_channels) + "class"

    model_name_short = "Complex CNN " + str(img_width) + "x" + str(img_height)

    return [model, model_name, model_name_short]


# cell_funcs: CuDNNGRU, CuDNNLSTM, GRU, LSTM
# dirs = 'hor': horizontal, 'ver': vertical
# bidirs = True: bidirectional, False: unidirectional
# dense = False: no dense layer at output, True: dense layer at output with size = dense_size
# filters: number of filters per pass
# example: model = rnn_stack(4, ('ver', 'hor', 'ver', 'hor'), (True, True, True, True),
#                                (CuDNNGRU, CuDNNGRU, CuDNNGRU, CuDNNGRU), (0.25, 0.25, 0.25, 0.25), (1, 1, 2, 2), (1, 1, 2, 2),
#                                (16, 16, 16, 16), False, 0, train_imdb.num_channels, train_imdb.image_width, train_imdb.image_height,
#                                train_imdb.num_classes)
def rnn_stack(num_layers, dirs, bidirs, cell_funcs, dropouts, p_widths, p_heights, filters, dense, dense_size,
        input_ch, input_width, input_height, out_classes, output_layers=True):
    # shape: (batchsize, channels, height, width)
    if K.image_data_format() == 'channels_last':
        x = Input(batch_shape=(None, input_width, input_height, input_ch))
        batch_norm_axis = -1
    else:
        x = Input(batch_shape=(None, input_ch, input_width, input_height))
        batch_norm_axis = 1

    data = x

    h = input_height
    w = input_width
    c = input_ch

    input = data

    for layer_ind in range(num_layers):
        p_width = p_widths[layer_ind]
        p_height = p_heights[layer_ind]
        n_patchesH = int(h / p_height)
        n_patchesW = int(w / p_width)
        patch_size = p_width * p_height * c

        norm = BatchNormalization(axis=batch_norm_axis)(input)

        output = single_rnn(norm, n_patchesW, n_patchesH, p_width, p_height, patch_size, filters[layer_ind],
                            dirs[layer_ind], bidirs[layer_ind], cell_funcs[layer_ind], dropouts[layer_ind])

        if K.image_data_format() == 'channels_last':
            _, w, h, c = int_shape(output)
        else:
            _, c, w, h = int_shape(output)

        input = output

    concat_flatten = Flatten()(input)

    if output_layers is True:
        if dense is True:
            dense_1 = Dense(dense_size)(concat_flatten)
        else:
            dense_1 = concat_flatten

        dense = Dense(out_classes)(dense_1)
        full_out = Activation('softmax')(dense)

        model = Model(inputs=x, outputs=full_out)
    else:
        model = Model(inputs=x, outputs=concat_flatten)

    return [model, "RNN", "RNN"]


def single_rnn(x, n_patchesW, n_patchesH, p_width, p_height, patch_size, filters,
               direction='hor', bidirectional=False, cell_func=CuDNNGRU, dropout=0.25):

    if direction == 'hor':
        transpose = True
        output_shape = (n_patchesW, patch_size)
    elif direction == 'ver':
        transpose = False
        output_shape = (n_patchesH, patch_size)
    else:
        transpose = True
        output_shape = (n_patchesW, patch_size)

    x_patch_seqs = Lambda(transform_rnn_input, output_shape=output_shape,
                               arguments={'p_height': p_height, 'p_width': p_width, 'transpose': transpose})(x)

    if bidirectional is True:
        lr = Bidirectional(cell_func(filters, input_shape=output_shape, return_sequences=True),
                       merge_mode='concat', input_shape=output_shape)(x_patch_seqs)
        out_filters = 2 * filters
    else:
        lr = cell_func(filters, input_shape=output_shape, return_sequences=True)(x_patch_seqs)
        out_filters = filters

    if K.image_data_format() == 'channels_last':
        lr_reshape = Lambda(transform_rnn_output, output_shape=(n_patchesW, n_patchesH, out_filters),
                            arguments={'n_patchesW': n_patchesW, 'n_patchesH': n_patchesH, 'filters': int(out_filters / 2),
                                       'transpose': transpose})(lr)
    else:
        lr_reshape = Lambda(transform_rnn_output, output_shape=(out_filters, n_patchesW, n_patchesH),
                            arguments={'n_patchesW': n_patchesW, 'n_patchesH': n_patchesH, 'filters': int(out_filters / 2),
                                       'transpose': transpose})(lr)

    lr_reshape = Dropout(dropout)(lr_reshape)

    return lr_reshape


def transform_rnn_input(x, p_height, p_width, transpose):
    # input shape: x: (batchsize, channels, width, height) (channels first)
    # or x: (batchsize, width, height, channels) (channels last)

    import tensorflow as tf
    from keras.backend import reshape, permute_dimensions, int_shape
    import keras.backend as K

    if K.image_data_format() == 'channels_last':
        channels_first = False
    else:
        channels_first = True

    # swap rows and columns if transpose==True
    # widths and heights of patches will also need to be swapped
    if transpose:
        # tranposing
        if channels_first:
            x = permute_dimensions(x, pattern=(0, 1, 3, 2))
            # result shape: (batchsize, channels, height, width)
        else:
            x = permute_dimensions(x, pattern=(0, 2, 1, 3))
            # result shape: (batchsize, height, width, channels)

        if channels_first:
            b_size, c, h, w = int_shape(x)
        else:
            b_size, h, w, c = int_shape(x)

        n_patches_h = h / p_height  # number of patches top to bottom
        n_patches_w = w / p_width    # number of patches left to right
        patch_size = p_width * p_height * c  # size of the 3D cube corresponding to each patch

        if channels_first:
            x_patch = reshape(x, shape=(-1, c, int(n_patches_h), p_height, int(n_patches_w), p_width))
            # result shape: (batchsize, channels, n_patchesH, p_height, n_patchesW, p_width)
            x_patch_shuf = permute_dimensions(x_patch, pattern=(0, 2, 4, 3, 5, 1))
            # result shape: (batchsize, n_patchesH, n_patchesW, p_height, p_width, channels)
        else:
            x_patch = reshape(x, shape=(-1, int(n_patches_h), p_height, int(n_patches_w), p_width, c))
            # result shape: (batchsize, n_patchesH, p_height, n_patchesW, p_width, channels)
            x_patch_shuf = permute_dimensions(x_patch, pattern=(0, 1, 3, 2, 4, 5))
            # result shape: (batchsize, n_patchesH, n_patchesW, p_height, p_width, channels)

        x_patch_seqs = reshape(x_patch_shuf, shape=(-1, int(n_patches_w), patch_size))
        # result shape: (batchsize * n_patchesH, n_patchesW, p_height * p_width * channels)

    else:
        # shape: (batchsize, channels, width, height) (channels first)
        # or (batchsize, width, height, channels) (channels last)
        if channels_first:
            b_size, c, w, h = int_shape(x)
        else:
            b_size, w, h, c = int_shape(x)

        n_patches_h = h / p_height  # number of patches top to bottom
        n_patches_w = w / p_width  # number of patches left to right
        patch_size = p_width * p_height * c  # size of the 3D cube corresponding to each patch

        if channels_first:
            x_patch = reshape(x, shape=(-1, c, int(n_patches_w), p_width, int(n_patches_h), p_height))
            # result shape: (batchsize, channels, n_patchesW, p_width, n_patchesH, p_height)
        else:
            x_patch = reshape(x, shape=(-1, int(n_patches_w), p_width, int(n_patches_h), p_height, c))
            # result shape: (batchsize, n_patchesW, p_width, n_patchesH, p_height, channels)

        if channels_first:
            x_patch_shuf = permute_dimensions(x_patch, pattern=(0, 2, 4, 3, 5, 1))
            # result shape: (batchsize, n_patchesW, n_patchesH, p_width, p_height, channels)
        else:
            x_patch_shuf = permute_dimensions(x_patch, pattern=(0, 1, 3, 2, 4, 5))
            # result shape: (batchsize, n_patchesW, n_patchesH, p_width, p_height, channels)

        x_patch_seqs = reshape(x_patch_shuf, shape=(-1, int(n_patches_h), patch_size))
        # result shape: (batchsize * n_patchesW, n_patchesH, p_width * p_height * channels)

    return x_patch_seqs


def transform_rnn_output(x, n_patchesH, n_patchesW, filters, transpose):
    import tensorflow as tf
    from keras.backend import reshape, permute_dimensions
    import keras.backend as K

    if K.image_data_format() == 'channels_last':
        channels_first = False
    else:
        channels_first = True

    # swap the number of patches in each direction if transpose==True
    if transpose:
        # shape: (batchsize * n_patchesH, n_patchesW, 2 * filters)
        x_res = reshape(x, shape=(-1, n_patchesH, n_patchesW, 2 * filters))
        # result shape: (batchsize, n_patchesH, n_patchesW, 2 * filters)

        # swap the rows and columns back if transpose==True
        x_res = permute_dimensions(x_res, pattern=(0, 2, 1, 3))
        # result shape: (batchsize, n_patchesW, n_patchesH, 2 * filters)

        # swap channels back
        if channels_first:
            x_res = permute_dimensions(x_res, pattern=(0, 3, 1, 2))
            # result shape: (batchsize, 2 * filters, n_patchesW, n_patchesH)
        else:
            # all good
            # result shape: (batchsize, n_patchesW, n_patchesH, 2 * filters)
            pass
    else:
        # shape: (batchsize * n_patchesW, n_patchesH, 2 * filters)
        x_res = reshape(x, shape=(-1, n_patchesW, n_patchesH, 2 * filters))
        # result shape: (batchsize, n_patchesW, n_patchesH, 2 * filters)

        # swap channels back
        if channels_first:
            x_res = permute_dimensions(x_res, pattern=(0, 3, 1, 2))
            # result shape: (batchsize, 2 * filters, n_patchesW, n_patchesH)
        else:
            # all good
            # result shape: (batchsize, n_patchesW, n_patchesH, 2 * filters)
            pass

    return x_res
