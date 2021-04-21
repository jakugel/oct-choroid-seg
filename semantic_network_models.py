from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.layers import Dropout, BatchNormalization, Input, Activation, Add, GlobalAveragePooling2D, Reshape, Dense, multiply, Permute, maximum
from keras import backend as K
from keras.models import Model


def batch_activate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def convolution_block(x, filters, kernel, strides=(1, 1), padding='same', bat_act=True, conv_block_type=1, dilation_rate=1):
    if conv_block_type == 1:
        x = Conv2D(filters, kernel, strides=strides, padding=padding, dilation_rate=dilation_rate)(x)
        if bat_act is True:
            x = batch_activate(x)
    elif conv_block_type == 2:
        if bat_act is True:
            x = Activation('relu')(x)
        x = Conv2D(filters, kernel, strides=strides, padding=padding, dilation_rate=dilation_rate)(x)

        if bat_act is True:
            x = BatchNormalization()(x)
    elif conv_block_type == 3:
        if bat_act is True:
            x = batch_activate(x)
        x = Conv2D(filters, kernel, strides=strides, padding=padding, dilation_rate=dilation_rate)(x)
    return x


def residual_block(block_input, num_filters, conv_layers=2, block_type=3):
    inp = block_input

    if block_type == 1:
        # conv-bn-relu (add before final relu)
        for i in range(conv_layers - 1):
            inp = convolution_block(inp, num_filters, (3, 3),  conv_block_type=1)

        x = convolution_block(inp, num_filters, (3, 3), conv_block_type=1)
        x = BatchNormalization()(x)
        x = Add()([x, block_input])
        x = Activation('relu')(x)
    elif block_type == 2:
        #conv-bn-relu (add before final BN)
        for i in range(conv_layers - 1):
            inp = convolution_block(inp, num_filters, (3, 3), conv_block_type=1)

        x = convolution_block(inp, num_filters, (3, 3), conv_block_type=1)
        x = Add()([x, block_input])
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    elif block_type == 3:
        # conv-bn-relu (add at end)
        for i in range(conv_layers):
            inp = convolution_block(inp, num_filters, (3, 3), conv_block_type=1)

        x = Add()([inp, block_input])
    elif block_type == 4:
        #relu-conv-bn (add at end)
        for i in range(conv_layers):
            inp = convolution_block(inp, num_filters, (3, 3), conv_block_type=2)

        x = Add()([inp, block_input])
    elif block_type == 5:
        #bn-relu-conv (add at end)
        for i in range(conv_layers):
            inp = convolution_block(inp, num_filters, (3, 3), conv_block_type=3)

        x = Add()([inp, block_input])
    return x


def cse_block(inp, ratio=2):
    init = inp
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def sse_block(inp):
    x = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', use_bias=False)(inp)
    x = multiply([inp, x])

    return x


def scse_block(inp, ratio=2):
    x1 = cse_block(inp, ratio)
    x2 = sse_block(inp)

    x = maximum([x1, x2])

    return x


def unet_enc_block(inp, size, enc_kernel=(3, 3), conv_layers=2, pool='max', se=None, cSE_ratio=2):
    for i in range(conv_layers):
        inp = convolution_block(inp, filters=size, kernel=enc_kernel)

    if se == 'scSE':
        inp = scse_block(inp, ratio=cSE_ratio)
    elif se == 'cSE':
        inp = cse_block(inp, ratio=cSE_ratio)
    elif se == 'sSE':
        inp = sse_block(inp)

    c = inp
    if pool == 'max':
        inp = MaxPooling2D(pool_size=(2, 2))(inp)
    else:
        pass

    return [inp, c]


def upsample_conv(inp, size, kernel):
    x = UpSampling2D()(inp)
    x = convolution_block(x, filters=size, kernel=kernel)

    return x


def unet_dec_block(inp, size, concat_map, enc_kernel=(3, 3), dec_kernel=(2, 2), conv_layers=2, se=None, cSE_ratio=2):
    x = upsample_conv(inp, size, dec_kernel)

    x = concatenate([x, concat_map])
    [x, _] = unet_enc_block(x, size, enc_kernel=enc_kernel, conv_layers=conv_layers, pool=False, se=se, cSE_ratio=cSE_ratio)

    return x


def resnet_enc_block(inp, size, enc_kernel=(3, 3), block_layers=2, res_layers=1, pool='max', se=None, cSE_ratio=2):
    inp = convolution_block(inp, filters=size, kernel=enc_kernel)

    for i in range(res_layers):
        inp = residual_block(inp, size, block_layers)

    if se == 'scSE':
        inp = scse_block(inp, ratio=cSE_ratio)
    elif se == 'cSE':
        inp = cse_block(inp, ratio=cSE_ratio)
    elif se == 'sSE':
        inp = sse_block(inp)

    c = inp
    if pool == 'max':
        inp = MaxPooling2D(pool_size=(2, 2))(inp)
    else:
        pass

    return [inp, c]


def resnet_dec_block(inp, size, concat_map, enc_kernel=(3, 3), dec_kernel=(2, 2),
                     block_layers=2, res_layers=1, se=None, cSE_ratio=2, skip_type='concat'):
    x = upsample_conv(inp, size, dec_kernel)

    if skip_type == 'concat':
        x = concatenate([x, concat_map])
    elif skip_type == 'add':
        x = Add()([x, concat_map])

    [x, _] = resnet_enc_block(x, size, enc_kernel=enc_kernel, block_layers=block_layers, res_layers=res_layers,
                              pool=False, se=se, cSE_ratio=cSE_ratio)

    return x


# TODO extend generic unet to allow for dropout parameter (separate dropout in bottleneck and dropout per pooling layer)
def unet(start_neurons, pool_layers, conv_layers, enc_kernel, dec_kernel, input_channels, output_channels, se=None, cSE_ratio=2, pool='max', width=None, height=None):
    if K.image_data_format() == 'channels_last':
        inp = Input(batch_shape=(None, width, height, input_channels))
    else:
        inp = Input(batch_shape=(None, input_channels, width, height))

    x = inp

    enc = []

    for i in range(pool_layers):
        [x, c] = unet_enc_block(x, start_neurons * (2 ** i), enc_kernel=enc_kernel, conv_layers=conv_layers, se=se, cSE_ratio=cSE_ratio, pool=pool)

        enc.append(c)

    [x, _] = unet_enc_block(x, start_neurons * (2 ** pool_layers), enc_kernel=enc_kernel, conv_layers=conv_layers,
                            pool=False)
    x = Dropout(0.5)(x)

    for i in range(pool_layers):
        x = unet_dec_block(x, start_neurons * (2 ** (pool_layers - 1 - i)), enc[pool_layers - 1 - i],
                           enc_kernel=enc_kernel, dec_kernel=dec_kernel, conv_layers=conv_layers, se=se, cSE_ratio=cSE_ratio)

    o = Conv2D(filters=output_channels, kernel_size=(1, 1), strides=(1, 1), activation="softmax")(x)

    arch_params = "{:d}F, {:d}P, {:d}C".format(start_neurons, pool_layers, conv_layers) + ", " + \
                  str(enc_kernel) + "-" + str(dec_kernel) + "K" + "_convs" + "_" + pool + " pooling"

    if se is None:
        model_desc = "U-net (" + arch_params + ") {:d}class".format(output_channels)
    else:
        if se == 'sSE':
            model_desc = "U-net (" + arch_params + ", " + se + ") {:d}class".format(output_channels)
        else:
            model_desc = "U-net (" + arch_params + ", " + se + "_r=" + str(cSE_ratio) + ") {:d}class".format(output_channels)

    model_desc_short = "U-net"

    return [Model(inputs=inp, outputs=o), model_desc, model_desc_short]


def resnet(start_neurons, pool_layers, block_layers, res_layers, enc_kernel, dec_kernel, input_channels,
           output_channels, se=None, cSE_ratio=2, skip_type='concat', pool='max', pyramid_bin_sizes=None, pyramid_reduction_factors=None,
           width=None, height=None):
    if K.image_data_format() == 'channels_last':
        inp = Input(batch_shape=(None, width, height, input_channels))
    else:
        inp = Input(batch_shape=(None, input_channels, width, height))

    x = inp

    enc = []

    for i in range(pool_layers):
        [x, c] = resnet_enc_block(x, start_neurons * (2 ** i), enc_kernel=enc_kernel, block_layers=block_layers,
                                  res_layers=res_layers, se=se, cSE_ratio=cSE_ratio, pool=pool)

        enc.append(c)

    [x, _] = resnet_enc_block(x, start_neurons * (2 ** pool_layers), enc_kernel=enc_kernel, block_layers=block_layers,
                              res_layers=res_layers, pool=False)
    x = Dropout(0.5)(x)

    for i in range(pool_layers):
        x = resnet_dec_block(x, start_neurons * (2 ** (pool_layers - 1 - i)), enc[pool_layers - 1 - i],
                             enc_kernel=enc_kernel, dec_kernel=dec_kernel, block_layers=block_layers,
                             res_layers=res_layers, se=se, skip_type=skip_type, cSE_ratio=cSE_ratio)

    o = Conv2D(filters=output_channels, kernel_size=(1, 1), strides=(1, 1), activation="softmax")(x)

    arch_params = "{:d}F, {:d}P, {:d}C, {:d}R".format(start_neurons, pool_layers, block_layers, res_layers, pool) + ", " + \
                  str(enc_kernel) + "-" + str(dec_kernel) + "K" + "_" + "_" + pool + " pooling"

    if pool == 'pyramid' or pool == 'pyramid_max' or pool == 'pyramid_avg':
        arch_params += "_bins" + str(pyramid_bin_sizes) + "_red_factors" + str(pyramid_reduction_factors)

    if se is None:
        model_desc = "Residual U-net " + " (" + arch_params + ") {:d}class".format(output_channels)
    else:
        if se == 'sSE':
            model_desc = "Residual U-net " + " (" + arch_params + ", " + se + ") {:d}class".format(output_channels)
        else:
            model_desc = "Residual U-net " + " (" + arch_params + ", " + se + "_r=" + str(cSE_ratio) + ") {:d}class".format(output_channels)

    model_desc_short = "Residual U-net"

    return [Model(inputs=inp, outputs=o), model_desc, model_desc_short]


# TODO add RNN functionality to semantic networks (RNN bottleneck)
