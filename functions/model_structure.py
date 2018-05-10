import keras
from keras.layers import Dense, Reshape, concatenate
from keras.layers import BatchNormalization, Dropout, Flatten, Conv2D, Conv1D, MaxPooling2D, AveragePooling2D
from kapre.time_frequency import Melspectrogram
import keras.backend.tensorflow_backend as KTF
from functions.Custom_layers import Std2DLayer


def extract_melspec(sr=12000, input_tensor=None, tf='melgram', fmin=0.0, fmax=6000, n_mels=96, decibel=True,
                    trainable_fb=False, trainable_kernel=False, n_dft=512, n_hop=256):
    x = Melspectrogram(n_dft=n_dft, n_hop=n_hop, power_melgram=2.0,
                       trainable_kernel=trainable_kernel,
                       trainable_fb=trainable_fb,
                       return_decibel_melgram=decibel,
                       sr=sr, n_mels=n_mels,
                       fmin=fmin, fmax=fmax,
                       name=tf)(input_tensor)
    return x


def domain_classifier(x, units, output_activation):
    for i in range(0, len(units)-1):
        x = Dense(units[i], kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')(x)
        x = keras.layers.advanced_activations.ELU(alpha=1.0)(x)
    x = Dense(units[len(units)-1], activation=output_activation,
              kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')(x)
    return x


def enforced_domain_classifier(x, units, kernels, strides, paddings, output_activation):
    if len(units) > 1:
        x = Reshape((KTF.int_shape(x)[1], 1))(x)
        for i in range(0, len(units)-1):
            x = Conv1D(units[i], kernel_size=kernels[i], strides=strides[i], padding=paddings[i])(x)
            x = BatchNormalization(axis=-1)(x)
            x = keras.layers.advanced_activations.ELU(alpha=1.0)(x)
        x = Flatten()(x)
    x = Dense(units[len(units)-1], activation=output_activation,
              kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')(x)
    return x


def regression_classifier(x, units):
    for i in range(0, len(units)-1):
        x = Dense(units[i], activation="relu",
                  kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')(x)
    x = Dense(units[len(units)-1], activation="tanh",
              kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')(x)
    return x


def enforced_regression_classifier(x, units, kernels, strides, paddings):
    if len(units) > 1:
        x = Reshape((KTF.int_shape(x)[1], 1))(x)
        for i in range(0, len(units)-1):
            x = Conv1D(filters=units[i],
                       kernel_size=kernels[i],
                       strides=strides[i],
                       padding=paddings[i],
                       kernel_initializer='glorot_uniform',
                       bias_initializer='glorot_uniform')(x)
            x = BatchNormalization(axis=-1)(x)
            x = keras.layers.advanced_activations.ELU(alpha=1.0)(x)
        x = Flatten()(x)
    x = Dense(units[len(units)-1], activation="tanh",
              kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')(x)
    return x


def compact_cnn_extractor(x, filters, kernels, strides, paddings, poolings, dr_rate):
    for i in range(0, len(filters)):
        x = Conv2D(filters=filters[i],
                   kernel_size=kernels[i],
                   strides=strides[i],
                   padding=paddings[i],
                   kernel_initializer='glorot_uniform',
                   bias_initializer='glorot_uniform')(x)
        x = BatchNormalization(axis=-1)(x)
        x = keras.layers.advanced_activations.ELU(alpha=1.0)(x)
        if poolings[i] != (1, 1):
            x = MaxPooling2D(pool_size=poolings[i])(x)
        if dr_rate[i] != 0:
            x = Dropout(rate=dr_rate[i])(x)
    mp = MaxPooling2D(pool_size=poolings[len(poolings)-1])(x)
    ap = AveragePooling2D(pool_size=poolings[len(poolings)-1])(x)
    sp = Std2DLayer(axis=2)(x)
    x = concatenate([mp, ap, sp])
    x = Flatten()(x)
    if dr_rate[len(dr_rate)-1] != 0:
        x = Dropout(rate=dr_rate[len(dr_rate)-1])(x)
    return x



