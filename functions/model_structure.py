import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Activation, Convolution1D, MaxPooling1D, Reshape, GlobalAveragePooling1D
from keras.layers import BatchNormalization, LSTM, Dropout, TimeDistributed, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from kapre.time_frequency import Melspectrogram
from functions import metric


# def CRNN_ori(input_shape=128, lr_decay=0.01, dr_rate=0.3, unit_size=128,
#              cnn_act='relu', kernel_size=32, stride=1, padding='valid',
#              rnn_act='tanh', rnn_recur_act='hard_sigmoid', go_backwards=True,
#              nn_act='relu'):
#     input1 = Input(shape=(input_shape, 1))
#     input2 = Input(shape=(input_shape, 1))
#     input3 = Input(shape=(input_shape, 1))
#     CNN1 = Convolution1D(filters=unit_size, kernel_size=kernel_size, strides=stride, padding=padding)(input1)
#     CNN2 = Convolution1D(filters=unit_size, kernel_size=kernel_size, strides=stride, padding=padding)(input2)
#     CNN3 = Convolution1D(filters=unit_size, kernel_size=kernel_size, strides=stride, padding=padding)(input3)
#     BN1 = BatchNormalization()(CNN1)
#     BN2 = BatchNormalization()(CNN2)
#     BN3 = BatchNormalization()(CNN3)
#     BN1 = Activation(cnn_act)(BN1)
#     BN2 = Activation(cnn_act)(BN2)
#     BN3 = Activation(cnn_act)(BN3)
#     MP1 = MaxPooling1D(pool_size=int((input_shape - kernel_size) / stride + 1))(BN1)
#     MP2 = MaxPooling1D(pool_size=int((input_shape - kernel_size) / stride + 1))(BN2)
#     MP3 = MaxPooling1D(pool_size=int((input_shape - kernel_size) / stride + 1))(BN3)
#     DP1 = Dropout(rate=dr_rate)(MP1)
#     DP2 = Dropout(rate=dr_rate)(MP2)
#     DP3 = Dropout(rate=dr_rate)(MP3)
#     merged_vector = concatenate([DP1, DP2, DP3], axis=1)
#     lstm1 = LSTM(units=unit_size, activation=rnn_act, recurrent_activation=rnn_recur_act,
#                  dropout=dr_rate, recurrent_dropout=dr_rate, return_sequences=True,
#                  go_backwards=go_backwards)(merged_vector)
#     lstm2 = LSTM(units=unit_size, activation=rnn_act, recurrent_activation=rnn_recur_act,
#                  dropout=dr_rate, recurrent_dropout=dr_rate, return_sequences=True,
#                  go_backwards=go_backwards)(lstm1)
#     dense = TimeDistributed(Dense(units=unit_size))(lstm2)
#     BN4 = BatchNormalization()(dense)
#     BN4 = Activation(nn_act)(BN4)
#     output = TimeDistributed(Dense(units=1, activation='sigmoid'))(BN4)
#     model = Model(inputs=[input1, input2, input3], outputs=[output])
#     adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=lr_decay)
#     model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
#     model.summary()
#     return model


def DCNN(input_shape=660550, dr_rate=0.5, padding='same',
         pool_size=3, pool_stride=3,
         cnn_act='relu',
         cnn_filter1=128, cnn_kernel1=3, cnn_stride1=3, cnn_stride1_p=1,
         cnn_filter2=256, cnn_kernel2=3, cnn_stride2=1,
         cnn_filter3=512, cnn_kernel3=3, cnn_kernel3_p=1, cnn_stride3=1,
         nn_units=1, nn_act='tanh'):
    input1 = Input(shape=(input_shape, 1))
    CNN1 = Convolution1D(filters=cnn_filter1, kernel_size=cnn_kernel1, strides=cnn_stride1, padding=padding)(input1)
    CNN1 = BatchNormalization()(CNN1)
    CNN1 = Activation(cnn_act)(CNN1)
    CNN2 = Convolution1D(filters=cnn_filter1, kernel_size=cnn_kernel1, strides=cnn_stride1_p, padding=padding)(CNN1)
    CNN2 = BatchNormalization()(CNN2)
    CNN2 = Activation(cnn_act)(CNN2)
    CNN2 = MaxPooling1D(pool_size=pool_size, strides=pool_stride, padding=padding)(CNN2)
    CNN3 = Convolution1D(filters=cnn_filter1, kernel_size=cnn_kernel1, strides=cnn_stride1_p, padding=padding)(CNN2)
    CNN3 = BatchNormalization()(CNN3)
    CNN3 = Activation(cnn_act)(CNN3)
    CNN3 = MaxPooling1D(pool_size=pool_size, strides=pool_stride, padding=padding)(CNN3)
    CNN4 = Convolution1D(filters=cnn_filter2, kernel_size=cnn_kernel2, strides=cnn_stride2, padding=padding)(CNN3)
    CNN4 = BatchNormalization()(CNN4)
    CNN4 = Activation(cnn_act)(CNN4)
    CNN4 = MaxPooling1D(pool_size=pool_size, strides=pool_stride, padding=padding)(CNN4)
    CNN5 = Convolution1D(filters=cnn_filter2, kernel_size=cnn_kernel2, strides=cnn_stride2, padding=padding)(CNN4)
    CNN5 = BatchNormalization()(CNN5)
    CNN5 = Activation(cnn_act)(CNN5)
    CNN5 = MaxPooling1D(pool_size=pool_size, strides=pool_stride, padding=padding)(CNN5)
    CNN6 = Convolution1D(filters=cnn_filter2, kernel_size=cnn_kernel2, strides=cnn_stride2, padding=padding)(CNN5)
    CNN6 = BatchNormalization()(CNN6)
    CNN6 = Activation(cnn_act)(CNN6)
    CNN6 = MaxPooling1D(pool_size=pool_size, strides=pool_stride, padding=padding)(CNN6)
    CNN7 = Convolution1D(filters=cnn_filter2, kernel_size=cnn_kernel2, strides=cnn_stride2, padding=padding)(CNN6)
    CNN7 = BatchNormalization()(CNN7)
    CNN7 = Activation(cnn_act)(CNN7)
    CNN7 = MaxPooling1D(pool_size=pool_size, strides=pool_stride, padding=padding)(CNN7)
    CNN8 = Convolution1D(filters=cnn_filter2, kernel_size=cnn_kernel2, strides=cnn_stride2, padding=padding)(CNN7)
    CNN8 = BatchNormalization()(CNN8)
    CNN8 = Activation(cnn_act)(CNN8)
    CNN8 = MaxPooling1D(pool_size=pool_size, strides=pool_stride, padding=padding)(CNN8)
    CNN9 = Convolution1D(filters=cnn_filter2, kernel_size=cnn_kernel2, strides=cnn_stride2, padding=padding)(CNN8)
    CNN9 = BatchNormalization()(CNN9)
    CNN9 = Activation(cnn_act)(CNN9)
    CNN9 = MaxPooling1D(pool_size=pool_size, strides=pool_stride, padding=padding)(CNN9)
    CNN10 = Convolution1D(filters=cnn_filter3, kernel_size=cnn_kernel3, strides=cnn_stride3, padding=padding)(CNN9)
    CNN10 = BatchNormalization()(CNN10)
    CNN10 = Activation(cnn_act)(CNN10)
    CNN10 = MaxPooling1D(pool_size=pool_size, strides=pool_stride, padding=padding)(CNN10)
    CNN11 = Convolution1D(filters=cnn_filter3, kernel_size=cnn_kernel3_p, strides=cnn_stride3, padding=padding)(CNN10)
    CNN11 = BatchNormalization()(CNN11)
    CNN11 = Activation(cnn_act)(CNN11)
    DP1 = Dropout(rate=dr_rate)(CNN11)
    DP1 = Flatten()(DP1)
    output1 = Dense(units=nn_units, activation=nn_act)(DP1)
    model = Model(inputs=[input1], outputs=[output1])
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=[metric.R2])
    model.summary()
    return model


def compact_cnn(sr=12000, sec_length=29, tf='melgram', fmin=0.0, fmax=6000, n_mels=96, decibel=True,
                trainable_fb=False, trainable_kernel=False, loss='mean_squared_error'):
    input_length = sr*sec_length
    model = Sequential()
    model.add(Melspectrogram(n_dft=512, n_hop=256, power_melgram=2.0,
                             input_shape=(1, input_length),
                             trainable_kernel=trainable_kernel,
                             trainable_fb=trainable_fb,
                             return_decibel_melgram=decibel,
                             sr=sr, n_mels=n_mels,
                             fmin=fmin, fmax=fmax,
                             name=tf))
    poolings = [(2, 4), (3, 4), (2, 5), (2, 4), (4, 4)]
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     padding='same',
                     kernel_initializer='Zeros',
                     bias_initializer='Zeros'))
    model.add(BatchNormalization(axis=-1))
    model.add(keras.layers.advanced_activations.ELU(alpha=1.0))
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     padding='same',
                     kernel_initializer='Zeros',
                     bias_initializer='Zeros'))
    model.add(BatchNormalization(axis=-1))
    model.add(keras.layers.advanced_activations.ELU(alpha=1.0))
    model.add(MaxPooling2D(pool_size=(3, 4)))
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     padding='same',
                     kernel_initializer='Zeros',
                     bias_initializer='Zeros'))
    model.add(BatchNormalization(axis=-1))
    model.add(keras.layers.advanced_activations.ELU(alpha=1.0))
    model.add(MaxPooling2D(pool_size=(2, 5)))
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     padding='same',
                     kernel_initializer='Zeros',
                     bias_initializer='Zeros'))
    model.add(BatchNormalization(axis=-1))
    model.add(keras.layers.advanced_activations.ELU(alpha=1.0))
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     padding='same',
                     kernel_initializer='Zeros',
                     bias_initializer='Zeros'))
    model.add(BatchNormalization(axis=-1))
    model.add(keras.layers.advanced_activations.ELU(alpha=1.0))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Flatten())
    model.add(Dense(1, activation='tanh', kernel_initializer='Zeros', bias_initializer='Zeros'))
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss=loss, metrics=[metric.R2])
    model.summary()
    return model


def functional_compact_cnn(sr=12000, sec_length=29, tf='melgram', fmin=0.0, fmax=6000, n_mels=96, decibel=True,
                           trainable_fb=False, trainable_kernel=False, loss='mean_squared_error',
                           model_type="source_extractor"):
    input_length = sr*sec_length
    # source_extractor_tensor = Sequential()
    audio_input = Input(shape=(1, input_length))
    source_extractor_tensor = Melspectrogram(n_dft=512, n_hop=256, power_melgram=2.0,
                                             input_shape=(1, input_length),
                                             trainable_kernel=trainable_kernel,
                                             trainable_fb=trainable_fb,
                                             return_decibel_melgram=decibel,
                                             sr=sr, n_mels=n_mels,
                                             fmin=fmin, fmax=fmax,
                                             name=tf)(audio_input)
    source_extractor_tensor = Conv2D(filters=32,
                                     kernel_size=(3, 3),
                                     padding='same',
                                     kernel_initializer='Zeros',
                                     bias_initializer='Zeros')(source_extractor_tensor)
    source_extractor_tensor = BatchNormalization(axis=-1)(source_extractor_tensor)
    source_extractor_tensor = keras.layers.advanced_activations.ELU(alpha=1.0)(source_extractor_tensor)
    source_extractor_tensor = MaxPooling2D(pool_size=(2, 4))(source_extractor_tensor)
    source_extractor_tensor = Conv2D(filters=32,
                                     kernel_size=(3, 3),
                                     padding='same',
                                     kernel_initializer='Zeros',
                                     bias_initializer='Zeros')(source_extractor_tensor)
    source_extractor_tensor = BatchNormalization(axis=-1)(source_extractor_tensor)
    source_extractor_tensor = keras.layers.advanced_activations.ELU(alpha=1.0)(source_extractor_tensor)
    source_extractor_tensor = MaxPooling2D(pool_size=(3, 4))(source_extractor_tensor)
    source_extractor_tensor = Conv2D(filters=32,
                                     kernel_size=(3, 3),
                                     padding='same',
                                     kernel_initializer='Zeros',
                                     bias_initializer='Zeros')(source_extractor_tensor)
    source_extractor_tensor = BatchNormalization(axis=-1)(source_extractor_tensor)
    source_extractor_tensor = keras.layers.advanced_activations.ELU(alpha=1.0)(source_extractor_tensor)
    source_extractor_tensor = MaxPooling2D(pool_size=(2, 5))(source_extractor_tensor)
    source_extractor_tensor = Conv2D(filters=32,
                                     kernel_size=(3, 3),
                                     padding='same',
                                     kernel_initializer='Zeros',
                                     bias_initializer='Zeros')(source_extractor_tensor)
    source_extractor_tensor = BatchNormalization(axis=-1)(source_extractor_tensor)
    source_extractor_tensor = keras.layers.advanced_activations.ELU(alpha=1.0)(source_extractor_tensor)
    source_extractor_tensor = MaxPooling2D(pool_size=(2, 4))(source_extractor_tensor)
    source_extractor_tensor = Conv2D(filters=32,
                                     kernel_size=(3, 3),
                                     padding='same',
                                     kernel_initializer='Zeros',
                                     bias_initializer='Zeros')(source_extractor_tensor)
    source_extractor_tensor = BatchNormalization(axis=-1)(source_extractor_tensor)
    source_extractor_tensor = keras.layers.advanced_activations.ELU(alpha=1.0)(source_extractor_tensor)
    source_extractor_tensor = MaxPooling2D(pool_size=(4, 4))(source_extractor_tensor)
    encoded_audio = Flatten()(source_extractor_tensor)

    # extract feature extraction input and output
    source_extractor = Model(inputs=audio_input, outputs=encoded_audio, name=model_type)
    # source_extractor.summary()

    output = Dense(1, activation="tanh", kernel_initializer='Zeros', bias_initializer='Zeros')(encoded_audio)
    model = Model(inputs=audio_input, outputs=output)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss=loss, metrics=[metric.R2])
    # model.summary()
    return model, source_extractor, audio_input, encoded_audio


def domain_classifier(encoded_audio_tensor):
    x = Dense(500, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')(encoded_audio_tensor)
    x = Dense(500, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')(x)
    x = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')(x)
    return x


def regression_classifier(encoded_audio_tensor):
    x = Dense(1, activation="tanh", kernel_initializer='Zeros', bias_initializer='Zeros')(encoded_audio_tensor)
    return x



