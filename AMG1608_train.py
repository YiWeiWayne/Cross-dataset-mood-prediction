import numpy as np
import keras.backend.tensorflow_backend as KTF
import os
import tensorflow as tf
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from functions import model_structure, callback_wayne, Transfer_funcs, metric
import datetime
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam


# GPU speed limit
def get_session(gpu_fraction=1):
    # Assume that you have 6GB of GPU memory and want to allocate ~2GB
    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
KTF.set_session(get_session())

action = 'pitch+lw'
action_description = 'Change features to pitch+lw '
feature = 'pitch+lw'  # 1.melSpec 2.rCTA 3.melSpec_lw 4.rTA
source_dataset_name = 'AMG_1608'
target_dataset_name = 'CH_818'
save_path = '/data/Wayne'
emotions = ['valence']
source_dataset_path = save_path + '/Dataset/AMG1838_original/amg1838_mp3_original'
source_label_path = save_path + '/Dataset/AMG1838_original/AMG1608/amg1608_v2.xls'
target_dataset_path = save_path + '/Dataset/CH818/mp3'
target_label_path = save_path + '/Dataset/CH818/label/CH818_Annotations.xlsx'
sec_length = 29
output_sample_rate = 22050
patience = []
batch_size = 8
epochs = 300
now = datetime.datetime.now()
localtime = str(now.year) + str(now.month).zfill(2) + str(now.day).zfill(2) + '.' +\
            str(now.hour).zfill(2) + str(now.minute).zfill(2) + '.' + str(now.second).zfill(2)
execute_name = save_path + '/(' + action + ')' + source_dataset_name + '_' + localtime
loss = 'mean_squared_error'
# loss = 'mean_absolute_error'
save_best_only = False
save_weights_only = False
monitor = 'train_R2_pearsonr'
mode = 'max'
load_pretrained_weights = True
regressor_units = [1]
if feature == 'melSpec':
    filters = [32, 32, 32, 32, 32]
    kernels = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
    poolings = [(2, 4), (3, 4), (2, 5), (2, 4), (4, 4)]
    paddings = ['same', 'same', 'same', 'same', 'same']
elif feature == 'melSpec_lw':  # dim(96, 1249, 1)
    filters = [32, 32, 32, 32, 32]
    kernels = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
    poolings = [(2, 4), (3, 4), (2, 5), (2, 4), (4, 3)]
    paddings = ['same', 'same', 'same', 'same', 'same']
elif feature == 'rCTA':  # dim(30, 142, 1)
    filters = [32, 32, 32, 32, 32]
    kernels = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
    poolings = [(1, 1), (2, 2), (2, 5), (2, 7), (3, 2)]
    paddings = ['same', 'same', 'same', 'same', 'same']
    dr_rate = [0, 0, 0, 0.3, 0.3]
elif feature == 'rTA':
    filters = [32, 32, 32, 32, 32]
    kernels = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
    poolings = [(4, 1), (2, 2), (5, 5), (7, 7), (2, 2)]
    paddings = ['same', 'same', 'same', 'same', 'same']
elif feature == 'pitch':
    filters = [32, 32, 32, 32, 32]
    kernels = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
    poolings = [(5, 8), (4, 4), (2, 5), (3, 5), (3, 3)]
    paddings = ['same', 'same', 'same', 'same', 'same']
elif feature == 'pitch+lw':  # dim(360, 1249, 1)
    filters = [32, 32, 32, 32, 32]
    kernels = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
    poolings = [(2, 4), (4, 4), (4, 5), (2, 4), (5, 3)]
    paddings = ['same', 'same', 'same', 'same', 'same']
    dr_rate = [0, 0, 0, 0.3, 0.3]

para_line = []
para_line.append('action:' + str(action) + '\n')
para_line.append('action_description:' + str(action_description) + '\n')
para_line.append('feature:' + feature + '\n')
para_line.append('source_dataset_name:' + source_dataset_name + '\n')
para_line.append('target_dataset_name:' + target_dataset_name + '\n')
para_line.append('source_dataset_path:' + source_dataset_path + '\n')
para_line.append('source_label_path:' + source_label_path + '\n')
para_line.append('target_dataset_path:' + target_dataset_path + '\n')
para_line.append('target_label_path:' + target_label_path + '\n')
para_line.append('save_path:' + save_path + '\n')
para_line.append('execute_name:' + execute_name + '\n')
para_line.append('sec_length:' + str(sec_length) + '\n')
para_line.append('output_sample_rate:' + str(output_sample_rate) + '\n')
para_line.append('patience:' + str(patience) + '\n')
para_line.append('batch_size:' + str(batch_size) + '\n')
para_line.append('epochs:' + str(epochs) + '\n')
para_line.append('loss:' + str(loss) + '\n')
para_line.append('save_best_only:' + str(save_best_only) + '\n')
para_line.append('save_weights_only:' + str(save_weights_only) + '\n')
para_line.append('monitor:' + str(monitor) + '\n')
para_line.append('mode:' + str(mode) + '\n')
para_line.append('load_pretrained_weights:' + str(load_pretrained_weights) + '\n')
para_line.append('filters:' + str(filters) + '\n')
para_line.append('kernels:' + str(kernels) + '\n')
para_line.append('poolings:' + str(poolings) + '\n')
para_line.append('paddings:' + str(paddings) + '\n')
para_line.append('dr_rate:' + str(dr_rate) + '\n')
if not os.path.exists(execute_name):
    os.makedirs(execute_name)
paraFile = open(os.path.join(execute_name, 'Parameters.txt'), 'w')
paraFile.writelines(para_line)
paraFile.close()

# # transfer mp3 to wav file
# Transfer_funcs.audio_to_wav(dataset_name=source_dataset_name, dataset_path=source_dataset_path,
#                             label_path=source_label_path,
#                             sec_length=sec_length, output_sample_rate=output_sample_rate, save_path=save_path)
# Transfer_funcs.audio_to_wav(dataset_name=target_dataset_name, dataset_path=target_dataset_path,
#                             label_path=target_label_path,
#                             sec_length=sec_length, output_sample_rate=output_sample_rate, save_path=save_path)

# # Generate Train X and Train Y
# Transfer_funcs.wav_to_npy(dataset_name=source_dataset_name, label_path=source_label_path,
#                           output_sample_rate=output_sample_rate, save_path=save_path)
# Transfer_funcs.wav_to_npy(dataset_name=target_dataset_name, label_path=target_label_path,
#                           output_sample_rate=output_sample_rate, save_path=save_path)

# # Transfer X from format npy to mat
# Transfer_funcs.npy_to_mat(dataset_name=source_dataset_name,
#                           output_sample_rate=output_sample_rate, save_path=save_path)
# Transfer_funcs.npy_to_mat(dataset_name=target_dataset_name,
#                           output_sample_rate=output_sample_rate, save_path=save_path)

# load data
Train_Y_valence = np.load(save_path + '/' + source_dataset_name + '/Train_Y_valence.npy')
Train_Y_arousal = np.load(save_path + '/' + source_dataset_name + '/Train_Y_arousal.npy')
Train_X = np.load(save_path + '/' + source_dataset_name +
                  '/Train_X@' + str(output_sample_rate) + 'Hz_' + feature + '.npy')

Val_Y_valence = np.load(save_path + '/' + target_dataset_name + '/Train_Y_valence.npy')
Val_Y_arousal = np.load(save_path + '/' + target_dataset_name + '/Train_Y_arousal.npy')
Val_X = np.load(save_path + '/' + target_dataset_name +
                '/Train_X@' + str(output_sample_rate) + 'Hz_' + feature + '.npy')
print('Train_X: ' + str(Train_X.shape))

# Training
for emotion_axis in emotions:
    if KTF._SESSION:
        print('Reset session.')
        KTF.clear_session()
        KTF.set_session(get_session())
    if emotion_axis == 'valence':
        Train_Y = Train_Y_valence
        Val_Y = Val_Y_valence
    else:
        Train_Y = Train_Y_arousal
        Val_Y = Val_Y_arousal
    # Generator for source and target data
    # train_data_generator = ADDA_funcs.data_generator(Train_X, Train_Y, batch_size)
    # val_data_generator = ADDA_funcs.data_generator(Val_X, Val_Y, batch_size)
    feature_tensor = Input(shape=(Train_X.shape[1], Train_X.shape[2], 1))
    extractor = model_structure.compact_cnn_extractor(x=feature_tensor,
                                                      filters=filters, kernels=kernels, poolings=poolings,
                                                      paddings=paddings, dr_rate=dr_rate)
    regressor = model_structure.regression_classifier(x=extractor, units=regressor_units)
    model = Model(inputs=feature_tensor, outputs=regressor)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
    model.summary()
    if load_pretrained_weights:
        model.load_weights(save_path + '/compact_cnn_weights.h5', by_name=True)
    model_path = execute_name + '/' + emotion_axis + '/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model.save(model_path + '/init_run.h5')
    LossR2Logger_ModelCheckPoint = callback_wayne.LossR2Logger_ModelCheckPoint(
        train_data=(Train_X, Train_Y), val_data=(Val_X, Val_Y),
        file_name=model_path + '/log', run_num=0,
        filepath=model_path, monitor=monitor, verbose=0,
        save_best_only=save_best_only, save_weights_only=save_weights_only,
        mode=mode, period=1)
    model.fit(Train_X, Train_Y, batch_size=batch_size, epochs=epochs,
              callbacks=[LossR2Logger_ModelCheckPoint],
              verbose=1, shuffle=True, validation_data=(Val_X, Val_Y))

    # model.fit_generator(generator=train_data_generator, epochs=epochs, verbose=1,
    #                     steps_per_epoch=int(len(Train_Y)/batch_size),
    #                     validation_steps=int(len(Val_Y)/batch_size),
    #                     callbacks=[LossR2Logger_ModelCheckPoint], validation_data=val_data_generator)