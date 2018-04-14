import numpy as np
import scipy.io as sio
import keras.backend.tensorflow_backend as KTF
import os
import tensorflow as tf
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import keras
from keras.callbacks import ModelCheckpoint
from functions import model_structure, callback_wayne
from sklearn.model_selection import KFold
import time
from pyexcel_xls import get_data
import json
import librosa
import datetime



# GPU speed limit
def get_session(gpu_fraction=0.6):
    # Assume that you have 6GB of GPU memory and want to allocate ~2GB
    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
KTF.set_session(get_session())


source_dataset_name = 'AMG_1608'
target_dataset_name = 'CH_818'
save_path = '/mnt/data/Wayne'
dataset_path = save_path + '/Dataset/AMG1838_original/amg1838_mp3_original'
wav_path = save_path + '/Dataset/AMG1838_original/amg1608_wav'
label_path = save_path + '/Dataset/AMG1838_original/AMG1608/amg1608_v2.xls'
sec_length = 29
sample_rate = 22050
output_sample_rate = 12000
patience = []
batch_size = 16
epochs = 100
wav_path = wav_path + '_' + str(output_sample_rate) + 'Hz'
now = datetime.datetime.now()
localtime = str(now.year) + str(now.month).zfill(2) + str(now.day).zfill(2) + '.' +\
            str(now.hour).zfill(2) + str(now.minute).zfill(2) + '.' + str(now.second).zfill(2)
execute_name = save_path + '/' + source_dataset_name + '_' + localtime
loss = 'mean_squared_error'
save_best_only = True
save_weights_only = False
monitor = 'train_R2_pearsonr'
mode = 'max'

para_line = []
para_line.append('source_dataset_name:' + source_dataset_name + '\n')
para_line.append('target_dataset_name:' + target_dataset_name + '\n')
para_line.append('save_path:' + save_path + '\n')
para_line.append('execute_name:' + execute_name + '\n')
para_line.append('dataset_path:' + dataset_path + '\n')
para_line.append('label_path:' + label_path + '\n')
para_line.append('wav_path:' + wav_path + '\n')
para_line.append('sec_length:' + str(sec_length) + '\n')
para_line.append('sample_rate:' + str(sample_rate) + '\n')
para_line.append('output_sample_rate:' + str(output_sample_rate) + '\n')
para_line.append('patience:' + str(patience) + '\n')
para_line.append('batch_size:' + str(batch_size) + '\n')
para_line.append('epochs:' + str(epochs) + '\n')
para_line.append('loss:' + str(loss) + '\n')
para_line.append('save_best_only:' + str(save_best_only) + '\n')
para_line.append('save_weights_only:' + str(save_weights_only) + '\n')
para_line.append('monitor:' + str(monitor) + '\n')
para_line.append('mode:' + str(mode) + '\n')

if not os.path.exists(execute_name):
    os.makedirs(execute_name)
paraFile = open(os.path.join(execute_name, 'Parameters.txt'), 'w')
paraFile.writelines(para_line)
paraFile.close()

# load data
Train_Y_valence = np.load(save_path + '/' + source_dataset_name + '/Train_Y_valence.npy')
Train_Y_arousal = np.load(save_path + '/' + source_dataset_name + '/Train_Y_arousal.npy')
Train_X = np.load(save_path + '/' + source_dataset_name + '/Train_X.npy')

Val_Y_valence = np.load(save_path + '/' + target_dataset_name + '/Train_Y_valence.npy')
Val_Y_arousal = np.load(save_path + '/' + target_dataset_name + '/Train_Y_arousal.npy')
Val_X = np.load(save_path + '/' + target_dataset_name + '/Train_X.npy')

# # load xls data
# data = get_data(label_path)
# encodedjson = json.dumps(data)
# decodejson = json.loads(encodedjson)
# decodejson = decodejson['amg1608_v2']

# # transfer mp3 to wav file
# if not os.path.exists(wav_path):
#     os.makedirs(wav_path)
# if True:
#     for i in range(1, len(decodejson)):
#         print(str(i))
#         if os.path.exists(dataset_path + '/' + str(decodejson[i][2]) + '.mp3'):
#             print(dataset_path + '/' + str(decodejson[i][2]) + '.mp3')
#         else:
#             print('fail')
#         y, sr = librosa.load(dataset_path + '/' + str(decodejson[i][2]) + '.mp3', sr=output_sample_rate)
#         print(y.shape)
#         print(str(sr))
#         if y.shape[0] >= output_sample_rate*sec_length:
#             librosa.output.write_wav(path=wav_path + '/' + str(i).zfill(4) + '@' + str(output_sample_rate) +
#                                      '$' + str(decodejson[i][2]) + '.wav',
#                                      y=y[0:int(output_sample_rate*sec_length)], sr=output_sample_rate)
#         else:
#             print('Shorter: ' + str(y.shape[0]) + '/' + str(sample_rate*sec_length))

# # Generate Train X and Train Y
# Train_Y_valence = []
# Train_Y_arousal = []
# Train_X = []
# for i in range(1, len(decodejson)):
#     print(str(i))
#     Train_Y_valence.append(decodejson[i][7])
#     Train_Y_arousal.append(decodejson[i][8])
#     y, sr = librosa.load(wav_path + '/' + str(i).zfill(4) + '@' + str(output_sample_rate) +
#                          '$' + str(decodejson[i][2]) + '.wav', sr=output_sample_rate)
#     Train_X.append(y)
# Train_Y_valence = np.hstack(Train_Y_valence)
# Train_Y_arousal = np.hstack(Train_Y_arousal)
# Train_X = np.vstack(Train_X)
# Train_X = Train_X.reshape((Train_X.shape[0], 1, Train_X.shape[1]))
# print(Train_Y_valence.shape)
# print(Train_Y_arousal.shape)
# print(Train_X.shape)
# if not os.path.exists(save_path + '/' + source_dataset_name):
#     os.makedirs(save_path + '/' + source_dataset_name)
# np.save(save_path + '/' + source_dataset_name + '/Train_X.npy', Train_X)
# np.save(save_path + '/' + source_dataset_name + '/Train_Y_valence.npy', Train_Y_valence)
# np.save(save_path + '/' + source_dataset_name + '/Train_Y_arousal.npy', Train_Y_arousal)

# Training
for emotion_axis in ['valence', 'arousal']:
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
    model = model_structure.compact_cnn(loss=loss)
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
