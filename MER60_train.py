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



# # GPU speed limit
# def get_session(gpu_fraction=0.6):
#     # Assume that you have 6GB of GPU memory and want to allocate ~2GB
#     gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=gpu_fraction)
#     return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
# KTF.set_session(get_session())


dataset_name = 'MER_60'
save_path = '/data/Wayne'
dataset_path = save_path + '/Dataset/MER60_wav/wav_30sec'
wav_path = save_path + '/Dataset//MER60_wav/mer60_wav'
label_path = save_path + '/Dataset/MER60_wav/label/Yije.mat'
# dataset_path = '../../Dataset/AMG1838_original/amg1838_mp3_original'
# wav_path = '../../Dataset/AMG1838_original/amg1608_wav'
# label_path = '../../Dataset/AMG1838_original/AMG1608/amg1608_v2.xls'
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
execute_name = save_path + '/' + dataset_name + '_' + localtime
loss = 'mean_squared_error'

para_line = []
para_line.append('dataset_name:' + dataset_name + '\n')
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
if not os.path.exists(execute_name):
    os.makedirs(execute_name)
paraFile = open(os.path.join(execute_name, 'Parameters.txt'), 'w')
paraFile.writelines(para_line)
paraFile.close()

# # transfer mp3 to wav file
# if not os.path.exists(wav_path):
#     os.makedirs(wav_path)
# if True:
#     for i in range(1, 5):
#         for j in range(1, 16):
#             print(str(i) + '.' + str(j).zfill(2))
#             for root, subdirs, files in os.walk(dataset_path):
#                 for f in files:
#                     if os.path.splitext(f)[1] == '.wav' and f[0:4].startswith(str(i)+'.'+str(j).zfill(2)):
#                         y, sr = librosa.load(os.path.join(root, f), sr=output_sample_rate)
#                         print(y.shape)
#                         print(str(sr))
#                         if y.shape[0] >= output_sample_rate*sec_length:
#                             librosa.output.write_wav(path=wav_path + '/' + str((i-1)*15+j).zfill(2) + '@' +
#                                                           str(output_sample_rate) + '$' + f,
#                                                      y=y[0:int(output_sample_rate*sec_length)], sr=output_sample_rate)
#                         else:
#                             print('Shorter: ' + str(y.shape[0]) + '/' + str(sample_rate*sec_length))

# load X & Y
Train_Y_valence = np.load(save_path + '/' + dataset_name + '/Train_Y_valence.npy')
Train_Y_arousal = np.load(save_path + '/' + dataset_name + '/Train_Y_arousal.npy')
Train_X = np.load(save_path + '/' + dataset_name + '/Train_X.npy')

# # load mat data
# Train_Y = sio.loadmat(label_path)
# Train_Y = Train_Y['Yije']
# Train_Y = np.nanmean(Train_Y, axis=1)
# Train_Y_valence = Train_Y[:, 0]
# Train_Y_arousal = Train_Y[:, 1]
#
# Train_X = []
# for i in range(1, len(Train_Y)+1):
#     print(str(i))
#     for root, subdirs, files in os.walk(wav_path):
#         for f in files:
#             if f[0:2].startswith(str(i).zfill(2)):
#                 y, sr = librosa.load(os.path.join(root, f), sr=output_sample_rate)
#                 Train_X.append(y)
# Train_X = np.vstack(Train_X)
# Train_X = Train_X.reshape((Train_X.shape[0], 1, Train_X.shape[1]))
# print(Train_Y_valence.shape)
# print(Train_Y_arousal.shape)
# print(Train_X.shape)
# if not os.path.exists(save_path + '/' + dataset_name):
#     os.makedirs(save_path + '/' + dataset_name)
# np.save(save_path + '/' + dataset_name + '/Train_X.npy', Train_X)
# np.save(save_path + '/' + dataset_name + '/Train_Y_valence.npy', Train_Y_valence)
# np.save(save_path + '/' + dataset_name + '/Train_Y_arousal.npy', Train_Y_arousal)

# # Cross validation split
# kf = KFold(n_splits=10, shuffle=True)
#
# # Training
# with tf.device('/gpu:0'):
#     for emotion_axis in ['valence', 'arousal']:
#         for i, (train, test) in enumerate(kf.split(Train_X)):
#             if KTF._SESSION:
#                 print('Reset session.')
#                 KTF.clear_session()
#                 KTF.set_session(get_session())
#             if emotion_axis == 'valence':
#                 Y = Train_Y_valence
#             else:
#                 Y = Train_Y_arousal
#             print("%s %s" % (train, test))
#             model = model_structure.compact_cnn(loss=loss)
#             model.load_weights('test.h5', by_name=True)
#             model_path = execute_name + '/' + emotion_axis + '/CV' + str(i) + '/'
#             if not os.path.exists(model_path):
#                 os.makedirs(model_path)
#             model.save(model_path + '/init_run.h5')
#             np.save(model_path + 'train.npy', train)
#             np.save(model_path + 'test.npy', test)
#             # earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=0,
#             #                                               mode='min')
#             # checkPoints = ModelCheckpoint(filepath=model_path + '/R2_{val_R2:.10f}_loss_{val_loss:.10f}_CV' + str(i)
#             #                                        + '.h5', monitor='val_R2',
#             #                               verbose=0, save_best_only=True, mode='max',
#             #                               period=1)
#             # Log_callback = callback_wayne.Loss_R2_Logger((Train_X[train, :], Y[train]), (Train_X[test, :], Y[test]),
#             #                                              file_name=model_path + '/log_CV', run_num=i)
#             LossR2Logger_ModelCheckPoint = callback_wayne.LossR2Logger_ModelCheckPoint(
#                 train_data=(Train_X[train, :], Y[train]), val_data=(Train_X[test, :], Y[test]),
#                 file_name=model_path + '/log_CV', run_num=i,
#                 filepath=model_path, monitor='val_loss', verbose=0,
#                 save_best_only=False, save_weights_only=True,
#                 mode='min', period=1)
#             model.fit(Train_X[train, :], Y[train], batch_size=batch_size, epochs=epochs,
#                       callbacks=[LossR2Logger_ModelCheckPoint],
#                       verbose=1, shuffle=True, validation_data=(Train_X[test, :], Y[test]))
