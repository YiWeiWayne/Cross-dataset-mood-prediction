import numpy as np
import keras.backend.tensorflow_backend as KTF
import os
import tensorflow as tf
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from functions import model_structure, callback_wayne, metric
import json
import csv
from pyexcel_xls import save_data
from collections import OrderedDict
from keras.models import Model, load_model
from keras.layers import Input
from keras.optimizers import SGD, Adam
from kapre.time_frequency import Melspectrogram


# GPU speed limit
def get_session(gpu_fraction=0.6):
    # Assume that you have 6GB of GPU memory and want to allocate ~2GB
    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
KTF.set_session(get_session())

# Parameters
action = '22K'
algorithm = 'ADDA'
source_dataset_name = 'AMG_1608'
target_dataset_name = 'CH_818'
save_path = '/data/Wayne'
source_execute_name = save_path + '/(' + action + ')' + source_dataset_name + '_20180420.2130.27'
transfer_execute_name = save_path + '/(' + action + ')' + \
                        algorithm + '_S_' + source_dataset_name + '_T_' + target_dataset_name + '_20180421.1156.13'
loss = 'mean_squared_error'
fold = 10
encoded_size = 32
emotions = ['valence', 'arousal']
para_File = open(source_execute_name + '/Parameters.txt', 'r')
parameters = para_File.readlines()
para_File.close()
for i in range(0, len(parameters)):
    if 'feature:' in parameters[i]:
        feature = parameters[i][len('feature:'):-1]
        print(str(feature))
    elif 'output_sample_rate:' in parameters[i]:
        output_sample_rate = int(parameters[i][len('output_sample_rate:'):-1])
        print(str(output_sample_rate))

print('Logging classifier model...')
print('Testing: ' + target_dataset_name)
CH818_R2_pearsonr_max = dict(zip(emotions, np.zeros((len(emotions)))))
R2_pearsonr_max = dict(zip(emotions, np.zeros((len(emotions)))))
Train_Y_valence = np.load(save_path + '/' + target_dataset_name + '/Train_Y_valence.npy')
Train_Y_arousal = np.load(save_path + '/' + target_dataset_name + '/Train_Y_arousal.npy')
Train_X = np.load(save_path + '/' + target_dataset_name +
                  '/Train_X' + '@' + str(output_sample_rate) + 'Hz_' + feature + '.npy')
print('Train_Y_valence: ' + str(Train_Y_valence.shape))
print('Train_Y_arousal: ' + str(Train_Y_arousal.shape))
print('Train_X: ' + str(Train_X.shape))
for emotion in emotions:
    if emotion == 'valence':
        Y_true = Train_Y_valence
    elif emotion == 'arousal':
        Y_true = Train_Y_arousal
    print(emotion)
    if os.path.exists(source_execute_name + '/' + emotion + '/log_0_logs.json'):
        with open(source_execute_name + '/' + emotion + '/log_0_logs.json', "r") as fb:
            data = json.load(fb)
            R2_pearsonr_max[emotion] = max(data['train_R2_pearsonr'])
    for root, subdirs, files in os.walk(source_execute_name + '/' + emotion):
        for f in files:
            if os.path.splitext(f)[1] == '.h5' and 'train_R2pr_'+format(R2_pearsonr_max[emotion], '.5f') in f:
                print(f)
                model = load_model(os.path.join(root, f), custom_objects={'Melspectrogram': Melspectrogram,
                                                                          'R2': metric.R2})
                Y_predict = model.predict(Train_X)
                Y_true = Y_true.reshape(-1, 1)
                CH818_R2_pearsonr_max[emotion] = np.square(pearsonr(Y_true, Y_predict)[0][0])
                print(CH818_R2_pearsonr_max[emotion])
print('CH818 maximum:')
print('R2_pearsonr_max: ' + str(CH818_R2_pearsonr_max))
print('Averaging Valence R2_pearsonr_max: ' + str(np.mean(CH818_R2_pearsonr_max['valence'])))
print('Averaging Arousal R2_pearsonr_max: ' + str(np.mean(CH818_R2_pearsonr_max['arousal'])))
data = OrderedDict()
data.update({"R2_pearsonr_max": [[emotions[0]], [CH818_R2_pearsonr_max[emotions[0]]],
                                 [emotions[1]], [CH818_R2_pearsonr_max[emotions[1]]]]})
save_data(source_execute_name + '/' + target_dataset_name + '_regressor_' + action + '.xls', data)

print('Logging domain transfer model...')
print('Testing: (adapted)' + target_dataset_name)
CH818_R2_pearsonr_max = dict(zip(emotions, np.zeros((len(emotions)))))
source_R2_pearsonr_max = dict(zip(emotions, np.zeros((len(emotions)))))
target_R2_pearsonr_max = dict(zip(emotions, np.zeros((len(emotions)))))
print('Train_Y_valence: ' + str(Train_Y_valence.shape))
print('Train_Y_arousal: ' + str(Train_Y_arousal.shape))
print('Train_X: ' + str(Train_X.shape))
for emotion in emotions:
    if emotion == 'valence':
        Y_true = Train_Y_valence
    elif emotion == 'arousal':
        Y_true = Train_Y_arousal
    print(emotion)
    print('Loading target feature extractor_classifier model...')
    if os.path.exists(transfer_execute_name + '/' + emotion + '/log_0_logs.json'):
        with open(transfer_execute_name + '/' + emotion + '/log_0_logs.json', "r") as fb:
            data = json.load(fb)
            target_R2_pearsonr_max[emotion] = max(data['val_R2_pearsonr'])
    for root, subdirs, files in os.walk(transfer_execute_name + '/' + emotion):
        for f in files:
            if os.path.splitext(f)[1] == '.h5' and 'R2pr_' + format(target_R2_pearsonr_max[emotion], '.5f') in f:
                print(f)
                target_classifier_model = load_model(os.path.join(root, f),
                                                     custom_objects={'Melspectrogram': Melspectrogram,
                                                                     'R2': metric.R2})
                Y_predict = target_classifier_model.predict(Train_X)
                Y_true = Y_true.reshape(-1, 1)
                CH818_R2_pearsonr_max[emotion] = np.square(pearsonr(Y_true, Y_predict)[0][0])
                print(target_dataset_name + ': ' + str(CH818_R2_pearsonr_max[emotion]))
print('CH818 maximum:')
print('R2_pearsonr_max: ' + str(CH818_R2_pearsonr_max))
print('Averaging Valence R2_pearsonr_max: ' + str(np.mean(CH818_R2_pearsonr_max['valence'])))
print('Averaging Arousal R2_pearsonr_max: ' + str(np.mean(CH818_R2_pearsonr_max['arousal'])))
data = OrderedDict()
data.update({"R2_pearsonr_max": [[emotions[0]], [CH818_R2_pearsonr_max[emotions[0]]],
                                 [emotions[1]], [CH818_R2_pearsonr_max[emotions[1]]]]})
save_data(transfer_execute_name + '/' + target_dataset_name + '_transfer_regressor_' + action + '.xls', data)
