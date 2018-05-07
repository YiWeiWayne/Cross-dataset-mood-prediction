import numpy as np
import keras.backend.tensorflow_backend as KTF
import os
import tensorflow as tf
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from functions import model_structure, callback_wayne, metric
import json
import csv
from pyexcel_xls import save_data
from collections import OrderedDict
from keras.models import Model, load_model
from keras.layers import Input, concatenate
from keras.optimizers import SGD, Adam
from kapre.time_frequency import Melspectrogram
from keras import backend as K


# GPU speed limit
def get_session(gpu_fraction=0.6):
    # Assume that you have 6GB of GPU memory and want to allocate ~2GB
    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
KTF.set_session(get_session())


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

# Parameters
action = '%ml%rc%pl'
algorithm = ['', 'ADDA', 'WADDA']
source_dataset_name = 'AMG_1608'
target_dataset_name = 'CH_818'
source_data_num = 1608
target_data_num = 818
save_path = '/mnt/data/Wayne'
source_execute_name = save_path + '/(' + action + ')' + source_dataset_name + '_20180427.1613.34'
loss = 'mean_squared_error'
emotions = ['valence', 'arousal']
features = ['melSpec_lw', 'rCTA', 'pitch+lw']
actions = ['22K+lw', 'rCTA', 'pitch+lw']
classifier_units = [1]
filters = dict(zip(features, np.zeros((len(features), 5))))
kernels = dict(zip(features, np.zeros((len(features), 5, 2))))
poolings = dict(zip(features, np.zeros((len(features), 5, 2))))
feature_sizes = dict(zip(features, np.zeros((len(features), 3))))
for feature in features:
    if feature == 'melSpec':
        filters[feature] = [32, 32, 32, 32, 32]
        kernels[feature] = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
        poolings[feature] = [(2, 4), (3, 4), (2, 5), (2, 4), (4, 4)]
        feature_sizes[feature] = [96, 2498, 1]
    elif feature == 'melSpec_lw':
        filters[feature] = [32, 32, 32, 32, 32]
        kernels[feature] = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
        poolings[feature] = [(2, 4), (3, 4), (2, 5), (2, 4), (4, 3)]
        feature_sizes[feature] = [96, 1249, 1]
    elif feature == 'rCTA':
        filters[feature] = [32, 32, 32, 32, 32]
        kernels[feature] = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
        poolings[feature] = [(1, 1), (2, 2), (2, 5), (2, 7), (3, 2)]
        feature_sizes[feature] = [30, 142, 1]
    elif feature == 'rTA':
        filters[feature] = [32, 32, 32, 32, 32]
        kernels[feature] = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
        poolings[feature] = [(4, 1), (2, 2), (5, 5), (7, 7), (2, 2)]
        feature_sizes[feature] = [571, 142, 1]
    elif feature == 'pitch':
        filters[feature] = [32, 32, 32, 32, 32]
        kernels[feature] = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
        poolings[feature] = [(5, 8), (4, 4), (2, 5), (3, 5), (3, 3)]
        feature_sizes[feature] = [360, 2498, 1]
    elif feature == 'pitch+lw':
        filters[feature] = [32, 32, 32, 32, 32]
        kernels[feature] = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
        poolings[feature] = [(5, 4), (4, 4), (2, 5), (3, 5), (3, 3)]
        feature_sizes[feature] = [360, 1249, 1]

# Load parameters
para_File = open(source_execute_name + '/Parameters.txt', 'r')
parameters = para_File.readlines()
para_File.close()
for i in range(0, len(parameters)):
    if 'output_sample_rate:' in parameters[i]:
        output_sample_rate = int(parameters[i][len('output_sample_rate:'):-1])
        print(str(output_sample_rate))
# ADDA models
pretrain_path = dict(zip(algorithm, [['', '', ''], ['', '', ''], ['', '', '']]))
pretrain_path[algorithm[0]] = [
    save_path + '/(' + actions[0] + ')' + algorithm[0] + source_dataset_name + '_20180422.1036.41',
    save_path + '/(' + actions[1] + ')' + algorithm[0] + source_dataset_name + '_20180423.1037.52',
    save_path + '/(' + actions[2] + ')' + algorithm[0] + source_dataset_name + '_20180425.1104.11']
pretrain_path[algorithm[1]] = [
    save_path + '/(' + actions[0] + ')' + algorithm[1] + '_S_' + source_dataset_name + '_T_'
    + target_dataset_name + '_20180422.1215.44',
    save_path + '/(' + actions[1] + ')' + algorithm[1] + '_S_' + source_dataset_name + '_T_'
    + target_dataset_name + '_20180423.1056.32',
    save_path + '/(' + actions[2] + ')' + algorithm[1] + '_S_' + source_dataset_name + '_T_'
    + target_dataset_name + '_20180425.2036.06']
pretrain_path[algorithm[2]] = [
    save_path + '/(' + actions[0] + ')' + algorithm[2] + '_S_' + source_dataset_name + '_T_'
    + target_dataset_name + '_20180426.1616.43',
    save_path + '/(' + actions[1] + ')' + algorithm[2] + '_S_' + source_dataset_name + '_T_'
    + target_dataset_name + '_20180426.1829.25',
    save_path + '/(' + actions[2] + ')' + algorithm[2] + '_S_' + source_dataset_name + '_T_'
    + target_dataset_name + '_20180427.0935.09']

# Source regressor
print('Logging classifier model...')
print('Testing: ' + target_dataset_name)
CH818_R2_pearsonr_max = dict(zip(emotions, np.zeros((len(emotions)))))
R2_pearsonr_max = dict(zip(emotions, np.zeros((len(emotions)))))

# load data
Train_Y = dict(zip(emotions, np.zeros((source_data_num, 1))))
Train_Y['valence'] = np.load(save_path + '/' + target_dataset_name + '/Train_Y_valence.npy')
Train_Y['arousal'] = np.load(save_path + '/' + target_dataset_name + '/Train_Y_arousal.npy')
Train_X = dict(zip(features, np.zeros((source_data_num, 1, 1, 1))))
for feature in features:
    print(feature)
    Train_X[feature] = np.load(save_path + '/' + target_dataset_name +
                               '/Train_X@' + str(output_sample_rate) + 'Hz_' + feature + '.npy')
    print("Train_X shape:" + str(Train_X[feature].shape))
for i in range(0, 3):
    print('Logging ' + algorithm[i] + ' model...')
    print('Testing: (adapted)' + target_dataset_name)
    CH818_R2_pearsonr_max = dict(zip(emotions, np.zeros((len(emotions)))))
    source_R2_pearsonr_max = dict(zip(emotions, np.zeros((len(emotions)))))
    target_R2_pearsonr_max = dict(zip(emotions, np.zeros((len(emotions)))))
    print('Train_Y_valence: ' + str(Train_Y['valence'].shape))
    print('Train_Y_arousal: ' + str(Train_Y['arousal'].shape))
    Y_predict = dict(zip(emotions, [['', '', ''], ['', '', '']]))
    R2pr = dict(zip(emotions, [['', '', ''], ['', '', '']]))
    for emotion in emotions:
        Y_true = Train_Y[emotion]
        Y_true = Y_true.reshape(-1, 1)
        print(emotion)
        print('Loading target feature extractor_classifier model...')
        for j in range(0, 3):
            if os.path.exists(pretrain_path[algorithm[i]][j] + '/' + emotion + '/log_0_logs.json'):
                with open(pretrain_path[algorithm[i]][j] + '/' + emotion + '/log_0_logs.json',
                          "r") as fb:
                    print(pretrain_path[algorithm[i]][j] + '/' + emotion + '/log_0_logs.json')
                    data = json.load(fb)
                    if i == 0:
                        max_temp = max(data['train_R2_pearsonr'])
                    else:
                        max_temp = max(data['val_R2_pearsonr'])
            for root, subdirs, files in os.walk(pretrain_path[algorithm[i]][j] + '/' + emotion):
                for f in files:
                    if os.path.splitext(f)[1] == '.h5' and 'R2pr_' + format(max_temp, '.5f') in f:
                        print(f)
                        model = load_model(os.path.join(root, f), custom_objects={'Melspectrogram': Melspectrogram,
                                                                                  'R2': metric.R2,
                                                                                  'R2pr': metric.R2pr})
                        Y_predict[emotion][j] = model.predict([Train_X[features[j]]], batch_size=4)
                        print(np.square(pearsonr(Y_true, Y_predict[emotion][j])[0][0]))
        Y_pred = np.mean([Y_predict[emotion][0], Y_predict[emotion][1], Y_predict[emotion][2]], axis=0)
        print(Y_pred.shape)
        CH818_R2_pearsonr_max[emotion] = np.square(pearsonr(Y_true, Y_pred)[0][0])
        print(target_dataset_name + ': ' + str(CH818_R2_pearsonr_max[emotion]))
    print('CH818 maximum:')
    print('R2_pearsonr_max: ' + str(CH818_R2_pearsonr_max))
    print('Averaging Valence R2_pearsonr_max: ' + str(np.mean(CH818_R2_pearsonr_max['valence'])))
    print('Averaging Arousal R2_pearsonr_max: ' + str(np.mean(CH818_R2_pearsonr_max['arousal'])))
    data = OrderedDict()
    data.update({"R2_pearsonr_max": [[emotions[0]], [CH818_R2_pearsonr_max[emotions[0]]],
                                     [emotions[1]], [CH818_R2_pearsonr_max[emotions[1]]]]})
    save_data(source_execute_name + '/' + target_dataset_name + '_' + algorithm[i]
              + '_regressor_' + action + '.xls', data)
