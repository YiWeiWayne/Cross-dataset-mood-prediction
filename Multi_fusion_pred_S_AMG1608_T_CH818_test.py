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
from keras.layers import Input, concatenate
from keras.optimizers import SGD, Adam
from kapre.time_frequency import Melspectrogram
from keras import backend as K
from functions.Custom_layers import Std2DLayer
from keras.utils.vis_utils import plot_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


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
source_dataset_name = 'AMG_1608'
target_dataset_name = 'CH_818'
source_data_num = 1608
target_data_num = 818
algorithm = [source_dataset_name, 'WADDA']
save_path = '/mnt/data/Wayne'
# save_path = '../../Data'
# source_execute_name = save_path + '/(' + action + ')' + source_dataset_name + '_20180514.0114.37'
save_key = 'pearsonr'  # 1.R2 2.pearsonr
sim_train_key = False
loss = 'mean_squared_error'
emotions = ['valence', 'arousal']
features = ['melSpec_lw', 'pitch+lw', 'rCTA']
actions = ['melSpec_lw', 'pitch+lw', 'rCTA']
filters = dict(zip(features, np.zeros((len(features), 5))))
kernels = dict(zip(features, np.zeros((len(features), 5, 2))))
strides = dict(zip(features, np.zeros((len(features), 5, 2))))
paddings = dict(zip(features, np.zeros((len(features), 5, 2))))
dr_rate = dict(zip(features, np.zeros((len(features), 5, 2))))
poolings = dict(zip(features, np.zeros((len(features), 5, 2))))
feature_sizes = dict(zip(features, np.zeros((len(features), 3))))
for feature in features:
    if feature == 'melSpec':
        filters[feature] = [32, 32, 32, 32, 32]
        kernels[feature] = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
        poolings[feature] = [(2, 4), (3, 4), (2, 5), (2, 4), (4, 4)]
        feature_sizes[feature] = [96, 2498, 1]
    elif feature == 'melSpec_lw':  # dim(96, 1249, 1)
        filters[feature] = [128, 128, 128, 128, 128]
        kernels[feature] = [(96, 4), (1, 4), (1, 3), (1, 3), (1, 3)]
        strides[feature] = [(1, 3), (1, 2), (1, 3), (1, 3), (1, 2)]
        paddings[feature] = ['valid', 'valid', 'valid', 'valid', 'valid']
        poolings[feature] = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 11)]
        dr_rate[feature] = [0, 0, 0, 0, 0, 0]
        feature_sizes[feature] = [96, 1249, 1]
    elif feature == 'rCTA':  # dim(30, 142, 1)
        filters[feature] = [128, 128, 128]
        kernels[feature] = [(30, 4), (1, 3), (1, 3)]
        strides[feature] = [(1, 3), (1, 2), (1, 2)]
        paddings[feature] = ['valid', 'valid', 'valid']
        poolings[feature] = [(1, 1), (1, 1), (1, 1), (1, 11)]
        dr_rate[feature] = [0, 0, 0, 0]
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
    elif feature == 'pitch+lw':  # dim(360, 1249, 1)
        filters[feature] = [128, 128, 128, 128, 128]
        kernels[feature] = [(360, 4), (1, 4), (1, 3), (1, 3), (1, 3)]
        strides[feature] = [(1, 3), (1, 2), (1, 3), (1, 3), (1, 2)]
        paddings[feature] = ['valid', 'valid', 'valid', 'valid', 'valid']
        poolings[feature] = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 11)]
        dr_rate[feature] = [0, 0, 0, 0, 0, 0]
        feature_sizes[feature] = [360, 1249, 1]

output_sample_rate = 22050
pretrain_path = dict(zip(algorithm, [['', '', ''], ['', '', '']]))
if sim_train_key:
    # simultaneous training
    pretrain_path[algorithm[0]] = [
        save_path + '/(' + actions[0] + ')' + algorithm[0] + '_S_' + source_dataset_name + '_T_'
        + target_dataset_name + '_20180614.1652.41',
        save_path + '/(' + actions[1] + ')' + algorithm[0] + '_S_' + source_dataset_name + '_T_'
        + target_dataset_name + '_20180614.1652.15',
        save_path + '/(' + actions[2] + ')' + algorithm[0] + '_S_' + source_dataset_name + '_T_'
        + target_dataset_name + '_20180614.1653.03']
    pretrain_path[algorithm[1]] = pretrain_path[algorithm[0]]
else:
    # separate training
    pretrain_path[algorithm[0]] = [
        save_path + '/(' + actions[0] + ')' + algorithm[0] + '_20180619.0827.44',
        save_path + '/(' + actions[1] + ')' + algorithm[0] + '_20180619.0827.23',
        save_path + '/(' + actions[2] + ')' + algorithm[0] + '_20180619.0819.30']
    pretrain_path[algorithm[1]] = [
        save_path + '/(' + actions[0] + ')' + algorithm[1] + '_S_' + source_dataset_name + '_T_'
        + target_dataset_name + '_20180619.1010.08',
        save_path + '/(' + actions[1] + ')' + algorithm[1] + '_S_' + source_dataset_name + '_T_'
        + target_dataset_name + '_20180619.1010.39',
        save_path + '/(' + actions[2] + ')' + algorithm[1] + '_S_' + source_dataset_name + '_T_'
        + target_dataset_name + '_20180619.1012.09']



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

# Single model
for j in range(0, 3):  # feature
    for i in range(0, len(algorithm)):  # algorithm
        print('Logging ' + algorithm[i] + ' model...')
        print('Testing: (adapted)' + target_dataset_name)
        R2_pearsonr_max = dict(zip(emotions, np.zeros((len(emotions)))))
        MAE_max = dict(zip(emotions, np.zeros((len(emotions)))))
        MSE_max = dict(zip(emotions, np.zeros((len(emotions)))))
        print('Train_Y_valence: ' + str(Train_Y['valence'].shape))
        print('Train_Y_arousal: ' + str(Train_Y['arousal'].shape))
        Y_predict = dict(zip(emotions, [['', '', ''], ['', '', '']]))
        dict_data = OrderedDict()
        if i == 0:
            data_type = 'train'
            save_type = 'source'
        else:
            data_type = 'val'
            save_type = 'target'
        for emotion in emotions:
            Y_true = Train_Y[emotion]
            Y_true = Y_true.reshape(-1, 1)
            print(emotion)
            print('Loading target feature extractor_classifier model...')
            if os.path.exists(pretrain_path[algorithm[i]][j] + '/' + emotion + '/log_0_logs.json'):
                with open(pretrain_path[algorithm[i]][j] + '/' + emotion + '/log_0_logs.json',
                          "r") as fb:
                    print(pretrain_path[algorithm[i]][j] + '/' + emotion + '/log_0_logs.json')
                    data = json.load(fb)
                    if save_key == 'pearsonr':
                        max_temp = np.square(max(data[data_type+'_pearsonr']))
                    else:
                        max_temp = max(data[data_type+'_R2_pearsonr'])
            for root, subdirs, files in os.walk(pretrain_path[algorithm[i]][j] + '/' + emotion):
                for f in files:
                    if os.path.splitext(f)[1] == '.h5' and data_type+'_R2pr_' + format(max_temp, '.5f') in f:
                        print('Source ' + actions[j])
                        print(f)
                        model = load_model(os.path.join(root, f), custom_objects={'Std2DLayer': Std2DLayer})
                        Y_predict[emotion][j] = model.predict([Train_X[features[j]]], batch_size=4)
                        Y = np.concatenate((Y_true, Y_predict[emotion][j]), axis=1)
                        dict_data.update({emotion: Y})
                        R2_pearsonr_max[emotion] = np.square(pearsonr(Y_true, Y_predict[emotion][j])[0][0])
                        MAE_max[emotion] = mean_absolute_error(Y_true, Y_predict[emotion][j])
                        MSE_max[emotion] = mean_squared_error(Y_true, Y_predict[emotion][j])
                        print(R2_pearsonr_max[emotion])
        print('CH818 maximum:')
        print('R2_pearsonr_max: ' + str(R2_pearsonr_max))
        dict_data.update({"R2_pearsonr_max": [[emotions[0]], [R2_pearsonr_max[emotions[0]]],
                                                     [emotions[1]], [R2_pearsonr_max[emotions[1]]]]})
        dict_data.update({"MAE_max": [[emotions[0]], [MAE_max[emotions[0]]],
                                             [emotions[1]], [MAE_max[emotions[1]]]]})
        dict_data.update({"MSE_max": [[emotions[0]], [MSE_max[emotions[0]]],
                                             [emotions[1]], [MSE_max[emotions[1]]]]})
        save_data(pretrain_path[algorithm[i]][j] + '/' + save_type + '_regressor_' + actions[j] + '.xls', dict_data)

# Fusion
for feature_index in [[0, 1], [1, 2], [0, 2], [0, 1, 2]]:  # fusion feature
    for i in range(0, len(algorithm)):  # algorithm
        print('Logging ' + algorithm[i] + ' model...')
        print('Testing: (adapted)' + target_dataset_name)
        R2_pearsonr_max = dict(zip(emotions, np.zeros((len(emotions)))))
        MAE_max = dict(zip(emotions, np.zeros((len(emotions)))))
        MSE_max = dict(zip(emotions, np.zeros((len(emotions)))))
        print('Train_Y_valence: ' + str(Train_Y['valence'].shape))
        print('Train_Y_arousal: ' + str(Train_Y['arousal'].shape))
        Y_predict = dict(zip(emotions, [['', '', ''], ['', '', '']]))
        dict_data = OrderedDict()
        if i == 0:
            data_type = 'train'
            save_type = 'source'
        else:
            data_type = 'val'
            save_type = 'target'
        for emotion in emotions:
            Y_true = Train_Y[emotion]
            Y_true = Y_true.reshape(-1, 1)
            print(emotion)
            print('Loading target feature extractor_classifier model...')
            for j in feature_index:
                if os.path.exists(pretrain_path[algorithm[i]][j] + '/' + emotion + '/log_0_logs.json'):
                    with open(pretrain_path[algorithm[i]][j] + '/' + emotion + '/log_0_logs.json',
                              "r") as fb:
                        print(pretrain_path[algorithm[i]][j] + '/' + emotion + '/log_0_logs.json')
                        data = json.load(fb)
                        if save_key == 'pearsonr':
                            max_temp = np.square(max(data[data_type+'_pearsonr']))
                            sign = np.sign(data[data_type+'_pearsonr'][np.argmax(data[data_type+'_pearsonr'])])
                        else:
                            max_temp = max(data[data_type+'_R2_pearsonr'])
                            sign = np.sign(data[data_type+'pearsonr'][np.argmax(data[data_type+'_R2_pearsonr'])])
                        print('max_temp:'+str(max_temp))
                        print('sign:' + str(sign))
                for root, subdirs, files in os.walk(pretrain_path[algorithm[i]][j] + '/' + emotion):
                    for f in files:
                        if os.path.splitext(f)[1] == '.h5' and data_type+'_R2pr_' + format(max_temp, '.5f') in f:
                            print(data_type + actions[j])
                            print(f)
                            model = load_model(os.path.join(root, f), custom_objects={'Std2DLayer': Std2DLayer})
                            Y_predict[emotion][j] = model.predict([Train_X[features[j]]], batch_size=4)
                            Y_predict[emotion][j] = Y_predict[emotion][j] * sign
                            print(np.square(pearsonr(Y_true, Y_predict[emotion][j])[0][0]))
            # fusion
            Y_pred = np.zeros(Y_true.shape)
            for j in feature_index:
                Y_pred = np.add(Y_predict[emotion][j], Y_pred)
            Y_pred = Y_pred/len(feature_index)
            print(Y_pred.shape)
            Y = np.concatenate((Y_true, Y_pred), axis=1)
            dict_data.update({emotion: Y})
            R2_pearsonr_max[emotion] = np.square(pearsonr(Y_true, Y_pred)[0][0])
            MAE_max[emotion] = mean_absolute_error(Y_true, Y_pred)
            MSE_max[emotion] = mean_squared_error(Y_true, Y_pred)
            print(R2_pearsonr_max[emotion])
        print('CH818 maximum:')
        print('R2_pearsonr_max: ' + str(R2_pearsonr_max))
        dict_data.update({"R2_pearsonr_max": [[emotions[0]], [R2_pearsonr_max[emotions[0]]],
                                                     [emotions[1]], [R2_pearsonr_max[emotions[1]]]]})
        dict_data.update({"MAE_max": [[emotions[0]], [MAE_max[emotions[0]]],
                                             [emotions[1]], [MAE_max[emotions[1]]]]})
        dict_data.update({"MSE_max": [[emotions[0]], [MSE_max[emotions[0]]],
                                             [emotions[1]], [MSE_max[emotions[1]]]]})
        save_data(save_path + '/' + save_type + '_regressor_' + action + '_' + str(feature_index) + '.xls', dict_data)
