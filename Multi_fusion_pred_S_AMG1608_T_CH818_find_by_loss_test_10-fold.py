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
from keras.optimizers import Adam
from sklearn.model_selection import KFold
import math


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
fold = 10
algorithm = ['CV' + source_dataset_name, 'WADDA', source_dataset_name]
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
pretrain_path = dict(zip(algorithm, np.empty(shape=(len(algorithm), 3)+(0,)).tolist()))
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
        save_path + '/(' + actions[0] + ')' + algorithm[0] + '_20180628.0711.01',
        save_path + '/(' + actions[1] + ')' + algorithm[0] + '_20180628.0712.39',
        save_path + '/(' + actions[2] + ')' + algorithm[0] + '_20180628.0715.02']
    pretrain_path[algorithm[1]] = [
        save_path + '/(' + actions[0] + ')' + algorithm[1] + '_S_' + source_dataset_name + '_T_'
        + target_dataset_name + '_20180623.1045.07',
        save_path + '/(' + actions[1] + ')' + algorithm[1] + '_S_' + source_dataset_name + '_T_'
        + target_dataset_name + '_20180623.1150.00',
        save_path + '/(' + actions[2] + ')' + algorithm[1] + '_S_' + source_dataset_name + '_T_'
        + target_dataset_name + '_20180623.1153.14']
    pretrain_path[algorithm[2]] = [
            save_path + '/(' + actions[0] + ')' + algorithm[2] + '_20180619.0827.44',
            save_path + '/(' + actions[1] + ')' + algorithm[2] + '_20180619.0827.23',
            save_path + '/(' + actions[2] + ')' + algorithm[2] + '_20180619.0819.30']

# find best from log
epoch_s = [['', '', ''], ['', '', '']]
epoch_e = [['', '', ''], ['', '', '']]
th = [['', '', ''], ['', '', '']]
dis_th = [['', '', ''], ['', '', '']]
f_bias_min = [['', '', ''], ['', '', '']]
f_bias_max = [['', '', ''], ['', '', '']]
d_bias_max = [['', '', ''], ['', '', '']]
d_bias_min = [['', '', ''], ['', '', '']]
R2 = [['', '', ''], ['', '', '']]
p = [['', '', ''], ['', '', '']]
observer_target = [['', '', ''], ['', '', '']]
epoch_s = [[0, 0, 0], [0, 0, 0]]
epoch_e = [[2000, 2000, 500], [2000, 2000, 500]]
th = [[3, 3, 2], [4, 3, 3]]
dis_th = [[-5, -2, -5], [0, -5, -5]]
f_bias_min = [[3, 5, 9], [10, 7, 4]]
f_bias_max = [[4, 6, 10], [15, 8, 5]]
d_bias_min = [[0, 0, 1], [0, 2, 0]]
d_bias_max = [[5, 1, 2], [5, 3, 5]]
observe_target = [['dis', 'fake', 'fake'], ['dis', 'dis', 'dis']]
index0 = 'train_loss_fake'
index1 = 'train_loss_dis'
index2 = 'val_R2_pearsonr'
index3 = 'val_pearsonr'
for e in range(0, len(emotions)):
    for f in range(0, len(features)):
        if os.path.exists(pretrain_path[algorithm[1]][f] + '/' + emotions[e] + '/log_0_logs.json'):
            with open(pretrain_path[algorithm[1]][f] + '/' + emotions[e] + '/log_0_logs.json', "r") as fb:
                data = json.load(fb)
            thr = th[e][f]
            dis_thr = dis_th[e][f]
            observe_range = th[e][f]
            fake_bias_min = f_bias_min[e][f]
            fake_bias_max = f_bias_max[e][f]
            dis_bias_min = d_bias_min[e][f]
            dis_bias_max = d_bias_max[e][f]
            observe_start = 0
            observe_start = epoch_s[e][f]
            observe_end = epoch_e[e][f]
            best_num = epoch_e[e][f] - epoch_s[e][f]
            sort_index = np.argsort(
                -np.ones(len(data[index0][observe_start:observe_end])) * data[index0][observe_start:observe_end])
            skip_key = False
            ori_i = -1
            while (not skip_key) and (ori_i != best_num-1):
                ori_i = ori_i + 1
                i = observe_start + sort_index[ori_i]
                fake_p = 0
                dis_p = 0
                fake_n = 0
                dis_n = 0
                for j in range(i - observe_range, i):
                    if data[index0][j + 1] > data[index0][j]:
                        fake_p = fake_p + 1
                    if data[index1][j + 1] < data[index1][j]:
                        dis_n = dis_n + 1
                    if data[index0][j + 1] < data[index0][j]:
                        fake_n = fake_n + 1
                    if data[index1][j + 1] > data[index1][j]:
                        dis_p = dis_p + 1
                if observe_target[e][f] == 'fake':
                    if fake_p >= thr and data[index1][i] >= dis_thr and \
                                            fake_bias_min <= np.abs(
                                                data[index0][i] - data[index0][i - observe_range]) <= fake_bias_max and \
                                            dis_bias_min <= np.abs(
                                                data[index1][i] - data[index1][i - observe_range]) <= dis_bias_max:
                        skip_key = True
                        R2[e][f] = data[index2][i]
                        p[e][f] = data[index3][i]
                        print(emotions[e] + '/' + features[f] + '/' + str(i) + '/' + str(R2[e][f]) + '/' + str(p[e][f]))
                elif observe_target[e][f] == 'dis':
                    if dis_p >= thr and data[index1][i] >= dis_thr and \
                                            fake_bias_min <= np.abs(
                                                data[index0][i] - data[index0][i - observe_range]) <= fake_bias_max and \
                                            dis_bias_min <= np.abs(
                                                data[index1][i] - data[index1][i - observe_range]) <= dis_bias_max:
                        skip_key = True
                        R2[e][f] = data[index2][i]
                        p[e][f] = data[index3][i]
                        print(emotions[e] + '/' + features[f] + '/' + str(i) + '/' + str(R2[e][f]) + '/' + str(p[e][f]))
            else:
                print('OK/' + str(ori_i))

# # Source regressor
# print('Logging classifier model...')
# print('Testing: ' + target_dataset_name)
#
#
#
# if not os.path.exists(save_path + '/' + target_dataset_name + '/CV/train0.npy'):
#     os.makedirs(save_path + '/' + target_dataset_name + '/CV')
#     kf = KFold(n_splits=fold, shuffle=True)
#     for i, (train, test) in enumerate(kf.split(Train_Y['valence'])):
#         np.save(save_path + '/' + target_dataset_name + '/CV/train' + str(i) + '.npy', train)
#         np.save(save_path + '/' + target_dataset_name + '/CV/test' + str(i) + '.npy', test)


# # # Single model
# for i in range(0, len(algorithm)):        # algorithm
#     if i == 0:
#         load_data_num = source_data_num
#         load_data_name = source_dataset_name
#     else:
#         load_data_num = target_data_num
#         load_data_name = target_dataset_name
#     # load data
#     Train_Y = dict(zip(emotions, np.zeros((load_data_num, 1))))
#     Train_Y['valence'] = np.load(save_path + '/' + load_data_name + '/Train_Y_valence.npy')
#     Train_Y['arousal'] = np.load(save_path + '/' + load_data_name + '/Train_Y_arousal.npy')
#     Train_X = dict(zip(features, np.zeros((load_data_num, 1, 1, 1))))
#     for feature in features:
#         print(feature)
#         Train_X[feature] = np.load(save_path + '/' + load_data_name +
#                                    '/Train_X@' + str(output_sample_rate) + 'Hz_' + feature + '.npy')
#         print("Train_X shape:" + str(Train_X[feature].shape))
#     for j in range(0, 3):  # feature
#         print('Logging ' + algorithm[i] + ' model...')
#         print('Testing: (adapted)' + load_data_name)
#         R2_pearsonr_max = dict(zip(emotions, np.zeros((len(emotions), fold))))
#         MAE_max = dict(zip(emotions, np.zeros((len(emotions), fold))))
#         MSE_max = dict(zip(emotions, np.zeros((len(emotions), fold))))
#         print('Train_Y_valence: ' + str(Train_Y['valence'].shape))
#         print('Train_Y_arousal: ' + str(Train_Y['arousal'].shape))
#         Y_predict = dict(zip(emotions, np.empty(shape=(len(emotions), fold)+(0,)).tolist()))
#         Y_true = dict(zip(emotions, np.empty(shape=(len(emotions), fold)+(0,)).tolist()))
#         dict_data = OrderedDict()
#         if i == 0:
#             data_type = 'train'
#         else:
#             data_type = 'val'
#         for emotion in emotions:
#             tar_len = int(math.ceil(float(load_data_num) / float(fold)))
#             print(tar_len)
#             print(3*fold)
#             Y = np.zeros((tar_len, 3*fold))
#             for CV in range(0, fold):
#                 print(emotion + '/CV' + str(CV))
#                 test = np.load(save_path + '/' + load_data_name + '/CV/test' + str(CV) + '.npy')
#                 Y_true[emotion][CV] = Train_Y[emotion]
#                 Y_true[emotion][CV] = Y_true[emotion][CV][test]
#                 Y_true[emotion][CV] = Y_true[emotion][CV].reshape(-1, 1)
#                 print(emotion)
#                 print('Loading target feature extractor_classifier model...')
#                 if i == 0:
#                     file_path = pretrain_path[algorithm[i]][j] + '/' + emotion + '/CV' + str(CV)
#                     if os.path.exists(file_path + '/log_CV_' + str(CV) + '_logs.json'):
#                         with open(file_path + '/log_CV_' + str(CV) + '_logs.json',
#                                   "r") as fb:
#                             print(file_path + '/log_CV_' + str(CV) + '_logs.json')
#                             data = json.load(fb)
#                 else:
#                     file_path = pretrain_path[algorithm[i]][j] + '/' + emotion
#                     if os.path.exists(file_path + '/log_0_logs.json'):
#                         with open(file_path + '/log_0_logs.json',
#                                   "r") as fb:
#                             print(file_path + '/log_0_logs.json')
#                             data = json.load(fb)
#                 if save_key == 'pearsonr':
#                     max_temp = np.square(max(data[data_type + '_pearsonr'][1:]))
#                     sign = np.sign(data[data_type + '_pearsonr'][np.argmax(data[data_type + '_pearsonr'][1:])])
#                 else:
#                     max_temp = max(data[data_type + '_R2_pearsonr'][1:])
#                     sign = np.sign(data[data_type + '_pearsonr'][np.argmax(data[data_type + '_R2_pearsonr'][1:])])
#                 if i == 1 and emotion == 'valence':
#                     max_temp = R2[0][j]
#                     sign = np.sign(p[0][j])
#                 elif i == 1 and emotion == 'arousal':
#                     max_temp = R2[1][j]
#                     sign = np.sign(p[1][j])
#                 print('max_temp:' + str(max_temp))
#                 print('sign:' + str(sign))
#                 for root, subdirs, files in os.walk(file_path):
#                     for f in files:
#                         if os.path.splitext(f)[1] == '.h5' and data_type+'_R2pr_' + format(max_temp, '.5f') in f:
#                             print(algorithm[i] + actions[j])
#                             print(f)
#                             model = load_model(os.path.join(root, f), custom_objects={'Std2DLayer': Std2DLayer})
#                             Y_predict[emotion][CV] = model.predict([Train_X[features[j]][test, :, :]], batch_size=4)
#                             index = np.ones(len(Y_true[emotion][CV]))*CV
#                             index = index.reshape(-1, 1)
#                             print(Y_true[emotion][CV].shape)
#                             print(Y_predict[emotion][CV].shape)
#                             print(index.shape)
#                             Y[:len(index), CV * 3] = index[:, 0]
#                             Y[:len(index), CV * 3 + 1] = Y_true[emotion][CV][:, 0]
#                             Y[:len(index), CV * 3 + 2] = Y_predict[emotion][CV][:, 0]
#                             print(Y.shape)
#                             R2_pearsonr_max[emotion][CV] = np.square(pearsonr(Y_true[emotion][CV],
#                                                                               Y_predict[emotion][CV])[0][0])
#                             MAE_max[emotion][CV] = mean_absolute_error(Y_true[emotion][CV], Y_predict[emotion][CV])
#                             MSE_max[emotion][CV] = mean_squared_error(Y_true[emotion][CV], Y_predict[emotion][CV])
#                             print(R2_pearsonr_max[emotion][CV])
#             dict_data.update({emotion: Y})
#         print('CH818 maximum:')
#         print('R2_pearsonr_max: ' + str(R2_pearsonr_max))
#         dict_data.update({"R2_pearsonr_max": [[emotions[0]], R2_pearsonr_max[emotions[0]],
#                                               ['average', np.mean(R2_pearsonr_max[emotions[0]])],
#                                               [emotions[1]], R2_pearsonr_max[emotions[1]],
#                                               ['average', np.mean(R2_pearsonr_max[emotions[1]])]]})
#         dict_data.update({"MAE_max": [[emotions[0]], MAE_max[emotions[0]],
#                                       ['average', np.mean(MAE_max[emotions[0]])],
#                                       [emotions[1]], MAE_max[emotions[1]],
#                                       ['average', np.mean(MAE_max[emotions[1]])]]})
#         dict_data.update({"MSE_max": [[emotions[0]], MSE_max[emotions[0]],
#                                       ['average', np.mean(MSE_max[emotions[0]])],
#                                       [emotions[1]], MSE_max[emotions[1]],
#                                       ['average', np.mean(MSE_max[emotions[1]])]]})
#         save_data(pretrain_path[algorithm[i]][j] + '/' + algorithm[i] + '_regressor_' + actions[j] + '.xls', dict_data)

# Fusion
for i in range(0, len(algorithm)):  # algorithm
    if i == 0:
        load_data_num = source_data_num
        load_data_name = source_dataset_name
    else:
        load_data_num = target_data_num
        load_data_name = target_dataset_name
    # load data
    Train_Y = dict(zip(emotions, np.zeros((load_data_num, 1))))
    Train_Y['valence'] = np.load(save_path + '/' + load_data_name + '/Train_Y_valence.npy')
    Train_Y['arousal'] = np.load(save_path + '/' + load_data_name + '/Train_Y_arousal.npy')
    Train_X = dict(zip(features, np.zeros((load_data_num, 1, 1, 1))))
    for feature in features:
        print(feature)
        Train_X[feature] = np.load(save_path + '/' + load_data_name +
                                   '/Train_X@' + str(output_sample_rate) + 'Hz_' + feature + '.npy')
        print("Train_X shape:" + str(Train_X[feature].shape))
    for feature_index in [[0, 1, 2], [0, 1], [1, 2], [0, 2]]:  # fusion feature
        print('Logging ' + algorithm[i] + str(feature_index) + ' model...')
        print('Testing: (adapted)' + load_data_name)
        R2_pearsonr_max = dict(zip(emotions, np.zeros((len(emotions), fold))))
        MAE_max = dict(zip(emotions, np.zeros((len(emotions), fold))))
        MSE_max = dict(zip(emotions, np.zeros((len(emotions), fold))))
        print('Train_Y_valence: ' + str(Train_Y['valence'].shape))
        print('Train_Y_arousal: ' + str(Train_Y['arousal'].shape))
        Y_predict = dict(zip(emotions, np.empty(shape=(len(emotions), 3, fold)+(0,)).tolist()))
        Y_true = dict(zip(emotions, np.empty(shape=(len(emotions), fold)+(0,)).tolist()))
        dict_data = OrderedDict()
        if i == 0:
            data_type = 'train'
        else:
            data_type = 'val'
        for emotion in emotions:
            tar_len = int(math.ceil(float(load_data_num) / float(fold)))
            print(tar_len)
            print(3 * fold)
            Y = np.zeros((tar_len, 3 * fold))
            for CV in range(0, fold):
                print(emotion + '/CV' + str(CV))
                test = np.load(save_path + '/' + load_data_name + '/CV/test' + str(CV) + '.npy')
                Y_true[emotion][CV] = Train_Y[emotion][test].reshape(-1, 1)
                for j in feature_index:
                    print(emotion)
                    print('Loading target feature extractor_classifier model...')
                    if i == 0:
                        file_path = pretrain_path[algorithm[i]][j] + '/' + emotion + '/CV' + str(CV)
                        if os.path.exists(file_path + '/log_CV_' + str(CV) + '_logs.json'):
                            with open(file_path + '/log_CV_' + str(CV) + '_logs.json',
                                      "r") as fb:
                                print(file_path + '/log_CV_' + str(CV) + '_logs.json')
                                data = json.load(fb)
                    else:
                        file_path = pretrain_path[algorithm[i]][j] + '/' + emotion
                        if os.path.exists(file_path + '/log_0_logs.json'):
                            with open(file_path + '/log_0_logs.json',
                                      "r") as fb:
                                print(file_path + '/log_0_logs.json')
                                data = json.load(fb)
                    if save_key == 'pearsonr':
                        max_temp = np.square(max(data[data_type + '_pearsonr'][1:]))
                        sign = np.sign(data[data_type + '_pearsonr'][np.argmax(data[data_type + '_pearsonr'][1:])])
                    else:
                        max_temp = max(data[data_type + '_R2_pearsonr'][1:])
                        sign = np.sign(data[data_type + '_pearsonr'][np.argmax(data[data_type + '_R2_pearsonr'][1:])])
                    if i == 1 and emotion == 'valence':
                        max_temp = R2[0][j]
                        sign = np.sign(p[0][j])
                    elif i == 1 and emotion == 'arousal':
                        max_temp = R2[1][j]
                        sign = np.sign(p[1][j])
                    print('max_temp:' + str(max_temp))
                    print('sign:' + str(sign))
                    for root, subdirs, files in os.walk(file_path):
                        for f in files:
                            if os.path.splitext(f)[1] == '.h5' and data_type+'_R2pr_' + format(max_temp, '.5f') in f:
                                print(data_type + actions[j])
                                print(f)
                                model = load_model(os.path.join(root, f), custom_objects={'Std2DLayer': Std2DLayer})
                                Y_predict[emotion][j][CV] = model.predict([Train_X[features[j]][test, :, :]], batch_size=4)
                                Y_predict[emotion][j][CV] = Y_predict[emotion][j][CV] * sign
                                print(np.square(pearsonr(Y_true[emotion][CV], Y_predict[emotion][j][CV])[0][0]))
                # fusion
                Y_pred = np.zeros(Y_true[emotion][CV].shape)
                for j in feature_index:
                    Y_pred = np.add(Y_predict[emotion][j][CV], Y_pred)
                Y_pred = Y_pred/len(feature_index)
                print(Y_pred.shape)
                index = np.ones(len(Y_true[emotion][CV])) * CV
                index = index.reshape(-1, 1)
                print(Y_true[emotion][CV].shape)
                print(Y_pred.shape)
                print(index.shape)
                Y[:len(index), CV * 3] = index[:, 0]
                Y[:len(index), CV * 3 + 1] = Y_true[emotion][CV][:, 0]
                Y[:len(index), CV * 3 + 2] = Y_pred[:, 0]
                print(Y.shape)
                R2_pearsonr_max[emotion][CV] = np.square(pearsonr(Y_true[emotion][CV], Y_pred)[0][0])
                MAE_max[emotion][CV] = mean_absolute_error(Y_true[emotion][CV], Y_pred)
                MSE_max[emotion][CV] = mean_squared_error(Y_true[emotion][CV], Y_pred)
                print(R2_pearsonr_max[emotion][CV])
            dict_data.update({emotion: Y})
        print('R2_pearsonr_max: ' + str(R2_pearsonr_max))
        dict_data.update({"R2_pearsonr_max": [[emotions[0]], R2_pearsonr_max[emotions[0]],
                                              ['average', np.mean(R2_pearsonr_max[emotions[0]])],
                                              [emotions[1]], R2_pearsonr_max[emotions[1]],
                                              ['average', np.mean(R2_pearsonr_max[emotions[1]])]]})
        dict_data.update({"MAE_max": [[emotions[0]], MAE_max[emotions[0]],
                                      ['average', np.mean(MAE_max[emotions[0]])],
                                      [emotions[1]], MAE_max[emotions[1]],
                                      ['average', np.mean(MAE_max[emotions[1]])]]})
        dict_data.update({"MSE_max": [[emotions[0]], MSE_max[emotions[0]],
                                      ['average', np.mean(MSE_max[emotions[0]])],
                                      [emotions[1]], MSE_max[emotions[1]],
                                      ['average', np.mean(MSE_max[emotions[1]])]]})
        save_data(save_path + '/' + algorithm[i] + '_regressor_' + action + '_' + str(feature_index) + '.xls', dict_data)
