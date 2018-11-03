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
observe_epoch = 2000
algorithm = ['CV' + source_dataset_name, 'WADDA', source_dataset_name]
save_path = '/mnt/data/Wayne'
save_key = 'pearsonr'  # 1.R2 2.pearsonr
loss = 'mean_squared_error'
emotions = ['valence', 'arousal']
features = ['melSpec_lw', 'pitch+lw', 'auto']
actions = features
output_sample_rate = 22050
pretrain_path = dict(zip(algorithm, np.empty(shape=(len(algorithm), 3)+(0,)).tolist()))
# Saved training path
pretrain_path[algorithm[0]] = [
    save_path + '/(' + actions[0] + ')' + algorithm[0] + '_20180628.0711.01',
    save_path + '/(' + actions[1] + ')' + algorithm[0] + '_20180628.0712.39',
    save_path + '/(' + actions[2] + ')' + algorithm[0] + '_20180807.1727.58'
]
pretrain_path[algorithm[1]] = [
    save_path + '/(' + actions[0] + ')' + algorithm[1] + '_S_' + source_dataset_name + '_T_'
    + target_dataset_name + '_20180623.1045.07',
    save_path + '/(' + actions[1] + ')' + algorithm[1] + '_S_' + source_dataset_name + '_T_'
    + target_dataset_name + '_20180623.1150.00',
    save_path + '/(' + actions[2] + ')' + algorithm[1] + '_S_' + source_dataset_name + '_T_'
    + target_dataset_name + '_20180805.0112.23'
]
pretrain_path[algorithm[2]] = [
    save_path + '/(' + actions[0] + ')' + algorithm[2] + '_20180619.0827.44',
    save_path + '/(' + actions[1] + ')' + algorithm[2] + '_20180619.0827.23',
    save_path + '/(' + actions[2] + ')' + algorithm[2] + '_20180804.1934.00'
]
constraint = 'val_MSE'

# # Single model
Y_predict = dict(zip(algorithm, np.empty(shape=(len(algorithm), 3, len(emotions), fold)+(0,)).tolist()))
Y_true = dict(zip(algorithm, np.empty(shape=(len(algorithm), len(emotions), fold)+(0,)).tolist()))
for i in range(0, len(algorithm)):        # algorithm
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
    for j in range(0, len(features)):  # feature
        print('Logging ' + algorithm[i] + ' model...')
        print('Testing: (adapted)' + load_data_name)
        R2_pearsonr_max = dict(zip(emotions, np.zeros((len(emotions), fold))))
        MAE_max = dict(zip(emotions, np.zeros((len(emotions), fold))))
        MSE_max = dict(zip(emotions, np.zeros((len(emotions), fold))))
        RMSE_max = dict(zip(emotions, np.zeros((len(emotions), fold))))
        print('Train_Y_valence: ' + str(Train_Y['valence'].shape))
        print('Train_Y_arousal: ' + str(Train_Y['arousal'].shape))
        dict_data = OrderedDict()
        if i == 2:
            data_type = 'train'
        else:
            data_type = 'val'
        for k in range(0, len(emotions)):
            tar_len = int(math.ceil(float(load_data_num) / float(fold)))
            Y = np.zeros((tar_len, 3*fold))
            for CV in range(0, fold):
                print(emotions[k] + '/CV' + str(CV))
                test = np.load(save_path + '/' + load_data_name + '/CV/test' + str(CV) + '.npy')
                Y_true[algorithm[i]][k][CV] = Train_Y[emotions[k]][test].reshape(-1, 1)
                print(emotions[k])
                print('Loading target feature extractor_classifier model...')
                if i == 0:
                    file_path = pretrain_path[algorithm[i]][j] + '/' + emotions[k] + '/CV' + str(CV)
                    if os.path.exists(file_path + '/log_CV_' + str(CV) + '_logs.json'):
                        with open(file_path + '/log_CV_' + str(CV) + '_logs.json',
                                  "r") as fb:
                            print(file_path + '/log_CV_' + str(CV) + '_logs.json')
                            data = json.load(fb)
                else:
                    file_path = pretrain_path[algorithm[i]][j] + '/' + emotions[k]
                    if os.path.exists(file_path + '/log_0_logs.json'):
                        with open(file_path + '/log_0_logs.json',
                                  "r") as fb:
                            print(file_path + '/log_0_logs.json')
                            data = json.load(fb)
                if save_key == 'pearsonr':
                    max_temp = np.square(max(data[data_type + '_pearsonr'][1:observe_epoch]))
                    sign = np.sign(data[data_type + '_pearsonr'][np.argmax(data[data_type + '_pearsonr'][1:observe_epoch])])
                else:
                    max_temp = max(data[data_type + '_R2_pearsonr'][1:observe_epoch])
                    sign = np.sign(data[data_type + '_pearsonr'][np.argmax(data[data_type + '_R2_pearsonr'][1:observe_epoch])])
                if i == 1:
                    index = np.argmax(
                        np.asarray(data['val_pearsonr'])[np.where(np.asarray(data['val_MSE'][:observe_epoch]) < data['val_MSE'][0])[0]])
                    index = np.where(np.asarray(data['val_MSE'][:observe_epoch]) < data['val_MSE'][0])[0][index]
                    max_temp = np.square(data[data_type + '_pearsonr'][index])
                    sign = np.sign(data[data_type + '_pearsonr'][index])
                    print(str(index))
                print('max_temp:' + str(max_temp))
                print('sign:' + str(sign))
                for root, subdirs, files in os.walk(file_path):
                    for f in files:
                        if os.path.splitext(f)[1] == '.h5' and data_type+'_R2pr_' + format(max_temp, '.5f') in f:
                            print(algorithm[i] + actions[j])
                            print(f)
                            if i != 0 and CV > 0:
                                print('Already load model.')
                            else:
                                model = load_model(os.path.join(root, f), custom_objects={'Std2DLayer': Std2DLayer})
                            Y_predict[algorithm[i]][j][k][CV] = model.predict([Train_X[features[j]][test, :]], batch_size=4)
                            index = np.ones(len(Y_true[algorithm[i]][k][CV]))*CV
                            index = index.reshape(-1, 1)
                            print(Y_true[algorithm[i]][k][CV].shape)
                            print(Y_predict[algorithm[i]][j][k][CV].shape)
                            print(index.shape)
                            Y[:len(index), CV * 3] = index[:, 0]
                            Y[:len(index), CV * 3 + 1] = Y_true[algorithm[i]][k][CV][:, 0]
                            Y[:len(index), CV * 3 + 2] = Y_predict[algorithm[i]][j][k][CV][:, 0]
                            print(Y.shape)
                            R2_pearsonr_max[emotions[k]][CV] = np.square(pearsonr(Y_true[algorithm[i]][k][CV],
                                                                                  Y_predict[algorithm[i]][j][k][CV])[0][0])
                            MAE_max[emotions[k]][CV] = mean_absolute_error(Y_true[algorithm[i]][k][CV],
                                                                           Y_predict[algorithm[i]][j][k][CV])
                            MSE_max[emotions[k]][CV] = mean_squared_error(Y_true[algorithm[i]][k][CV],
                                                                          Y_predict[algorithm[i]][j][k][CV])
                            RMSE_max[emotions[k]][CV] = np.sqrt(mean_squared_error(Y_true[algorithm[i]][k][CV],
                                                                                   Y_predict[algorithm[i]][j][k][CV]))
                            print(R2_pearsonr_max[emotions[k]][CV])
            dict_data.update({emotions[k]: Y})
        print('CH818 maximum:')
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
        dict_data.update({"RMSE_max": [[emotions[0]], RMSE_max[emotions[0]],
                                       ['average', np.mean(RMSE_max[emotions[0]])],
                                       [emotions[1]], RMSE_max[emotions[1]],
                                       ['average', np.mean(RMSE_max[emotions[1]])]]})
        save_data(algorithm[i] + str(observe_epoch) + constraint + '_regressor_' + actions[j] + '.xls', dict_data)
np.save(save_path + '/' + str(observe_epoch) + constraint + 'Y_true_CV.npy', Y_true)
np.save(save_path + '/' + str(observe_epoch) + constraint + 'Y_predict_CV.npy', Y_predict)

# Fusion
Y_true = np.load(save_path + '/' + str(observe_epoch) + constraint + 'Y_true_CV.npy')
Y_true = Y_true.item()
Y_predict = np.load(save_path + '/' + str(observe_epoch) + constraint + 'Y_predict_CV.npy')
Y_predict = Y_predict.item()
print(str(len(Y_true)))
for i in range(0, len(algorithm)):
    if i == 0:
        load_data_num = source_data_num
        load_data_name = source_dataset_name
    else:
        load_data_num = target_data_num
        load_data_name = target_dataset_name
    for feature_index in [[0, 1, 2], [0, 1], [0, 2], [1, 2]]:
        print('Logging ' + algorithm[i] + str(feature_index) + ' model...')
        R2_pearsonr_max = dict(zip(emotions, np.zeros((len(emotions), fold))))
        MAE_max = dict(zip(emotions, np.zeros((len(emotions), fold))))
        MSE_max = dict(zip(emotions, np.zeros((len(emotions), fold))))
        RMSE_max = dict(zip(emotions, np.zeros((len(emotions), fold))))
        dict_data = OrderedDict()
        for k in range(0, len(emotions)):
            tar_len = int(math.ceil(float(load_data_num) / float(fold)))
            Y = np.zeros((tar_len, 3 * fold))
            for CV in range(0, fold):
                Y_pred = np.zeros(Y_true[algorithm[i]][k][CV].shape)
                for j in feature_index:
                    Y_pred = np.add(Y_predict[algorithm[i]][j][k][CV], Y_pred)
                Y_pred = Y_pred/len(feature_index)
                index = np.ones(len(Y_true[algorithm[i]][k][CV])) * CV
                index = index.reshape(-1, 1)
                print(Y_true[algorithm[i]][k][CV].shape)
                print(Y_pred.shape)
                print(index.shape)
                Y[:len(index), CV * 3] = index[:, 0]
                Y[:len(index), CV * 3 + 1] = Y_true[algorithm[i]][k][CV][:, 0]
                Y[:len(index), CV * 3 + 2] = Y_pred[:, 0]
                print(Y.shape)
                R2_pearsonr_max[emotions[k]][CV] = np.square(pearsonr(Y_true[algorithm[i]][k][CV], Y_pred)[0][0])
                MAE_max[emotions[k]][CV] = mean_absolute_error(Y_true[algorithm[i]][k][CV], Y_pred)
                MSE_max[emotions[k]][CV] = mean_squared_error(Y_true[algorithm[i]][k][CV], Y_pred)
                RMSE_max[emotions[k]][CV] = np.sqrt(mean_squared_error(Y_true[algorithm[i]][k][CV], Y_pred))
                print(R2_pearsonr_max[emotions[k]][CV])
            dict_data.update({emotions[k]: Y})
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
        dict_data.update({"RMSE_max": [[emotions[0]], RMSE_max[emotions[0]],
                                       ['average', np.mean(RMSE_max[emotions[0]])],
                                       [emotions[1]], RMSE_max[emotions[1]],
                                       ['average', np.mean(RMSE_max[emotions[1]])]]})
        save_data(algorithm[i] + str(observe_epoch) + '_regressor_' + action + '_' + str(feature_index) + '.xls', dict_data)
