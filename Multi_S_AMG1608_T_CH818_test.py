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
algorithm = ['ADDA', 'WADDA']
source_dataset_name = 'AMG_1608'
target_dataset_name = 'CH_818'
source_data_num = 1608
target_data_num = 818
save_path = '/mnt/data/Wayne'
source_execute_name = save_path + '/(' + action + ')' + source_dataset_name + '_20180506.2114.37'
loss = 'mean_squared_error'
emotions = ['valence', 'arousal']
features = ['melSpec_lw', 'rCTA', 'pitch+lw']
actions = ['22K+lw', 'rCTA', 'pitch+lw']
enforced_regressor = True
regressor_units = [96, 96, 1]
if enforced_regressor:
    regressor_units = [32, 32, 1]
    regressor_kernels = [3, 3]
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
for emotion in emotions:
    Y_true = Train_Y[emotion]
    print(emotion)
    if os.path.exists(source_execute_name + '/' + emotion + '/log_0_logs.json'):
        with open(source_execute_name + '/' + emotion + '/log_0_logs.json', "r") as fb:
            data = json.load(fb)
            R2_pearsonr_max[emotion] = max(data['train_R2_pearsonr'])
    for root, subdirs, files in os.walk(source_execute_name + '/' + emotion):
        for f in files:
            if os.path.splitext(f)[1] == '.h5' and 'train_R2pr_'+format(R2_pearsonr_max[emotion], '.5f') in f:
                print(f)
                model = load_model(os.path.join(root, f))
                Y_predict = model.predict([Train_X[features[0]], Train_X[features[1]], Train_X[features[2]]],
                                          batch_size=4)
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

# ADDA models
pretrain_path = dict(zip(algorithm, [['', '', ''], ['', '', '']]))
pretrain_path[algorithm[0]] = [save_path + '/(' + actions[0] + ')' + algorithm[0] + '_S_' + source_dataset_name + '_T_'
                               + target_dataset_name + '_20180422.1215.44',
                               save_path + '/(' + actions[1] + ')' + algorithm[0] + '_S_' + source_dataset_name + '_T_'
                               + target_dataset_name + '_20180423.1056.32',
                               save_path + '/(' + actions[2] + ')' + algorithm[0] + '_S_' + source_dataset_name + '_T_'
                               + target_dataset_name + '_20180425.2036.06']
pretrain_path[algorithm[1]] = [save_path + '/(' + actions[0] + ')' + algorithm[1] + '_S_' + source_dataset_name + '_T_'
                               + target_dataset_name + '_20180426.1616.43',
                               save_path + '/(' + actions[1] + ')' + algorithm[1] + '_S_' + source_dataset_name + '_T_'
                               + target_dataset_name + '_20180426.1829.25',
                               save_path + '/(' + actions[2] + ')' + algorithm[1] + '_S_' + source_dataset_name + '_T_'
                               + target_dataset_name + '_20180427.0935.09']

# generate model
model_extract = ['', '', '']
feature_tensor0 = Input(shape=(feature_sizes[features[0]][0], feature_sizes[features[0]][1],
                               feature_sizes[features[0]][2]))
extractor0 = model_structure.compact_cnn_extractor(feature_tensor=feature_tensor0, filters=filters[features[0]],
                                                   kernels=kernels[features[0]], poolings=poolings[features[0]])
model_extract[0] = Model(feature_tensor0, extractor0)
feature_tensor1 = Input(shape=(feature_sizes[features[1]][0], feature_sizes[features[1]][1],
                               feature_sizes[features[1]][2]))
extractor1 = model_structure.compact_cnn_extractor(feature_tensor=feature_tensor1, filters=filters[features[1]],
                                                   kernels=kernels[features[1]], poolings=poolings[features[1]])
model_extract[1] = Model(feature_tensor1, extractor1)
feature_tensor2 = Input(shape=(feature_sizes[features[2]][0], feature_sizes[features[2]][1],
                               feature_sizes[features[2]][2]))
extractor2 = model_structure.compact_cnn_extractor(feature_tensor=feature_tensor2, filters=filters[features[2]],
                                                   kernels=kernels[features[2]], poolings=poolings[features[2]])
model_extract[2] = Model(feature_tensor2, extractor2)
extractor = concatenate([extractor0, extractor1, extractor2])
if enforced_regressor:
    regressor = model_structure.enforced_regression_classifier(x=extractor,
                                                               units=regressor_units,
                                                               kernels=regressor_kernels)
else:
    regressor = model_structure.regression_classifier(x=extractor, units=regressor_units)
model = Model(inputs=[feature_tensor0, feature_tensor1, feature_tensor2], outputs=regressor)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
model.summary()

for i in range(0, 2):
    print('Logging ' + algorithm[i] + ' model...')
    print('Testing: (adapted)' + target_dataset_name)
    CH818_R2_pearsonr_max = dict(zip(emotions, np.zeros((len(emotions)))))
    source_R2_pearsonr_max = dict(zip(emotions, np.zeros((len(emotions)))))
    target_R2_pearsonr_max = dict(zip(emotions, np.zeros((len(emotions)))))
    print('Train_Y_valence: ' + str(Train_Y['valence'].shape))
    print('Train_Y_arousal: ' + str(Train_Y['arousal'].shape))
    if os.path.exists(source_execute_name):
        for emotion in emotions:
            Y_true = Train_Y[emotion]
            print(emotion)
            print('Loading target feature extractor_classifier model...')
            if os.path.exists(source_execute_name + '/' + emotion + '/log_0_logs.json'):
                with open(source_execute_name + '/' + emotion + '/log_0_logs.json', "r") as fb:
                    data = json.load(fb)
                    target_R2_pearsonr_max[emotion] = max(data['val_R2_pearsonr'])
                    print(target_R2_pearsonr_max[emotion])
            for root, subdirs, files in os.walk(source_execute_name + '/' + emotion):
                for f in files:
                    if os.path.splitext(f)[1] == '.h5' and 'R2pr_' + format(target_R2_pearsonr_max[emotion], '.5f') in f:
                        print(f)
                        target_classifier_model = load_model(os.path.join(root, f))
                        # load regressor weights
                        model.set_weights(target_classifier_model.get_weights())
                        for j in range(0, 3):
                            if os.path.exists(pretrain_path[algorithm[i]][j] + '/' + emotion + '/log_0_logs.json'):
                                with open(pretrain_path[algorithm[i]][j] + '/' + emotion + '/log_0_logs.json',
                                          "r") as fb:
                                    data = json.load(fb)
                                    max_temp = max(data['val_R2_pearsonr'])
                                    print(max_temp)
                            for root, subdirs, files in os.walk(pretrain_path[algorithm[i]][j] + '/' + emotion):
                                for f in files:
                                    if os.path.splitext(f)[1] == '.h5' and 'R2pr_' + format(max_temp, '.5f') in f:
                                        print(f)
                                        pre_model = load_model(os.path.join(root, f))
                                        print(len(pre_model.get_weights()[:-2]))
                                        model_extract[i].set_weights(pre_model.get_weights()[:-2])
                        Y_predict = model.predict([Train_X[features[0]], Train_X[features[1]], Train_X[features[2]]],
                                                  batch_size=4)
                        Y_true = Y_true.reshape(-1, 1)
                        if emotion == 'arousal' and i == 0:
                            print('save log by target!')
                            np.save('pred.npy', Y_predict)
                            np.save('true.npy', Y_true)
                        CH818_R2_pearsonr_max[emotion] = np.square(pearsonr(Y_true, Y_predict)[0][0])
                        if np.isnan(CH818_R2_pearsonr_max[emotion]):
                            print('save log by nan!')
                            np.save('pred.npy', Y_predict)
                            np.save('true.npy', Y_true)
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
