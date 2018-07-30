import matplotlib
matplotlib.use('agg')
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from functions import model_structure, ADDA_funcs, callback_wayne
from keras.models import Model, load_model
from keras.layers import Input, concatenate
from keras.layers import Input, Dense
from keras.optimizers import SGD, Adam, RMSprop
import os, json
import numpy as np
import datetime
from keras.utils import generic_utils
from keras import backend as K
from functions.Custom_layers import Std2DLayer
import random
from scipy.stats import pearsonr
from keras.utils.vis_utils import plot_model
import threading
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# GPU speed limit
def get_session(gpu_fraction=0.3):
    # Assume that you have 6GB of GPU memory and want to allocate ~2GB
    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
KTF.set_session(get_session())


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

# load pretrain model
emotions = ['valence', 'arousal']
features = ['melSpec_lw', 'pitch+lw', 'rCTA']
actions = ['melSpec_lw', 'pitch+lw', 'rCTA']
action = 'FewShot'
monitor = 'val_pearsonr'
mode = 'max'
save_path = '/mnt/data/Wayne'
source_dataset_name = 'AMG_1608'
target_dataset_name = 'CH_818'
target_data_num = 818
target_sample_number = 64
output_sample_rate = 22050
save_best_only = False
save_weights_only = False
load_weights_target_feature_extractor = True
load_weights_source_classifier = True

lock_target_feature_extractor = True
lock_source_classifier = False
now = datetime.datetime.now()
localtime = str(now.year) + str(now.month).zfill(2) + str(now.day).zfill(2) + '.' +\
            str(now.hour).zfill(2) + str(now.minute).zfill(2) + '.' + str(now.second).zfill(2)
execute_name = save_path + '/(' + action + ')' + str(target_sample_number) + \
               '_lock_te_' + str(lock_target_feature_extractor) + '_sr_' + str(lock_source_classifier) + '_' + localtime
# select_best_model
observe_epoch = 2000
constraint = 'val_MSE'
algorithms = [source_dataset_name, 'WADDA']
pretrain_path = dict(zip(algorithms, np.empty(shape=(len(algorithms), 3)+(0,)).tolist()))
pretrain_path[algorithms[0]] = [
            save_path + '/(' + actions[0] + ')' + algorithms[0] + '_20180619.0827.44',
            save_path + '/(' + actions[1] + ')' + algorithms[0] + '_20180619.0827.23',
            save_path + '/(' + actions[2] + ')' + algorithms[0] + '_20180619.0819.30']
pretrain_path[algorithms[1]] = [
    save_path + '/(' + actions[0] + ')' + algorithms[1] + '_S_' + source_dataset_name + '_T_'
    + target_dataset_name + '_20180623.1045.07',
    save_path + '/(' + actions[1] + ')' + algorithms[1] + '_S_' + source_dataset_name + '_T_'
    + target_dataset_name + '_20180623.1150.00',
    save_path + '/(' + actions[2] + ')' + algorithms[1] + '_S_' + source_dataset_name + '_T_'
    + target_dataset_name + '_20180623.1153.14']
# NN structure
regressor_net = 'cnn'
units = [128, 1]
patience = []
batch_size = 16
epochs = 300
encoded_size = 128

# regressor parameters
if regressor_net == 'nn':
    regressor_units = [128, 64, 1]
    regressor_activations = ['elu', 'elu', 'tanh']
elif regressor_net == 'cnn':
    regressor_units = [64, 128, 256, 1]
    regressor_activations = ['elu', 'elu', 'elu', 'tanh']
    regressor_kernels = [8, 4, 2]
    regressor_strides = [4, 2, 1]
    regressor_paddings = ['valid', 'valid', 'valid']
    regressor_bn = False
regressor_optimizer = 'adam'
regressor_loss = 'mean_squared_error'

#  Feature extractor parameters
use_pooling = True
use_drop_out = False
use_mp = True
if use_pooling and use_mp:
    input_channel = 3
elif use_pooling and not use_mp:
    input_channel = 2
else:
    input_channel = 1
encoded_size = encoded_size*input_channel

# parameters
para_line = []
para_line.append('# setting Parameters \n')
para_line.append('action:' + action + '\n')
para_line.append('monitor:' + str(monitor) + '\n')
para_line.append('save_path:' + save_path + '\n')
para_line.append('source_dataset_name:' + source_dataset_name + '\n')
para_line.append('target_dataset_name:' + target_dataset_name + '\n')
para_line.append('target_data_num:' + str(target_data_num) + '\n')
para_line.append('target_sample_number:' + str(target_sample_number) + '\n')
para_line.append('execute_name:' + execute_name + '\n')
para_line.append('observe_epoch :' + str(observe_epoch) + '\n')
para_line.append('constraint:' + str(constraint) + '\n')
para_line.append('algorithms:' + str(algorithms) + '\n')
para_line.append('pretrain_path:' + str(pretrain_path) + '\n')
para_line.append('save_best_only:' + str(save_best_only) + '\n')
para_line.append('save_weights_only:' + str(save_weights_only) + '\n')
para_line.append('lock_source_classifier:' + str(lock_source_classifier) + '\n')
para_line.append('lock_target_feature_extractor:' + str(lock_target_feature_extractor) + '\n')
para_line.append('load_weights_source_classifier:' + str(load_weights_source_classifier) + '\n')
para_line.append('load_weights_target_feature_extractor:' + str(load_weights_target_feature_extractor) + '\n')
# network
para_line.append('\n# network Parameters \n')
para_line.append('units:' + str(units) + '\n')
para_line.append('batch_size:' + str(batch_size) + '\n')
para_line.append('epochs:' + str(epochs) + '\n')
if not os.path.exists(execute_name):
    os.makedirs(execute_name)
paraFile = open(os.path.join(execute_name, 'Parameters.txt'), 'w')
paraFile.writelines(para_line)
paraFile.close()

Y_predict = dict(zip(algorithms, np.empty(shape=(len(algorithms), 3, len(emotions))+(0,)).tolist()))
Y_true = dict(zip(algorithms, np.empty(shape=(len(algorithms), len(emotions))+(0,)).tolist()))
regressor_models = dict(zip(algorithms, np.empty(shape=(len(algorithms), 3, len(emotions))+(0,)).tolist()))
target_extractors = dict(zip(algorithms, np.empty(shape=(len(algorithms), 3, len(emotions))+(0,)).tolist()))
target_regressor_models = dict(zip(algorithms, np.empty(shape=(len(algorithms), 3, len(emotions))+(0,)).tolist()))
# Load data
Train_Y = dict(zip(emotions, np.zeros((target_data_num, 1))))
Train_Y['valence'] = np.load(save_path + '/' + target_dataset_name + '/Train_Y_valence.npy')
Train_Y['arousal'] = np.load(save_path + '/' + target_dataset_name + '/Train_Y_arousal.npy')
Train_X = dict(zip(features, np.zeros((target_data_num, 1, 1, 1))))
for feature in features:
    print(feature)
    Train_X[feature] = np.load(save_path + '/' + target_dataset_name +
                               '/Train_X@' + str(output_sample_rate) + 'Hz_' + feature + '.npy')
    print("Train_X shape:" + str(Train_X[feature].shape))

# index
if not os.path.exists(save_path + '/' + target_dataset_name + '/target_index.npy'):
    if not os.path.exists(save_path + '/' + target_dataset_name):
        os.makedirs(save_path + '/' + target_dataset_name)
    target_index = np.arange(target_data_num)
    np.random.shuffle(target_index)
    np.save(save_path + '/' + target_dataset_name + '/target_index.npy', target_index)
target_index = np.load(save_path + '/' + target_dataset_name + '/target_index.npy')

# Load pre-trained model
for i in range(len(algorithms)-1, 0, -1):        # algorithm
    if i == 0:
        data_type = 'train'
    else:
        data_type = 'val'
    for feature_index in [[0, 1, 2], [0, 1], [0, 2], [1, 2]]:
        print('Logging ' + algorithms[i] + str(feature_index) + ' model...')
        for k in range(0, len(emotions)):
            if KTF._SESSION:
                print('Reset session.')
                KTF.clear_session()
                KTF.set_session(get_session())
            target_tensor = Input(shape=(encoded_size,))
            if regressor_net == 'nn':
                regressor_model = Model(inputs=target_tensor,
                                        outputs=model_structure.nn_classifier(x=target_tensor,
                                                                              units=regressor_units,
                                                                              activations=regressor_activations))
            elif regressor_net == 'cnn':
                regressor_model = Model(inputs=target_tensor,
                                        outputs=model_structure.cnn_classifier(x=target_tensor,
                                                                               input_channel=input_channel,
                                                                               units=regressor_units,
                                                                               activations=regressor_activations,
                                                                               kernels=regressor_kernels,
                                                                               strides=regressor_strides,
                                                                               paddings=regressor_paddings,
                                                                               bn=regressor_bn))
            print("regressor_model summary:")
            with open(os.path.join(execute_name, 'regressor_model_summary.txt'), 'w') as fh:
                regressor_model.summary(print_fn=lambda x: fh.write(x + '\n'))
            for j in feature_index:
                feature = features[j]
                if feature == 'melSpec_lw':  # dim(96, 1249, 1)
                    filters = [128, 128, 128, 128, 128]
                    kernels = [(96, 4), (1, 4), (1, 3), (1, 3), (1, 3)]
                    strides = [(1, 3), (1, 2), (1, 3), (1, 3), (1, 2)]
                    paddings = ['valid', 'valid', 'valid', 'valid', 'valid']
                    poolings = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 11)]
                    dr_rate = [0, 0, 0, 0, 0, 0]
                elif feature == 'rCTA':  # dim(30, 142, 1)
                    filters = [128, 128, 128]
                    kernels = [(30, 4), (1, 3), (1, 3)]
                    strides = [(1, 3), (1, 2), (1, 2)]
                    paddings = ['valid', 'valid', 'valid']
                    poolings = [(1, 1), (1, 1), (1, 1), (1, 11)]
                    dr_rate = [0, 0, 0, 0]
                elif feature == 'pitch+lw':  # dim(360, 1249, 1)
                    filters = [128, 128, 128, 128, 128]
                    kernels = [(360, 4), (1, 4), (1, 3), (1, 3), (1, 3)]
                    strides = [(1, 3), (1, 2), (1, 3), (1, 3), (1, 2)]
                    paddings = ['valid', 'valid', 'valid', 'valid', 'valid']
                    poolings = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 11)]
                    dr_rate = [0, 0, 0, 0, 0, 0]
                regressor_models[algorithms[i]][j][k] = regressor_model
                target_feature_tensor = Input(shape=(Train_X[features[j]].shape[1], Train_X[features[j]].shape[2], 1))
                target_feature_extractor = model_structure.compact_cnn_extractor(x=target_feature_tensor,
                                                                                 filters=filters, kernels=kernels,
                                                                                 strides=strides,
                                                                                 paddings=paddings, poolings=poolings,
                                                                                 dr_rate=dr_rate,
                                                                                 use_pooling=use_pooling,
                                                                                 use_drop_out=use_drop_out,
                                                                                 use_mp=use_mp)
                target_extractor = Model(inputs=target_feature_tensor, outputs=target_feature_extractor)
                target_extractors[algorithms[i]][j][k] = target_extractor
                target_regressor_model = Model(inputs=target_feature_tensor,
                                               outputs=regressor_model(target_feature_extractor))
                target_regressor_model.compile(loss=regressor_loss, optimizer=regressor_optimizer, metrics=['accuracy'])
                target_regressor_models[algorithms[i]][j][k] = target_regressor_model
                print("target_regressor_model summary:")
                with open(os.path.join(execute_name, algorithms[i] + feature + '$target_regressor_model_summary.txt'), 'w') as fh:
                    target_regressor_model.summary(print_fn=lambda x: fh.write(x + '\n'))
                file_path = pretrain_path[algorithms[i]][j] + '/' + emotions[k]
                if os.path.exists(file_path + '/log_0_logs.json'):
                    with open(file_path + '/log_0_logs.json',
                              "r") as fb:
                        print(file_path + '/log_0_logs.json')
                        data = json.load(fb)
                max_temp = np.square(max(data[data_type + '_pearsonr'][1:observe_epoch]))
                sign = np.sign(data[data_type + '_pearsonr'][np.argmax(data[data_type + '_pearsonr'][1:observe_epoch])])
                if i == 1:
                    if np.square(max(data[data_type + '_pearsonr'][1:observe_epoch])) > np.square(
                            data[data_type + '_pearsonr'][0]):
                        mae_tmp = 1000
                        sort_tmp = -1
                        key = 0
                        while sort_tmp > -11 and key == 0:
                            if data[constraint][np.argsort(data[data_type + '_pearsonr'][1:observe_epoch])[sort_tmp]] < data[constraint][0]:
                                mae_tmp = data[constraint][np.argsort(data[data_type + '_pearsonr'][1:observe_epoch])[sort_tmp]]
                                sort_inex = sort_tmp
                                key = 1
                            if data[constraint][np.argsort(data[data_type + '_pearsonr'][1:observe_epoch])[sort_tmp]] < mae_tmp:
                                mae_tmp = data[constraint][np.argsort(data[data_type + '_pearsonr'][1:observe_epoch])[sort_tmp]]
                                sort_inex = sort_tmp
                            sort_tmp = sort_tmp - 1
                        max_temp = np.square(data[data_type + '_pearsonr'][
                                                 np.argsort(data[data_type + '_pearsonr'][1:observe_epoch])[
                                                     sort_inex] + 1])
                        sign = np.sign(data[data_type + '_pearsonr'][
                                            np.argsort(data[data_type + '_pearsonr'][1:observe_epoch])[
                                                sort_inex] + 1])
                        print(np.argsort(data[data_type + '_pearsonr'][1:observe_epoch])[sort_inex] + 1)
                print('max_temp:' + str(max_temp))
                print('sign:' + str(sign))
                for root, subdirs, files in os.walk(file_path):
                    for f in files:
                        if os.path.splitext(f)[1] == '.h5' and data_type + '_R2pr_' + format(max_temp, '.5f') in f:
                            print(algorithms[i] + actions[j])
                            print(f)
                            model = load_model(os.path.join(root, f), custom_objects={'Std2DLayer': Std2DLayer})
                            if regressor_bn:
                                reg_len = -(2 + 4) * len(regressor_units) + 4
                            else:
                                reg_len = -2 * len(regressor_units)
                            if load_weights_target_feature_extractor:
                                print('set target')
                                target_extractors[algorithms[i]][j][k].set_weights(model.get_weights()[:reg_len])
                            if load_weights_source_classifier:
                                print('set classifier')
                                regressor_models[algorithms[i]][j][k].set_weights(model.get_weights()[reg_len:])
                            # # models[algorithms[i]][j][k].save(str(i)+str(j)+str(k)+'.h5')
                            # print(len(models[algorithms[i]][j][k].layers))
                            # for layer in models[algorithms[i]][j][k].layers:
                            #     layer.name = layer.name+str(i)+str(j)+str(k)
                            # print(len(models[algorithms[i]][j][k].layers))
                            # models[algorithms[i]][j][k].save(str(i) + str(j) + str(k) + '_rename.h5')
                            Y_predict[algorithms[i]][j][k] = target_regressor_models[algorithms[i]][j][k].predict([Train_X[features[j]]], batch_size=4)
                            Y_true = Train_Y[emotions[k]].reshape(-1, 1)
                            R2 = np.square(pearsonr(Y_true, Y_predict[algorithms[i]][j][k])[0][0])
                            print(str(R2))
            # Concatenate model
            adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            regressor_loss = 'mean_squared_error'
            model_path = execute_name + '/' + algorithms[i] + '_' + emotions[k] + '_feature' + str(feature_index) + '/'
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            concat_tensor = concatenate([target_regressor_models[algorithms[i]][j][k].output for j in feature_index])
            concat_model = Model(inputs=[target_regressor_models[algorithms[i]][j][k].input for j in feature_index],
                                 outputs=concat_tensor)
            concat_model.summary()
            concat_model.save(model_path + '/init_concat.h5')
            with open(os.path.join(execute_name, 'concat_model_summary.txt'), 'w') as fh:
                # Pass the file handle in as a lambda function to make it callable
                concat_model.summary(print_fn=lambda x: fh.write(x + '\n'))
            x = model_structure.regression_classifier(concat_tensor, units)
            output_model = Model(inputs=[target_regressor_models[algorithms[i]][j][k].input for j in feature_index],
                                 outputs=x)
            output_model.summary()
            with open(os.path.join(execute_name, 'output_model_summary.txt'), 'w') as fh:
                # Pass the file handle in as a lambda function to make it callable
                output_model.summary(print_fn=lambda x: fh.write(x + '\n'))
            output_model.compile(optimizer=adam, loss=regressor_loss, metrics=['accuracy'])
            output_model.save(model_path + '/init_run.h5')
            LossR2Logger_ModelCheckPoint = callback_wayne.LossR2Logger_ModelCheckPoint(
                train_data=([Train_X[features[j]][target_index[0:target_sample_number], :] for j in feature_index],
                            Train_Y[emotions[k]][target_index[0:target_sample_number]]),
                val_data=([Train_X[features[j]][target_index[target_sample_number:], :] for j in feature_index],
                          Train_Y[emotions[k]][target_index[target_sample_number:]]),
                file_name=model_path + '/log', run_num=0,
                filepath=model_path, monitor=monitor, verbose=0,
                save_best_only=save_best_only, save_weights_only=save_weights_only,
                mode=mode, period=1)
            print('Lock source_classifier: ' + str(lock_source_classifier))
            print('Lock target_feature_extractor: ' + str(lock_target_feature_extractor))
            for j in feature_index:
                regressor_models[algorithms[i]][j][k].trainable = lock_source_classifier
                target_extractors[algorithms[i]][j][k].trainable = lock_target_feature_extractor
            output_model.fit([Train_X[features[j]][target_index[0:target_sample_number], :] for j in feature_index],
                             Train_Y[emotions[k]][target_index[0:target_sample_number]], batch_size=batch_size,
                             epochs=epochs,
                             callbacks=[LossR2Logger_ModelCheckPoint],
                             verbose=1, shuffle=True, validation_data=(
                [Train_X[features[j]][target_index[target_sample_number:], :] for j in feature_index],
                Train_Y[emotions[k]][target_index[target_sample_number:]]))
