import numpy as np
import keras.backend.tensorflow_backend as KTF
import os
import tensorflow as tf
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from functions import model_structure, callback_wayne, Transfer_funcs, metric, ADDA_funcs
import datetime
from keras.models import Model, load_model
from keras.layers import Input, concatenate
from keras.optimizers import Adam
from kapre.time_frequency import Melspectrogram
import json


# GPU speed limit
def get_session(gpu_fraction=1):
    # Assume that you have 6GB of GPU memory and want to allocate ~2GB
    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
KTF.set_session(get_session())


def make_trainable(model, trainable):
    model.trainable = trainable
    for l in model.layers:
        l.trainable = trainable

action = '%ml%rc%pl'
action_description = 'Change features to melSpec_lw, rCTA, pitch+lw and lock feature-extraction'
features = ('melSpec_lw', 'rCTA', 'pitch+lw')  # 1.melSpec 2.melSpec_lw 3.rCTA 4.rTA 5.pitch 6.pitch+lw
emotions = ['valence']
source_dataset_name = 'AMG_1608'
target_dataset_name = 'CH_818'
save_path = '/data/Wayne'
source_dataset_path = save_path + '/Dataset/AMG1838_original/amg1838_mp3_original'
source_label_path = save_path + '/Dataset/AMG1838_original/AMG1608/amg1608_v2.xls'
target_dataset_path = save_path + '/Dataset/CH818/mp3'
target_label_path = save_path + '/Dataset/CH818/label/CH818_Annotations.xlsx'
source_data_num = 1608
target_data_num = 818
sec_length = 29
output_sample_rate = 22050
patience = []
batch_size = 8
epochs = 200
now = datetime.datetime.now()
localtime = str(now.year) + str(now.month).zfill(2) + str(now.day).zfill(2) + '.' +\
            str(now.hour).zfill(2) + str(now.minute).zfill(2) + '.' + str(now.second).zfill(2)
execute_name = save_path + '/(' + action + ')' + source_dataset_name + '_' + localtime
loss = 'mean_squared_error'
save_best_only = False
save_weights_only = False
monitor = 'train_R2_pearsonr'
mode = 'max'
load_pretrained_weights = True
lock_feature_extraction = True
classifier_units = [96, 96, 1]
filters = dict(zip(features, np.zeros((len(features), 5))))
kernels = dict(zip(features, np.zeros((len(features), 5, 2))))
poolings = dict(zip(features, np.zeros((len(features), 5, 2))))
feature_sizes = dict(zip(features, np.zeros((len(features), 3))))
pretrain_path = dict(zip(features, ['', '', '']))
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
        pretrain_path[feature] = [save_path + '/(22K+lw)' + source_dataset_name + '_20180422.1036.41']
    elif feature == 'rCTA':
        filters[feature] = [32, 32, 32, 32, 32]
        kernels[feature] = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
        poolings[feature] = [(1, 1), (2, 2), (2, 5), (2, 7), (3, 2)]
        feature_sizes[feature] = [30, 142, 1]
        pretrain_path[feature] = [save_path + '/(rCTA)' + source_dataset_name + '_20180423.1037.52']
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
        pretrain_path[feature] = [save_path + '/(pitch+lw)' + source_dataset_name + '_20180425.1104.11']
para_line = []
para_line.append('action:' + str(action) + '\n')
para_line.append('action_description:' + str(action_description) + '\n')
para_line.append('features:' + str(features) + '\n')
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
para_line.append('lock_feature_extraction:' + str(lock_feature_extraction) + '\n')
para_line.append('classifier_units:' + str(classifier_units) + '\n')
for feature in features:
    para_line.append('feature:' + str(feature) + '\n')
    para_line.append('filters:' + str(filters[feature]) + '\n')
    para_line.append('kernels:' + str(kernels[feature]) + '\n')
    para_line.append('poolings:' + str(poolings[feature]) + '\n')
    para_line.append('feature_sizes:' + str(feature_sizes[feature]) + '\n')
    para_line.append('pretrain_path:' + str(pretrain_path[feature]) + '\n')
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
Train_Y = dict(zip(emotions, np.zeros((source_data_num, 1))))
Val_Y = dict(zip(emotions, np.zeros((target_data_num, 1))))
Train_Y['valence'] = np.load(save_path + '/' + source_dataset_name + '/Train_Y_valence.npy')
Train_Y['arousal'] = np.load(save_path + '/' + source_dataset_name + '/Train_Y_arousal.npy')
Val_Y['valence'] = np.load(save_path + '/' + target_dataset_name + '/Train_Y_valence.npy')
Val_Y['arousal'] = np.load(save_path + '/' + target_dataset_name + '/Train_Y_arousal.npy')
Train_X = dict(zip(features, np.zeros((source_data_num, 1, 1, 1))))
Val_X = dict(zip(features, np.zeros((target_data_num, 1, 1, 1))))
for feature in features:
    print(feature)
    Train_X[feature] = np.load(save_path + '/' + source_dataset_name +
                               '/Train_X@' + str(output_sample_rate) + 'Hz_' + feature + '.npy')
    Val_X[feature] = np.load(save_path + '/' + target_dataset_name +
                             '/Train_X@' + str(output_sample_rate) + 'Hz_' + feature + '.npy')
    print("Train_X shape:" + str(Train_X[feature].shape))
    print("Val_X shape:" + str(Val_X[feature].shape))


# Training
for emotion_axis in emotions:
    #  restart session
    if KTF._SESSION:
        print('Reset session.')
        KTF.clear_session()
        KTF.set_session(get_session())
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
    regressor = model_structure.regression_classifier(encoded_audio_tensor=extractor, units=classifier_units)
    model = Model(inputs=[feature_tensor0, feature_tensor1, feature_tensor2], outputs=regressor)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
    model.summary()
    # load mono feature trained-model as multi-feature pre-trained model
    if load_pretrained_weights:
        for i in range(0, 3):
            if os.path.exists(pretrain_path[features[i]][0] + '/' + emotion_axis + '/log_0_logs.json'):
                with open(pretrain_path[features[i]][0] + '/' + emotion_axis + '/log_0_logs.json', "r") as fb:
                    data = json.load(fb)
                    max_temp = max(data['train_R2_pearsonr'])
            for root, subdirs, files in os.walk(pretrain_path[features[i]][0] + '/' + emotion_axis):
                for f in files:
                    if os.path.splitext(f)[1] == '.h5' and 'train_R2pr_' + format(max_temp, '.5f') in f:
                        print(f)
                        pre_model = load_model(os.path.join(root, f), custom_objects={'Melspectrogram': Melspectrogram,
                                                                                      'R2': metric.R2,
                                                                                      'R2pr': metric.R2pr})
                        model_extract[i].set_weights(pre_model.get_weights()[:-2])
                        if lock_feature_extraction:
                            print('Lock feature extraction weights.')
                            make_trainable(model_extract[i], True)
    model_path = execute_name + '/' + emotion_axis + '/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model.save(model_path + '/init_run.h5')
    LossR2Logger_ModelCheckPoint = callback_wayne.LossR2Logger_ModelCheckPoint(
        train_data=([Train_X[features[0]], Train_X[features[1]], Train_X[features[2]]], Train_Y[emotion_axis]),
        val_data=([Val_X[features[0]], Val_X[features[1]], Val_X[features[2]]], Val_Y[emotion_axis]),
        file_name=model_path + '/log', run_num=0, filepath=model_path, monitor=monitor, verbose=0,
        save_best_only=save_best_only, save_weights_only=save_weights_only, mode=mode, period=1)
    model.fit(x=[Train_X[features[0]], Train_X[features[1]], Train_X[features[2]]],
              y=Train_Y[emotion_axis],
              batch_size=batch_size, epochs=epochs,
              callbacks=[LossR2Logger_ModelCheckPoint],
              verbose=1, shuffle=True,
              validation_data=([Val_X[features[0]], Val_X[features[1]], Val_X[features[2]]], Val_Y[emotion_axis]))
