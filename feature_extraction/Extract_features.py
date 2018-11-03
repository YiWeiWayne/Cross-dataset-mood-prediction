import numpy as np
import keras.backend.tensorflow_backend as KTF
import os
import tensorflow as tf
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from functions import model_structure, metric, Transfer_funcs
from keras.models import Model, load_model
from keras.layers import Input
from scipy.io import loadmat
from kapre.time_frequency import Melspectrogram
from scipy.stats import pearsonr
import json

# GPU speed limit
def get_session(gpu_fraction=0.6):
    # Assume that you have 6GB of GPU memory and want to allocate ~2GB
    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
KTF.set_session(get_session())

save_path = '/mnt/data/Wayne'
sec_length = 29
output_sample_rate = 22050

# Parameters
for dataset_name in ['CH_818']:
    # source_execute_name = save_path + '/AMG_1608_20180418.1807.00'
    print(dataset_name)
    if dataset_name == 'AMG_1608':
        data_num = 1608
    elif dataset_name == 'CH_818':
        data_num = 818

    # load Mel spectrum features
    print('load Mel spectrum features')
    Train_X = np.load(save_path + '/' + dataset_name + '/Train_X@' + str(output_sample_rate) + 'Hz.npy')
    Train_Y_valence = np.load(save_path + '/' + dataset_name + '/Train_Y_valence.npy')
    Train_Y_arousal = np.load(save_path + '/' + dataset_name + '/Train_Y_arousal.npy')

    input_tensor = Input(shape=(1, output_sample_rate*sec_length))
    feature_tensor = model_structure.extract_melspec(input_tensor=input_tensor, sr=output_sample_rate,
                                                     n_dft=1024, n_hop=512)
    feature_extractor = Model(inputs=input_tensor, outputs=feature_tensor)
    Train_X_feat = feature_extractor.predict(Train_X)
    print('Train_X_feat: ' + str(Train_X_feat.shape))
    np.save(save_path + '/' + dataset_name + '/Train_X@' + str(output_sample_rate) + 'Hz_melSpec_lw.npy', Train_X_feat)

    # # load rCTA
    # print('load rCTA')
    # if os.path.exists(save_path+'/'+dataset_name+'/Train_X@'+str(output_sample_rate)+'Hz_rCTA.mat'):
    #     Train_X = loadmat(save_path+'/'+dataset_name+'/Train_X@'+str(output_sample_rate)+'Hz_rCTA.mat')
    # Train_X = Train_X['Train_X_feature']
    # Train_X_feat = Train_X.reshape((Train_X.shape[0], Train_X.shape[1], Train_X.shape[2], 1))
    # print('Train_X_feat: ' + str(Train_X_feat.shape))
    # np.save(save_path + '/' + dataset_name + '/Train_X@' + str(output_sample_rate) + 'Hz_rCTA.npy', Train_X_feat)

    # # load rTA
    # print('load rTA')
    # if os.path.exists(save_path+'/'+dataset_name+'/Train_X@'+str(output_sample_rate)+'Hz_rTA.mat'):
    #     Train_X = loadmat(save_path+'/'+dataset_name+'/Train_X@'+str(output_sample_rate)+'Hz_rTA.mat')
    # Train_X = Train_X['Train_X_feature']
    # Train_X_feat = Train_X.reshape((Train_X.shape[0], Train_X.shape[1], Train_X.shape[2], 1))
    # print('Train_X_feat: ' + str(Train_X_feat.shape))
    # np.save(save_path + '/' + dataset_name + '/Train_X@' + str(output_sample_rate) + 'Hz_rTA.npy', Train_X_feat)

    # # load pitch
    # Train_X_feat = []
    # print('load pitch')
    # for i in range(1, data_num+1):
    #     if os.path.exists(save_path+'/'+dataset_name+'_pitch@'+str(output_sample_rate)+'Hz/'
    #                               +str(i).zfill(4)+'@'+str(output_sample_rate)+'_pitch_salience.npz'):
    #         print(str(i).zfill(4)+'@'+str(output_sample_rate)+'_pitch_salience.npz')
    #         file = np.load(save_path+'/'+dataset_name+'_pitch@'+str(output_sample_rate)+'Hz/'
    #                               +str(i).zfill(4)+'@'+str(output_sample_rate)+'_pitch_salience.npz')
    #         Train_X_feat.append(file['salience'])
    # Train_X_feat = np.asarray(Train_X_feat)
    # Train_X_feat = Train_X_feat.reshape((Train_X_feat.shape[0], Train_X_feat.shape[1], Train_X_feat.shape[2], 1))
    # print('Train_X_feat: ' + str(Train_X_feat.shape))
    # np.save(save_path + '/' + dataset_name + '/Train_X@' + str(output_sample_rate) + 'Hz_pitch.npy', Train_X_feat)

    # load pitch+lw
    Train_X_feat = []
    print('load pitch+lw')
    for i in range(1, data_num+1):
        if os.path.exists(save_path+'/'+dataset_name+'_pitch+lw@'+str(output_sample_rate)+'Hz/'
                                  +str(i).zfill(4)+'@'+str(output_sample_rate)+'_pitch_salience.npz'):
            print(str(i).zfill(4)+'@'+str(output_sample_rate)+'_pitch_salience.npz')
            file = np.load(save_path+'/'+dataset_name+'_pitch+lw@'+str(output_sample_rate)+'Hz/'
                                  +str(i).zfill(4)+'@'+str(output_sample_rate)+'_pitch_salience.npz')
            Train_X_feat.append(file['salience'])
    Train_X_feat = np.asarray(Train_X_feat)
    Train_X_feat = Train_X_feat.reshape((Train_X_feat.shape[0], Train_X_feat.shape[1], Train_X_feat.shape[2], 1))
    print('Train_X_feat: ' + str(Train_X_feat.shape))
    np.save(save_path + '/' + dataset_name + '/Train_X@' + str(output_sample_rate) + 'Hz_pitch+lw.npy', Train_X_feat)

    # load fourier-based rhythm
    print('load fourier-based rhythm')
    # data_types = ['fou', 'fou_lag', 'fou_cyc', 'auto', 'auto_lag', 'auto_cyc']
    data_types = ['auto']
    # mat_names = ['tempogram_fourier', 'tempogram_fourier_timeLag', 'cyclicTempogram_fourier',
    #              'tempogram_autocorrelation', 'tempogram_autocorrelation_timeLag', 'cyclicTempogram_autocorrelation']
    mat_names = ['tempogram_autocorrelation']
    mapping = dict(zip(data_types, mat_names))
    for data_type in data_types:
        Train_X_feat = []
        mat_name = mapping[data_type]
        for i in range(1, data_num + 1):
            print(str(i))
            if os.path.exists(save_path + '/' + dataset_name + '/tempogram/' + str(i).zfill(4) + '@' + data_type + '.mat'):
                print('yes')
                file = loadmat(save_path + '/' + dataset_name + '/tempogram/' + str(i).zfill(4) + '@' + data_type + '.mat')
                Train_X_feat.append(file[mat_name])
        Train_X_feat = np.asarray(Train_X_feat)
        print('Train_X_feat: ' + str(Train_X_feat.shape))
        Train_X_feat = Train_X_feat.reshape((Train_X_feat.shape[0], Train_X_feat.shape[1], Train_X_feat.shape[2], 1))
        print('Train_X_feat: ' + str(Train_X_feat.shape))
        np.save(save_path + '/' + dataset_name + '/Train_X@' + str(output_sample_rate) + 'Hz_' +data_type+'.npy', Train_X_feat)
