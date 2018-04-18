import os
from pyexcel_xls import get_data
import json
import librosa
from scipy.io import savemat
import numpy as np


def audio_to_wav(dataset_name, dataset_path, label_path, sec_length, output_sample_rate, save_path):
    wav_path = save_path + '/Dataset/' + dataset_name + '_wav@' + str(output_sample_rate) + 'Hz'
    # load xls data
    data = get_data(label_path)
    encodedjson = json.dumps(data)
    decodejson = json.loads(encodedjson)
    if dataset_name == 'AMG_1608':
        decodejson = decodejson['amg1608_v2']
    elif dataset_name == 'CH_818':
        decodejson = decodejson['Arousal']

    # transfer mp3 to wav file
    if not os.path.exists(wav_path):
        os.makedirs(wav_path)
    if True:
        for i in range(1, len(decodejson)):
            print(str(i).zfill(4))
            if dataset_name == 'AMG_1608':
                if os.path.exists(dataset_path + '/' + str(decodejson[i][2]) + '.mp3'):
                    print(dataset_path + '/' + str(decodejson[i][2]) + '.mp3')
                    y, sr = librosa.load(dataset_path + '/' + str(decodejson[i][2]) + '.mp3', sr=output_sample_rate)
                    print(y.shape)
                    print(str(sr))
            elif dataset_name == 'CH_818':
                for root, subdirs, files in os.walk(dataset_path):
                    for f in files:
                        if os.path.splitext(f)[1] == '.MP3' or os.path.splitext(f)[1] == '.mp3':
                            if f[0:4].startswith(str(i) + '='):
                                print(dataset_path + '/' + f)
                                y, sr = librosa.load(dataset_path + '/' + f, sr=output_sample_rate)
                                print(y.shape)
                                print(str(sr))
            if y.shape[0] >= output_sample_rate*sec_length:
                librosa.output.write_wav(path=wav_path + '/' + str(i).zfill(4) + '@' + str(output_sample_rate) + '.wav',
                                         y=y[0:int(output_sample_rate*sec_length)], sr=output_sample_rate)
            else:
                print('Shorter: ' + str(y.shape[0]) + '/' + str(output_sample_rate*sec_length))


def wav_to_npy(dataset_name, label_path, output_sample_rate, save_path):
    wav_path = save_path + '/Dataset/' + dataset_name + '_wav@' + str(output_sample_rate) + 'Hz'
    # load xls data
    data = get_data(label_path)
    encodedjson = json.dumps(data)
    decodejson = json.loads(encodedjson)

    # Generate Train X and Train Y
    Train_Y_valence = []
    Train_Y_arousal = []
    Train_X = []
    if dataset_name == 'AMG_1608':
        decodejson = decodejson['amg1608_v2']
        for i in range(1, len(decodejson)):
            print(str(i))
            Train_Y_valence.append(decodejson[i][7])
            Train_Y_arousal.append(decodejson[i][8])
            y, sr = librosa.load(wav_path + '/' + str(i).zfill(4) + '@' + str(output_sample_rate) + '.wav',
                                 sr=output_sample_rate)
            Train_X.append(y)
    elif dataset_name == 'CH_818':
        abs_max_label_value = 10
        Train_Y_arousal = decodejson['Arousal']
        Train_Y_arousal = Train_Y_arousal[1:len(Train_Y_arousal)]
        Train_Y_arousal = np.vstack(Train_Y_arousal)
        Train_Y_arousal = np.mean(Train_Y_arousal[:, 1:Train_Y_arousal.shape[1]], axis=1)
        Train_Y_arousal = Train_Y_arousal / abs_max_label_value
        Train_Y_valence = decodejson['Valence']
        Train_Y_valence = Train_Y_valence[1:len(Train_Y_valence)]
        Train_Y_valence = np.vstack(Train_Y_valence)
        Train_Y_valence = np.mean(Train_Y_valence[:, 1:Train_Y_valence.shape[1]], axis=1)
        Train_Y_valence = Train_Y_valence / abs_max_label_value
        for i in range(1, len(decodejson['Arousal'])):
            print(str(i))
            y, sr = librosa.load(wav_path + '/' + str(i).zfill(4) + '@' + str(output_sample_rate) + '.wav',
                                 sr=output_sample_rate)
            Train_X.append(y)

    Train_Y_valence = np.hstack(Train_Y_valence)
    Train_Y_arousal = np.hstack(Train_Y_arousal)
    Train_X = np.vstack(Train_X)
    Train_X = Train_X.reshape((Train_X.shape[0], 1, Train_X.shape[1]))
    print(Train_Y_valence.shape)
    print(Train_Y_arousal.shape)
    print(Train_X.shape)
    if not os.path.exists(save_path + '/' + dataset_name):
        os.makedirs(save_path + '/' + dataset_name)
    np.save(save_path + '/' + dataset_name + '/Train_X' + '@' + str(output_sample_rate) + 'Hz.npy', Train_X)
    np.save(save_path + '/' + dataset_name + '/Train_Y_valence.npy', Train_Y_valence)
    np.save(save_path + '/' + dataset_name + '/Train_Y_arousal.npy', Train_Y_arousal)


def npy_to_mat(dataset_name, output_sample_rate, save_path):
    Train_X = np.load(save_path + '/' + dataset_name + '/Train_X' + '@' + str(output_sample_rate) + 'Hz.npy')
    savemat(save_path + '/' + dataset_name + '/Train_X' + '@' + str(output_sample_rate) + 'Hz.mat',
            {"Train_X": [Train_X]})
