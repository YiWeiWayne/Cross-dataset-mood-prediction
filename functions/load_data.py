import os
import numpy as np
import time
from keras.utils import np_utils
import math


def training_data(feature_extraction,
                  directory_ok_18, directory_ok_75,
                  directory_w_18, directory_w_75,
                  directory_ng_18, directory_ng_75,
                  train_machine_set, train_machine, rumple_name, random_number,
                  N_para, W_para):
    print('Load training data...')
    X, Y, index, tmp_x, tmp_y, Extracted_index = [], [], [], [], [], []
    Total_num = 0
    train_machine_ok_num = 0
    train_machine_r_num = 0
    train_machine_ng_num = 0

    for machine_index in train_machine_set:
        if machine_index in train_machine:
            if os.path.exists(directory_ok_18 + '/' + 'Training_' + machine_index + '_' +
                             '18PPM_Everglades' + '_' + 'OK' + '_x.npy'):
                train_machine_ok_num += 1
            if os.path.exists(directory_w_18 + '/' + 'Training_' + machine_index + rumple_name + '_' +
                             '18PPM_Everglades' + '_' + 'OK' + '_x.npy'):
                train_machine_r_num += 1
            if os.path.exists(directory_ng_18 + '/' + 'Training_' + machine_index + '_' +
                             '18PPM_Everglades' + '_' + 'NG' + '_x.npy'):
                train_machine_ng_num += 1
    print('Machine numbers for OK data: '+str(train_machine_ok_num))
    print('Machine numbers for rumple data: '+str(train_machine_r_num))
    print('Machine numbers for NG data: '+str(train_machine_ng_num))
    for machine_index in train_machine_set:
        if machine_index in train_machine:
            print('Machine: ' + machine_index)
            if os.path.exists(directory_ok_18 + '/' + 'Training_' + machine_index + '_' + '18PPM_Everglades' + '_' + 'OK' + '_x.npy'):
                # OK_18ppm
                print('Loading ok data 18ppm by ' + feature_extraction + '...')
                tmp_x = np.load(directory_ok_18 + '/' + 'Training_' + machine_index + '_'
                                + '18PPM_Everglades' + '_' + 'OK' + '_x.npy')
                tmp_y = np.load(directory_ok_18 + '/' + 'Training_' + machine_index + '_'
                                + '18PPM_Everglades' + '_' + 'OK' + '_y.npy')
                if tmp_x.shape[0] < int(N_para.ok / (train_machine_ok_num * 2)):
                    print('(not enough)Extracted frame numbers:' + str(tmp_x.shape[0]))
                else:
                    print('Extracted frame numbers:' + str(int(N_para.ok / (train_machine_ok_num * 2))))
                index = np.arange(tmp_x.shape[0])
                for i in range(0, random_number):
                    np.random.shuffle(index)
                X.append(tmp_x[index[0:int(N_para.ok / (train_machine_ok_num * 2))], :])
                Y.append(tmp_y[index[0:int(N_para.ok / (train_machine_ok_num * 2))]])
                Extracted_index.append(Total_num+index[0:int(N_para.ok / (train_machine_ok_num * 2))])
                Total_num += tmp_x.shape[0]
                # OK_75ppm
                print('Loading ok data 75ppm by ' + feature_extraction + '...')
                tmp_x = np.load(directory_ok_75 + '/' + 'Training_' + machine_index + '_' + '75PPM_Everglades' + '_' + 'OK' + '_x.npy')
                tmp_y = np.load(directory_ok_75 + '/' + 'Training_' + machine_index + '_' + '75PPM_Everglades' + '_' + 'OK' + '_y.npy')
                if tmp_x.shape[0] < int(N_para.ok / (train_machine_ok_num * 2)):
                    print('(not enough)Extracted frame numbers:' + str(tmp_x.shape[0]))
                else:
                    print('Extracted frame numbers:' + str(int(N_para.ok / (train_machine_ok_num * 2))))
                index = np.arange(tmp_x.shape[0])
                for i in range(0, random_number):
                    np.random.shuffle(index)
                X.append(tmp_x[index[0:int(N_para.ok / (train_machine_ok_num * 2))], :])
                Y.append(tmp_y[index[0:int(N_para.ok / (train_machine_ok_num * 2))]])
                Extracted_index.append(Total_num + index[0:int(N_para.ok / (train_machine_ok_num * 2))])
                Total_num += tmp_x.shape[0]
            if os.path.exists(directory_w_18 + '/' + 'Training_' + machine_index + rumple_name + '_' + '18PPM_Everglades' + '_' + 'OK' + '_x.npy'):
                # W_18ppm
                print('Loading rumple data 18ppm by ' + feature_extraction + '...')
                tmp_x = np.load(directory_w_18 + '/' + 'Training_' + machine_index + rumple_name + '_' + '18PPM_Everglades' + '_' + 'OK' + '_x.npy')
                tmp_y = np.load(directory_w_18 + '/' + 'Training_' + machine_index + rumple_name + '_' + '18PPM_Everglades' + '_' + 'OK' + '_y.npy')
                if tmp_x.shape[0] < int(W_para.ok / (train_machine_r_num * 2)):
                    print('(not enough)Extracted frame numbers:' + str(tmp_x.shape[0]))
                else:
                    print('Extracted frame numbers:' + str(int(W_para.ok / (train_machine_r_num * 2))))
                index = np.arange(tmp_x.shape[0])
                for i in range(0, random_number):
                    np.random.shuffle(index)
                X.append(tmp_x[index[0:int(W_para.ok / (train_machine_r_num * 2))], :])
                Y.append(tmp_y[index[0:int(W_para.ok / (train_machine_r_num * 2))]])
                Extracted_index.append(Total_num + index[0:int(W_para.ok / (train_machine_r_num * 2))])
                Total_num += tmp_x.shape[0]
                # W_75ppm
                print('Loading rumple data 75ppm by ' + feature_extraction + '...')
                tmp_x = np.load(directory_w_75 + '/' + 'Training_' + machine_index + rumple_name + '_' + '75PPM_Everglades' + '_' + 'OK' + '_x.npy')
                tmp_y = np.load(directory_w_75 + '/' + 'Training_' + machine_index + rumple_name + '_' + '75PPM_Everglades' + '_' + 'OK' + '_y.npy')
                if tmp_x.shape[0] < int(W_para.ok / (train_machine_r_num * 2)):
                    print('(not enough)Extracted frame numbers:' + str(tmp_x.shape[0]))
                else:
                    print('Extracted frame numbers:' + str(int(W_para.ok / (train_machine_r_num * 2))))
                index = np.arange(tmp_x.shape[0])
                for i in range(0, random_number):
                    np.random.shuffle(index)
                X.append(tmp_x[index[0:int(W_para.ok / (train_machine_r_num * 2))], :])
                Y.append(tmp_y[index[0:int(W_para.ok / (train_machine_r_num * 2))]])
                Extracted_index.append(Total_num + index[0:int(W_para.ok / (train_machine_r_num * 2))])
                Total_num += tmp_x.shape[0]
            if os.path.exists(directory_ng_18 + '/' + 'Training_' + machine_index + '_' + '18PPM_Everglades' + '_' + 'NG' + '_x.npy'):
                # NG_18ppm
                print('Loading ng data 18ppm by ' + feature_extraction + '...')
                tmp_x = np.load(directory_ng_18 + '/' + 'Training_' + machine_index + '_' + '18PPM_Everglades' + '_' + 'NG' + '_x.npy')
                tmp_y = np.load(directory_ng_18 + '/' + 'Training_' + machine_index + '_' + '18PPM_Everglades' + '_' + 'NG' + '_y.npy')
                if tmp_x.shape[0] < int(N_para.ng / (train_machine_ng_num * 2)):
                    print('(not enough)Extracted frame numbers:' + str(tmp_x.shape[0]))
                else:
                    print('Extracted frame numbers:' + str(int(N_para.ng / (train_machine_ng_num * 2))))
                index = np.arange(tmp_x.shape[0])
                for i in range(0, random_number):
                    np.random.shuffle(index)
                X.append(tmp_x[index[0:int(N_para.ng / (train_machine_ng_num * 2))], :])
                Y.append(tmp_y[index[0:int(N_para.ng / (train_machine_ng_num * 2))]])
                Extracted_index.append(Total_num + index[0:int(N_para.ng / (train_machine_ng_num * 2))])
                Total_num += tmp_x.shape[0]
                # NG_75ppm
                print('Loading ng data 75ppm by ' + feature_extraction + '...')
                tmp_x = np.load(directory_ng_75 + '/' + 'Training_' + machine_index + '_' + '75PPM_Everglades' + '_' + 'NG' + '_x.npy')
                tmp_y = np.load(directory_ng_75 + '/' + 'Training_' + machine_index + '_' + '75PPM_Everglades' + '_' + 'NG' + '_y.npy')
                if tmp_x.shape[0] < int(N_para.ng / (train_machine_ng_num * 2)):
                    print('(not enough)Extracted frame numbers:' + str(tmp_x.shape[0]))
                else:
                    print('Extracted frame numbers:' + str(int(N_para.ng / (train_machine_ng_num * 2))))
                index = np.arange(tmp_x.shape[0])
                for i in range(0, random_number):
                    np.random.shuffle(index)
                X.append(tmp_x[index[0:int(N_para.ng / (train_machine_ng_num * 2))], :])
                Y.append(tmp_y[index[0:int(N_para.ng / (train_machine_ng_num * 2))]])
                Extracted_index.append(Total_num + index[0:int(N_para.ng / (train_machine_ng_num * 2))])
                Total_num += tmp_x.shape[0]
    del tmp_x, tmp_y
    return X, Y, Extracted_index, Total_num


def additional_training_data(X, Y, Extracted_index, Total_num, feature_extraction,
                             directory_add_ok_18, directory_add_ok_75,
                             directory_add_ng_18, directory_add_ng_75,
                             add_train_machine_set, add_train_machine, random_number,
                             A_para):
    print('Load additional training data...')
    tmp_x, tmp_y = [], []
    train_machine_ok_num = 0
    train_machine_ng_num = 0
    for machine_index in add_train_machine_set:
        if machine_index in add_train_machine:
            if os.path.exists(directory_add_ok_18 + '/' + 'Training_' + machine_index + '_' +
                              '18PPM_Everglades' + '_' + 'OK' + '_x.npy') or \
                    os.path.exists(directory_add_ok_75 + '/' + 'Training_' + machine_index + '_' +
                              '75PPM_Everglades' + '_' + 'OK' + '_x.npy'):
                train_machine_ok_num += 1
            if os.path.exists(directory_add_ng_18 + '/' + 'Training_' + machine_index + '_' +
                              '18PPM_Everglades' + '_' + 'NG' + '_x.npy') or \
                    os.path.exists(directory_add_ng_75 + '/' + 'Training_' + machine_index + '_' +
                              '75PPM_Everglades' + '_' + 'NG' + '_x.npy'):
                train_machine_ng_num += 1
    print('Machine numbers for OK data: '+str(train_machine_ok_num))
    print('Machine numbers for NG data: '+str(train_machine_ng_num))
    for machine_index in add_train_machine_set:
        if machine_index in add_train_machine:
            print('Load additional training data...')
            print('Machine: ' + machine_index)
            if os.path.exists(directory_add_ok_18 + '/' + 'Training_' + machine_index + '_' + '18PPM_Everglades' + '_' + 'OK' + '_x.npy') and\
                os.path.exists(directory_add_ok_75 + '/' + 'Training_' + machine_index + '_' + '75PPM_Everglades' + '_' + 'OK' + '_x.npy'):
                ok_ind = 2
            else:
                ok_ind = 1
            if os.path.exists(directory_add_ng_18 + '/' + 'Training_' + machine_index + '_' + '18PPM_Everglades' + '_' + 'NG' + '_x.npy') and\
                os.path.exists(directory_add_ng_75 + '/' + 'Training_' + machine_index + '_' + '75PPM_Everglades' + '_' + 'NG' + '_x.npy'):
                ng_ind = 2
            else:
                ng_ind = 1
            if os.path.exists(directory_add_ok_18 + '/' + 'Training_' + machine_index + '_' + '18PPM_Everglades' + '_' + 'OK' + '_x.npy'):
                # OK_18ppm
                print('Loading additional ok data 18ppm by ' + feature_extraction + '...')
                tmp_x = np.load(directory_add_ok_18 + '/' + 'Training_' + machine_index + '_' + '18PPM_Everglades' + '_' + 'OK' + '_x.npy')
                tmp_y = np.load(directory_add_ok_18 + '/' + 'Training_' + machine_index + '_' + '18PPM_Everglades' + '_' + 'OK' + '_y.npy')
                print('Total frame numbers: ' + str(tmp_x.shape[0]))
                if tmp_x.shape[0] < int(A_para.ok / (train_machine_ok_num * ok_ind)):
                    print('(not enough)Extracted frame numbers:' + str(tmp_x.shape[0]))
                else:
                    print('Extracted frame numbers:' + str(int(A_para.ok / (train_machine_ok_num * ok_ind))))
                index = np.arange(tmp_x.shape[0])
                for i in range(0, random_number):
                    np.random.shuffle(index)
                X.append(tmp_x[index[0:int(A_para.ok / (train_machine_ok_num * ok_ind))], :])
                Y.append(tmp_y[index[0:int(A_para.ok / (train_machine_ok_num * ok_ind))], :])
                Extracted_index.append(Total_num + index[0:int(A_para.ok / (train_machine_ok_num * ok_ind))])
                Total_num += tmp_x.shape[0]
            if os.path.exists(directory_add_ok_75 + '/' + 'Training_' + machine_index + '_' + '75PPM_Everglades' + '_' + 'OK' + '_x.npy'):
                # OK_75ppm
                print('Loading additional ok data 75ppm by ' + feature_extraction + '...')
                tmp_x = np.load(directory_add_ok_75 + '/' + 'Training_' + machine_index + '_' + '75PPM_Everglades' + '_' + 'OK' + '_x.npy')
                tmp_y = np.load(directory_add_ok_75 + '/' + 'Training_' + machine_index + '_' + '75PPM_Everglades' + '_' + 'OK' + '_y.npy')
                print('Total frame numbers: ' + str(tmp_x.shape[0]))
                if tmp_x.shape[0] < int(A_para.ok / (train_machine_ok_num * ok_ind)):
                    print('(not enough)Extracted frame numbers:' + str(tmp_x.shape[0]))
                else:
                    print('Extracted frame numbers:' + str(int(A_para.ok / (train_machine_ok_num * ok_ind))))
                index = np.arange(tmp_x.shape[0])
                for i in range(0, random_number):
                    np.random.shuffle(index)
                X.append(tmp_x[index[0:int(A_para.ok / (train_machine_ok_num * ok_ind))], :])
                Y.append(tmp_y[index[0:int(A_para.ok / (train_machine_ok_num * ok_ind))], :])
                Extracted_index.append(Total_num + index[0:int(A_para.ok / (train_machine_ok_num * ok_ind))])
                Total_num += tmp_x.shape[0]
            if os.path.exists(directory_add_ng_18 + '/' + 'Training_' + machine_index + '_' + '18PPM_Everglades' + '_' + 'NG' + '_x.npy'):
                # NG_18ppm
                print('Loading additional ng data 18ppm by ' + feature_extraction + '...')
                tmp_x = np.load(directory_add_ng_18 + '/' + 'Training_' + machine_index + '_' + '18PPM_Everglades' + '_' + 'NG' + '_x.npy')
                tmp_y = np.load(directory_add_ng_18 + '/' + 'Training_' + machine_index + '_' + '18PPM_Everglades' + '_' + 'NG' + '_y.npy')
                print('Total frame numbers: ' + str(tmp_x.shape[0]))
                if tmp_x.shape[0] < int(A_para.ng / (train_machine_ng_num * ng_ind)):
                    print('(not enough)Extracted frame numbers:' + str(tmp_x.shape[0]))
                else:
                    print('Extracted frame numbers:' + str(int(A_para.ng / (train_machine_ng_num * ng_ind))))
                index = np.arange(tmp_x.shape[0])
                for i in range(0, random_number):
                    np.random.shuffle(index)
                X.append(tmp_x[index[0:int(A_para.ng / (train_machine_ng_num * ng_ind))], :])
                Y.append(tmp_y[index[0:int(A_para.ng / (train_machine_ng_num * ng_ind))], :])
                Extracted_index.append(Total_num + index[0:int(A_para.ng / (train_machine_ng_num * ng_ind))])
                Total_num += tmp_x.shape[0]
            if os.path.exists(directory_add_ng_75 + '/' + 'Training_' + machine_index + '_' + '75PPM_Everglades' + '_' + 'NG' + '_x.npy'):
                # NG_75ppm
                print('Loading additional ng data 75ppm by ' + feature_extraction + '...')
                tmp_x = np.load(directory_add_ng_75 + '/' + 'Training_' + machine_index + '_' + '75PPM_Everglades' + '_' + 'NG' + '_x.npy')
                tmp_y = np.load(directory_add_ng_75 + '/' + 'Training_' + machine_index + '_' + '75PPM_Everglades' + '_' + 'NG' + '_y.npy')
                print('Total frame numbers: ' + str(tmp_x.shape[0]))
                if tmp_x.shape[0] < int(A_para.ng / (train_machine_ng_num*ng_ind)):
                    print('(not enough)Extracted frame numbers:' + str(tmp_x.shape[0]))
                else:
                    print('Extracted frame numbers:' + str(int(A_para.ng / (train_machine_ng_num*ng_ind))))
                index = np.arange(tmp_x.shape[0])
                for i in range(0, random_number):
                    np.random.shuffle(index)
                X.append(tmp_x[index[0:int(A_para.ng / (train_machine_ng_num*ng_ind))], :])
                Y.append(tmp_y[index[0:int(A_para.ng / (train_machine_ng_num*ng_ind))], :])
                Extracted_index.append(Total_num + index[0:int(A_para.ng / (train_machine_ng_num * ng_ind))])
                Total_num += tmp_x.shape[0]
    del tmp_x, tmp_y
    return X, Y, Extracted_index, Total_num


def data(OK_X, OK_Y, OK_index, NG_X, NG_Y, NG_index, Total_num, feature_extraction, directory_ok_18, directory_ok_75,
        directory_ng_18, directory_ng_75,
        add_train_machine_set, add_train_machine, random_number,
        Para):
    print('Load data...')
    tmp_x, tmp_y = [], []
    OK_info, NG_info = [], []
    train_machine_ok_num = 0
    train_machine_ng_num = 0
    for machine_index in add_train_machine_set:
        if machine_index in add_train_machine:
            if os.path.exists(directory_ok_18 + '/' + 'Training_' + machine_index + '_' +
                                      '18PPM_Everglades' + '_' + 'OK' + '_x.npy') or \
                    os.path.exists(directory_ok_75 + '/' + 'Training_' + machine_index + '_' +
                                           '75PPM_Everglades' + '_' + 'OK' + '_x.npy'):
                train_machine_ok_num += 1
            if os.path.exists(directory_ng_18 + '/' + 'Training_' + machine_index + '_' +
                                      '18PPM_Everglades' + '_' + 'NG' + '_x.npy') or \
                    os.path.exists(directory_ng_75 + '/' + 'Training_' + machine_index + '_' +
                                           '75PPM_Everglades' + '_' + 'NG' + '_x.npy'):
                train_machine_ng_num += 1
    print('Machine numbers for OK data: ' + str(train_machine_ok_num))
    print('Machine numbers for NG data: ' + str(train_machine_ng_num))
    for machine_index in add_train_machine_set:
        if machine_index in add_train_machine:
            print('Load additional training data...')
            print('Machine: ' + machine_index)
            # Find if there are 18PPM & 75PPM data
            ok_ind = 0
            if os.path.exists(directory_ok_18 + '/' + 'Training_' + machine_index + '_' + '18PPM_Everglades' + '_' + 'OK' + '_x.npy'):
                print(' Get 18PPM OK!')
                ok_ind += 1
            if os.path.exists(directory_ok_75 + '/' + 'Training_' + machine_index + '_' + '75PPM_Everglades' + '_' + 'OK' + '_x.npy'):
                print(' Get 75PPM OK!')
                ok_ind += 1
            ng_ind = 0
            if os.path.exists(directory_ng_18 + '/' + 'Training_' + machine_index + '_' + '18PPM_Everglades' + '_' + 'NG' + '_x.npy'):
                print(' Get 18PPM NG!')
                ng_ind += 1
            if os.path.exists(directory_ng_75 + '/' + 'Training_' + machine_index + '_' + '75PPM_Everglades' + '_' + 'NG' + '_x.npy'):
                print(' Get 75PPM NG!')
                ng_ind += 1

            # load 18PPM OK data
            if os.path.exists(directory_ok_18 + '/' + 'Training_' + machine_index + '_' + '18PPM_Everglades' + '_' + 'OK' + '_x.npy'):
                # OK_18ppm
                print('Loading additional ok data 18ppm by ' + feature_extraction + '...')
                tmp_x = np.load(
                    directory_ok_18 + '/' + 'Training_' + machine_index + '_' + '18PPM_Everglades' + '_' + 'OK' + '_x.npy')
                tmp_y = np.load(
                    directory_ok_18 + '/' + 'Training_' + machine_index + '_' + '18PPM_Everglades' + '_' + 'OK' + '_y.npy')
                print('Total frame numbers: ' + str(tmp_x.shape[0]))
                if tmp_x.shape[0] < int((Para.train+Para.val)/(2 * train_machine_ok_num * ok_ind)):
                    print('(not enough)Extracted frame numbers:' + str(tmp_x.shape[0]))
                else:
                    print('Extracted frame numbers:' + str(int((Para.train+Para.val)/(2 * train_machine_ok_num * ok_ind))))
                index = np.arange(tmp_x.shape[0])
                for i in range(0, random_number):
                    np.random.shuffle(index)
                OK_X.append(tmp_x[index[0:int((Para.train+Para.val)/(2 * train_machine_ok_num * ok_ind))], :])
                OK_Y.append(tmp_y[index[0:int((Para.train+Para.val)/(2 * train_machine_ok_num * ok_ind))], :])
                OK_index.append(Total_num + index[0:int((Para.train+Para.val)/(2 * train_machine_ok_num * ok_ind))])
                if 'r' in machine_index:
                    for i in range(0, int((Para.train+Para.val)/(2 * train_machine_ok_num * ok_ind))):
                        OK_info.append(('r', '18'))
                else:
                    for i in range(0, int((Para.train+Para.val)/(2 * train_machine_ok_num * ok_ind))):
                        OK_info.append(('OK', '18'))
                Total_num += tmp_x.shape[0]

            # load 75PPM OK data
            if os.path.exists(directory_ok_75 + '/' + 'Training_' + machine_index + '_' + '75PPM_Everglades' + '_' + 'OK' + '_x.npy'):
                # OK_75ppm
                print('Loading additional ok data 75ppm by ' + feature_extraction + '...')
                tmp_x = np.load(directory_ok_75 + '/' + 'Training_' + machine_index + '_' + '75PPM_Everglades' + '_' + 'OK' + '_x.npy')
                tmp_y = np.load(directory_ok_75 + '/' + 'Training_' + machine_index + '_' + '75PPM_Everglades' + '_' + 'OK' + '_y.npy')
                print('Total frame numbers: ' + str(tmp_x.shape[0]))
                if tmp_x.shape[0] < int((Para.train + Para.val) / (2 * train_machine_ok_num * ok_ind)):
                    print('(not enough)Extracted frame numbers:' + str(tmp_x.shape[0]))
                else:
                    print('Extracted frame numbers:' + str(
                          int((Para.train + Para.val) / (2 * train_machine_ok_num * ok_ind))))
                index = np.arange(tmp_x.shape[0])
                for i in range(0, random_number):
                    np.random.shuffle(index)
                OK_X.append(tmp_x[index[0:int((Para.train + Para.val) / (2 * train_machine_ok_num * ok_ind))], :])
                OK_Y.append(tmp_y[index[0:int((Para.train + Para.val) / (2 * train_machine_ok_num * ok_ind))], :])
                OK_index.append(Total_num + index[0:int((Para.train + Para.val) / (2 * train_machine_ok_num * ok_ind))])
                if 'r' in machine_index:
                    for i in range(0, int((Para.train+Para.val)/(2 * train_machine_ok_num * ok_ind))):
                        OK_info.append(('r', '75'))
                else:
                    for i in range(0, int((Para.train+Para.val)/(2 * train_machine_ok_num * ok_ind))):
                        OK_info.append(('OK', '75'))
                Total_num += tmp_x.shape[0]

            # load 18PPM NG data
            if os.path.exists(directory_ng_18 + '/' + 'Training_' + machine_index + '_' + '18PPM_Everglades' + '_' + 'NG' + '_x.npy'):
                # NG_18ppm
                print('Loading additional ng data 18ppm by ' + feature_extraction + '...')
                tmp_x = np.load(directory_ng_18 + '/' + 'Training_' + machine_index + '_' + '18PPM_Everglades' + '_' + 'NG' + '_x.npy')
                tmp_y = np.load(directory_ng_18 + '/' + 'Training_' + machine_index + '_' + '18PPM_Everglades' + '_' + 'NG' + '_y.npy')
                print('Total frame numbers: ' + str(tmp_x.shape[0]))
                if tmp_x.shape[0] < int((Para.train + Para.val) / (2 * train_machine_ng_num * ng_ind)):
                    print('(not enough)Extracted frame numbers:' + str(tmp_x.shape[0]))
                else:
                    print('Extracted frame numbers:' + str(
                          int((Para.train + Para.val) / (2 * train_machine_ng_num * ng_ind))))
                index = np.arange(tmp_x.shape[0])
                for i in range(0, random_number):
                    np.random.shuffle(index)
                NG_X.append(tmp_x[index[0:int((Para.train + Para.val) / (2 * train_machine_ng_num * ng_ind))], :])
                NG_Y.append(tmp_y[index[0:int((Para.train + Para.val) / (2 * train_machine_ng_num * ng_ind))], :])
                NG_index.append(Total_num + index[0:int((Para.train + Para.val) / (2 * train_machine_ng_num * ng_ind))])
                for i in range(0, int((Para.train+Para.val)/(2 * train_machine_ng_num * ng_ind))):
                    NG_info.append(('NG', '18'))
                Total_num += tmp_x.shape[0]

            # load 75PPM NG data
            if os.path.exists(directory_ng_75 + '/' + 'Training_' + machine_index + '_' + '75PPM_Everglades' + '_' + 'NG' + '_x.npy'):
                # OK_75ppm
                print('Loading additional ng data 75ppm by ' + feature_extraction + '...')
                tmp_x = np.load(directory_ng_75 + '/' + 'Training_' + machine_index + '_' + '75PPM_Everglades' + '_' + 'NG' + '_x.npy')
                tmp_y = np.load(directory_ng_75 + '/' + 'Training_' + machine_index + '_' + '75PPM_Everglades' + '_' + 'NG' + '_y.npy')
                print('Total frame numbers: ' + str(tmp_x.shape[0]))
                if tmp_x.shape[0] < int((Para.train + Para.val) / (2 * train_machine_ng_num * ng_ind)):
                    print('(not enough)Extracted frame numbers:' + str(tmp_x.shape[0]))
                else:
                    print('Extracted frame numbers:' + str(
                          int((Para.train + Para.val) / (2 * train_machine_ng_num * ng_ind))))
                index = np.arange(tmp_x.shape[0])
                for i in range(0, random_number):
                    np.random.shuffle(index)
                NG_X.append(tmp_x[index[0:int((Para.train + Para.val) / (2 * train_machine_ng_num * ng_ind))], :])
                NG_Y.append(tmp_y[index[0:int((Para.train + Para.val) / (2 * train_machine_ng_num * ng_ind))], :])
                NG_index.append(Total_num + index[0:int((Para.train + Para.val) / (2 * train_machine_ng_num * ng_ind))])
                for i in range(0, int((Para.train+Para.val)/(2 * train_machine_ng_num * ng_ind))):
                    NG_info.append(('NG', '75'))
                Total_num += tmp_x.shape[0]
    OK_X = np.row_stack(OK_X)
    OK_Y = np.row_stack(OK_Y)
    OK_index = np.hstack(OK_index)
    OK_info = np.row_stack(OK_info)
    NG_X = np.row_stack(NG_X)
    NG_Y = np.row_stack(NG_Y)
    NG_index = np.hstack(NG_index)
    NG_info = np.row_stack(NG_info)
    return OK_X, OK_Y, OK_index, OK_info, NG_X, NG_Y, NG_index, NG_info, Total_num


def frame_data(OK_X, OK_Y, OK_index, NG_X, NG_Y, NG_index, Total_num, feature_extraction,
               add_train_machine_set, add_train_machine, random_number,
               Para, window,feat_size,sound_path,data_set, nfilt, lowfreq, sample_rate, nfft):
    frame_size = math.floor(sample_rate * window * 0.001)
    ceplifter = feat_size
    print('Load data...')
    train_machine_ok_num = 0
    train_machine_ng_num = 0
    OK_data_index, NG_data_index = [], []
    for machine_index in add_train_machine_set:
        if machine_index in add_train_machine:
            label = 'OK'
            if os.path.exists(
                    os.path.join(sound_path, data_set, 'Training_' + machine_index + '_pree', '18PPM_Everglades',
                                 label)) or \
                    os.path.exists(
                        os.path.join(sound_path, data_set, 'Training_' + machine_index + '_pree', '75PPM_Everglades',
                                     label)):
                train_machine_ok_num += 1
            label = 'NG'
            if os.path.exists(
                    os.path.join(sound_path, data_set, 'Training_' + machine_index + '_pree', '18PPM_Everglades',
                                 label)) or \
                    os.path.exists(
                        os.path.join(sound_path, data_set, 'Training_' + machine_index + '_pree', '75PPM_Everglades',
                                     label)):
                train_machine_ng_num += 1
    print('Machine numbers for OK data: ' + str(train_machine_ok_num))
    print('Machine numbers for NG data: ' + str(train_machine_ng_num))
    for machine_index in add_train_machine_set:
        if machine_index in add_train_machine:
            print('Load additional training data...')
            print('Machine: ' + machine_index)
            # Find if there are 18PPM & 75PPM data
            label = 'OK'
            if os.path.exists(
                    os.path.join(sound_path, data_set, 'Training_' + machine_index + '_pree', '18PPM_Everglades',
                                 label)) and \
                    os.path.exists(
                        os.path.join(sound_path, data_set, 'Training_' + machine_index + '_pree', '75PPM_Everglades',
                                     label)):
                ok_ind = 2
            else:
                ok_ind = 1
            label = 'NG'
            if os.path.exists(
                    os.path.join(sound_path, data_set, 'Training_' + machine_index + '_pree', '18PPM_Everglades',
                                 label)) or \
                    os.path.exists(
                        os.path.join(sound_path, data_set, 'Training_' + machine_index + '_pree', '75PPM_Everglades',
                                     label)):
                ng_ind = 2
            else:
                ng_ind = 1
            for label in ['OK', 'NG']:
                for speed in ['18PPM_Everglades', '75PPM_Everglades']:
                    if os.path.exists(
                            os.path.join(sound_path, data_set, 'Training_' + machine_index + '_pree', speed,
                                         label)):
                        print('Loading '+label+'  data ' + speed + ' by ' + feature_extraction + '...')
                        data_index = np.load(
                            os.path.join(sound_path, data_set, 'Training_' + machine_index + '_pree', speed,
                                         'index.npy'))
                        file_name = np.load(
                            os.path.join(sound_path, data_set, 'Training_' + machine_index + '_pree', speed,
                                         'file_name.npy'))
                        print('Total frame numbers: ' + str(len(data_index)))
                        index = np.arange(len(data_index))
                        for i in range(0, random_number):
                            np.random.shuffle(index)
                        if label is 'OK':
                            for i in index[0:int((Para.train + Para.val) / (2 * train_machine_ok_num * ok_ind))]:
                                path = os.path.join(sound_path, data_set, 'Training_' + machine_index + '_pree', speed,
                                                    label,
                                                    file_name[data_index[i][0] - 1] + '.npy')
                                OK_X.append(frame_mfcc.extract(path=path, start=data_index[i][1], end=data_index[i][2],
                                                               frame_size=frame_size, winfunc=lambda x: np.hamming(x),
                                                               nfft=nfft, sample_rate=sample_rate, lowfreq=lowfreq,
                                                               feat_size=feat_size, ceplifter=ceplifter, nfilt=nfilt))
                                OK_Y.append(0)
                                OK_data_index.append([data_index[i,:]])
                                print(machine_index + '_' + speed + '_' + label + '_' + str(len(OK_X)))
                            OK_index.append(
                                Total_num + np.arange(0, int((Para.train + Para.val) / (2 * train_machine_ok_num * ok_ind))))
                            Total_num += int((Para.train + Para.val) / (2 * train_machine_ok_num * ok_ind))
                        else:
                            for i in index[0:int((Para.train + Para.val) / (2 * train_machine_ng_num * ng_ind))]:
                                path = os.path.join(sound_path, data_set, 'Training_' + machine_index + '_pree', speed,
                                                    label,
                                                    file_name[data_index[i][0] - 1] + '.npy')
                                NG_X.append(frame_mfcc.extract(path=path, start=data_index[i][1], end=data_index[i][2],
                                                               frame_size=frame_size, winfunc=lambda x: np.hamming(x),
                                                               nfft=nfft, sample_rate=sample_rate, lowfreq=lowfreq,
                                                               feat_size=feat_size, ceplifter=ceplifter, nfilt=nfilt))
                                NG_Y.append(1)
                                NG_data_index.append(data_index[i])
                                print(machine_index+'_'+speed+'_'+label+'_'+str(len(NG_X)))
                            NG_index.append(Total_num + np.arange(0, int((Para.train + Para.val) /
                                                                         (2 * train_machine_ng_num * ng_ind))))
                            Total_num += int((Para.train + Para.val) / (2 * train_machine_ng_num * ng_ind))
    np.save('OK_X',OK_X)
    np.save('OK_Y', OK_Y)
    np.save('OK_data_index', OK_data_index)
    np.save('OK_index', OK_index)
    np.save('NG_X', NG_X)
    np.save('NG_Y', NG_Y)
    np.save('NG_data_index',NG_data_index)
    np.save('NG_index', NG_index)
    OK_X = np.asarray(OK_X)
    OK_Y = np.asarray(OK_Y)
    OK_data_index = np.asarray(OK_data_index)
    OK_index = np.hstack(OK_index)
    NG_X = np.asarray(NG_X)
    NG_Y = np.asarray(NG_Y)
    NG_data_index = np.asarray(NG_data_index)
    NG_index = np.hstack(NG_index)
    return OK_X, OK_Y, OK_data_index, OK_index, NG_X, NG_Y, NG_data_index, NG_index


def data_shuffle(X, Y, Extracted_index, Extracted_info, random_number):
    X = np.row_stack(X)
    Y = np.row_stack(Y)
    Extracted_index = np.hstack(Extracted_index)
    Extracted_info = np.row_stack(Extracted_info)
    print('X shape:' + str(X.shape))
    print('Y shape :' + str(Y.shape))
    index = np.arange(X.shape[0])
    np.random.seed(np.long(time.time()))
    for i in range(0, random_number):
        np.random.shuffle(index)
    X = X[index, :]
    Y = Y[index]
    Extracted_index = Extracted_index[index]
    Extracted_info = Extracted_info[index, :]
    return X, Y, Extracted_index, Extracted_info


def separate_train_val_data(X, Y, Extracted_index, random_number, validation_ratio):
    index = np.arange(X.shape[0])
    np.random.seed(np.long(time.time()))
    for i in range(0,random_number):
        np.random.shuffle(index)
    Val_X = X[index[0:math.floor(X.shape[0]/validation_ratio)]]
    Train_X = X[index[math.floor(X.shape[0]/validation_ratio):X.shape[0]]]
    Val_Y = Y[index[0:math.floor(Y.shape[0]/validation_ratio)]]
    Train_Y = Y[index[math.floor(Y.shape[0]/validation_ratio):Y.shape[0]]]
    Val_index = Extracted_index[index[0:math.floor(Extracted_index.shape[0]/validation_ratio)]]
    Train_index = Extracted_index[index[math.floor(Extracted_index.shape[0] / validation_ratio):Extracted_index.shape[0]]]
    return Val_X, Val_Y, Val_index, Train_X, Train_Y, Train_index


def extract_train_val_data_from_OK_NG(OK_X, OK_Y, OK_index, OK_info, NG_X, NG_Y, NG_index, NG_info, random_number, Para):
    Train_X, Train_Y, Train_index, Train_info, Val_X, Val_Y, Val_index, Val_info = [], [], [], [], [], [], [], []
    # load OK
    index = np.arange(OK_X.shape[0])
    for i in range(0, random_number):
        np.random.shuffle(index)
    # index_OK = index
    Train_X.append(OK_X[index[0:int(Para.train / 2)], :])
    Train_Y.append(OK_Y[index[0:int(Para.train / 2)], :])
    Train_index.append(OK_index[index[0:int(Para.train / 2)]])
    Train_info.append(OK_info[index[0:int(Para.train / 2)], :])
    Val_X.append(OK_X[index[int(Para.train / 2):int((Para.train + Para.val) / 2)], :])
    Val_Y.append(OK_Y[index[int(Para.train / 2):int((Para.train + Para.val) / 2)], :])
    Val_index.append(OK_index[index[int(Para.train / 2):int((Para.train + Para.val) / 2)]])
    Val_info.append(OK_info[index[int(Para.train / 2):int((Para.train + Para.val) / 2)], :])

    # load NG
    index = np.arange(NG_X.shape[0])
    for i in range(0, random_number):
        np.random.shuffle(index)
    # index_NG = index
    Train_X.append(NG_X[index[0:int(Para.train / 2)], :])
    Train_Y.append(NG_Y[index[0:int(Para.train / 2)], :])
    Train_index.append(NG_index[index[0:int(Para.train / 2)]])
    Train_info.append(NG_info[index[0:int(Para.train / 2)], :])
    Val_X.append(NG_X[index[int(Para.train / 2):int((Para.train + Para.val) / 2)], :])
    Val_Y.append(NG_Y[index[int(Para.train / 2):int((Para.train + Para.val) / 2)], :])
    Val_index.append(NG_index[index[int(Para.train / 2):int((Para.train + Para.val) / 2)]])
    Val_info.append(NG_info[index[int(Para.train / 2):int((Para.train + Para.val) / 2)], :])
    return Train_X, Train_Y, Train_index, Train_info, Val_X, Val_Y, Val_index, Val_info


# def extract_train_val_data_from_OK_NG_v1(OK_X, OK_Y, OK_index, NG_X, NG_Y, NG_index, random_number, Para):
#     Train_X, Train_Y, Train_index, Val_X, Val_Y, Val_index = [], [], [], [], [], []
#     # load OK
#     index = np.arange(OK_X.shape[0])
#     for i in range(0, random_number):
#         np.random.shuffle(index)
#     Train_X.append(OK_X[index[0:int(Para.train / 2)], :])
#     Train_Y.append(OK_Y[index[0:int(Para.train / 2)]])
#     Train_index.append(OK_index[index[0:int(Para.train / 2)], :])
#     Val_X.append(OK_X[index[int(Para.train / 2):int((Para.train + Para.val) / 2)], :])
#     Val_Y.append(OK_Y[index[int(Para.train / 2):int((Para.train + Para.val) / 2)]])
#     Val_index.append(OK_index[index[int(Para.train / 2):int((Para.train + Para.val) / 2)], :])
#
#     # load NG
#     index = np.arange(NG_X.shape[0])
#     for i in range(0, random_number):
#         np.random.shuffle(index)
#     Train_X.append(NG_X[index[0:int(Para.train / 2)], :])
#     Train_Y.append(NG_Y[index[0:int(Para.train / 2)]])
#     Train_index.append(NG_index[index[0:int(Para.train / 2)], :])
#     Val_X.append(NG_X[index[int(Para.train / 2):int((Para.train + Para.val) / 2)], :])
#     Val_Y.append(NG_Y[index[int(Para.train / 2):int((Para.train + Para.val) / 2)]])
#     Val_index.append(NG_index[index[int(Para.train / 2):int((Para.train + Para.val) / 2)], :])
#     return Train_X, Train_Y, Train_index, Val_X, Val_Y, Val_index
