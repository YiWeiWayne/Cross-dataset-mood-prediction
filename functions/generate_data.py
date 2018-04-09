import os
from functions import Data_load


def additional_training_data(feature_extraction,
                             directory_add_ok_18, directory_add_ok_75,
                             directory_add_ng_18, directory_add_ng_75,
                             add_train_machine_set, add_train_machine,
                             window, Para, feat_size, train_data_set,
                             nfilt, lowfreq, preemph, sample_rate, nfft):
    # Generate file
    print('Generating file by '+feature_extraction+'...')
    if Para.train+Para.val != 0:
        if not os.path.exists(directory_add_ok_18):
           os.makedirs(directory_add_ok_18)
        if not os.path.exists(directory_add_ok_75):
           os.makedirs(directory_add_ok_75)
        if not os.path.exists(directory_add_ng_18):
           os.makedirs(directory_add_ng_18)
        if not os.path.exists(directory_add_ng_75):
           os.makedirs(directory_add_ng_75)
    # Generate
    for machine_index in add_train_machine_set:
        if machine_index in add_train_machine:
            print('Machine: ' + machine_index)
            # additional ok
            if Para.train+Para.val != 0:
                print('Generating additional ok data 18ppm by '+feature_extraction+'...')
                Data_load.load_data(window, Para.ok_s_18, feat_size, train_data_set, 'Training_' + machine_index,
                                    '18PPM_Everglades', 'OK', feature_extraction, nfilt, lowfreq, preemph, sample_rate,
                                    nfft,
                                    directory_add_ok_18)
                print('Generating additional ok data 75ppm by ' + feature_extraction + '...')
                Data_load.load_data(window, Para.ok_s_75, feat_size, train_data_set, 'Training_' + machine_index,
                                    '75PPM_Everglades', 'OK', feature_extraction, nfilt, lowfreq, preemph, sample_rate,
                                    nfft,
                                    directory_add_ok_75)
                print('Generating additional ng data 18ppm by ' + feature_extraction + '...')
                Data_load.load_data(window, Para.ng_s_18, feat_size, train_data_set,
                                    'Training_' + machine_index,
                                    '18PPM_Everglades', 'NG', feature_extraction, nfilt, lowfreq, preemph, sample_rate,
                                    nfft,
                                    directory_add_ng_18)
                print('Generating additional ng data 75ppm by ' + feature_extraction + '...')
                Data_load.load_data(window, Para.ng_s_75, feat_size, train_data_set,
                                    'Training_' + machine_index,
                                    '75PPM_Everglades', 'NG', feature_extraction, nfilt, lowfreq, preemph, sample_rate,
                                    nfft,
                                    directory_add_ng_75)
