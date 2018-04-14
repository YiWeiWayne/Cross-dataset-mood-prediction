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
from keras.models import Model
from keras.layers import Input
from keras.optimizers import SGD, Adam


# GPU speed limit
def get_session(gpu_fraction=0.3):
    # Assume that you have 6GB of GPU memory and want to allocate ~2GB
    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
KTF.set_session(get_session())

# Parameters
algorithm = 'ADDA'
source_dataset_name = 'AMG_1608'
target_dataset_name = 'CH_818'
save_path = '/mnt/data/Wayne'
source_execute_name = save_path + '/' + source_dataset_name + '_1522208082.13'
transfer_execute_name = save_path + '/' + \
                        algorithm + '_S_' + source_dataset_name + '_T_' + target_dataset_name + '_20180409.1219.50'
loss = 'mean_squared_error'
fold = 10
encoded_size = 32
emotions = ['valence', 'arousal']

print('Logging classifier model...')
# load regressor model
model = model_structure.compact_cnn(loss=loss)

# test CH818
target_dataset_name = 'CH_818'
print('Testing: ' + target_dataset_name)
CH818_R2_pearsonr_max = dict(zip(emotions, np.zeros((len(emotions), fold))))
R2_pearsonr_max = dict(zip(emotions, np.zeros((len(emotions), fold))))
Train_Y_valence = np.load(save_path + '/' + target_dataset_name + '/Train_Y_valence.npy')
Train_Y_arousal = np.load(save_path + '/' + target_dataset_name + '/Train_Y_arousal.npy')
Train_X = np.load(save_path + '/' + target_dataset_name + '/Train_X.npy')
print('Train_Y_valence: ' + str(Train_Y_valence.shape))
print('Train_Y_arousal: ' + str(Train_Y_arousal.shape))
print('Train_X: ' + str(Train_X.shape))
if True:
    for emotion in emotions:
        if emotion == 'valence':
            Y_true = Train_Y_valence
        elif emotion == 'arousal':
            Y_true = Train_Y_arousal
        for CV in range(0, fold):
            print(emotion + '/CV' + str(CV))
            if os.path.exists(source_execute_name + '/' + emotion + '/CV' + str(CV) + '/log_CV_' + str(CV).zfill(2) + '_logs.json'):
                with open(source_execute_name + '/' + emotion + '/CV' + str(CV) + '/log_CV_' + str(CV).zfill(2) + '_logs.json', "r") as fb:
                     data = json.load(fb)
                     R2_pearsonr_max[emotion][CV] = max(data['val_R2_pearsonr'])
            for root, subdirs, files in os.walk(source_execute_name + '/' + emotion + '/CV' + str(CV)):
                for f in files:
                    if os.path.splitext(f)[1] == '.h5' and 'R2pr_'+format(R2_pearsonr_max[emotion][CV], '.5f') in f:
                        print(f)
                        model.load_weights(os.path.join(root, f), by_name=True)
                        Y_predict = model.predict(Train_X)
                        Y_true = Y_true.reshape(-1, 1)
                        CH818_R2_pearsonr_max[emotion][CV] = np.square(pearsonr(Y_true, Y_predict)[0][0])
                        print(CH818_R2_pearsonr_max[emotion][CV])
print('CH818 maximum:')
print('R2_pearsonr_max: ' + str(CH818_R2_pearsonr_max))
print('Averaging Valence R2_pearsonr_max: ' + str(np.mean(CH818_R2_pearsonr_max['valence'])))
print('Averaging Arousal R2_pearsonr_max: ' + str(np.mean(CH818_R2_pearsonr_max['arousal'])))
data = OrderedDict()
data.update({"R2_pearsonr_max": [[emotions[0]], CH818_R2_pearsonr_max[emotions[0]],
                                 ['average', np.mean(CH818_R2_pearsonr_max[emotions[0]])],
                                 [emotions[1]], CH818_R2_pearsonr_max[emotions[1]],
                                 ['average', np.mean(CH818_R2_pearsonr_max[emotions[1]])]]})
save_data(source_execute_name + '/' + target_dataset_name + '_regressor.xls', data)

print('Logging domain transfer model...')
# Transfer learning output
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
_, target_extractor, target_audio_input, target_encoded_audio = model_structure.functional_compact_cnn(
    model_type="target_extractor")
target_tensor = Input(shape=(encoded_size, ))
classifier_model = Model(inputs=target_tensor,
                         outputs=model_structure.regression_classifier(target_tensor))
source_or_target_tensor = Input(shape=(encoded_size, ))
discriminator_model = Model(inputs=source_or_target_tensor,
                            outputs=model_structure.domain_classifier(source_or_target_tensor))
# Combine target model and classifier
target_classifier_model = Model(inputs=target_audio_input, outputs=classifier_model(target_encoded_audio))
target_classifier_model.compile(loss=loss, optimizer=adam, metrics=[metric.R2])
target_classifier_model.summary()
print('target-classifier-4:' + str(target_classifier_model.get_weights()[-4]))
print('target-classifier-2:' + str(target_classifier_model.get_weights()[-2]))

# Combine target model and discriminator
target_discriminator_model = Model(inputs=target_audio_input, outputs=discriminator_model(target_encoded_audio))
target_discriminator_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
print("target_discriminator_model summary:")
target_discriminator_model.summary()

print('Testing: (adapted)' + target_dataset_name)
CH818_R2_pearsonr_max = dict(zip(emotions, np.zeros((len(emotions), fold))))
source_R2_pearsonr_max = dict(zip(emotions, np.zeros((len(emotions), fold))))
target_R2_pearsonr_max = dict(zip(emotions, np.zeros((len(emotions), fold))))
Train_Y_valence = np.load(save_path + '/' + target_dataset_name + '/Train_Y_valence.npy')
Train_Y_arousal = np.load(save_path + '/' + target_dataset_name + '/Train_Y_arousal.npy')
Train_X = np.load(save_path + '/' + target_dataset_name + '/Train_X.npy')
print('Train_Y_valence: ' + str(Train_Y_valence.shape))
print('Train_Y_arousal: ' + str(Train_Y_arousal.shape))
print('Train_X: ' + str(Train_X.shape))
if True:
    for emotion in emotions:
        if emotion == 'valence':
            Y_true = Train_Y_valence
        elif emotion == 'arousal':
            Y_true = Train_Y_arousal
        for CV in range(0, 10):
            print(emotion + '/CV' + str(CV))
            print('Loading classifier model...')
            if os.path.exists(source_execute_name + '/' + emotion + '/CV' + str(CV) + '/log_CV_' + str(CV).zfill(
                    2) + '_logs.json'):
                with open(source_execute_name + '/' + emotion + '/CV' + str(CV) + '/log_CV_' + str(CV).zfill(2) + '_logs.json', "r") as fb:
                    data = json.load(fb)
                    source_R2_pearsonr_max[emotion][CV] = max(data['val_R2_pearsonr'])
                    print('source_R2_pearsonr_max: ' + str(source_R2_pearsonr_max[emotion][CV]))
            for root, subdirs, files in os.walk(source_execute_name + '/' + emotion + '/CV' + str(CV)):
                for f in files:
                    if os.path.splitext(f)[1] == '.h5' and 'R2pr_'+format(source_R2_pearsonr_max[emotion][CV], '.5f') in f:
                        print(f)
                        target_classifier_model.load_weights(os.path.join(root, f))
            print('Loading target feature extractor...')
            if os.path.exists(transfer_execute_name + '/' + emotion + '/CV' + str(CV) + '/log_CV' + str(CV) + '_logs.json'):
                with open(transfer_execute_name + '/' + emotion + '/CV' + str(CV) + '/log_CV' + str(CV) + '_logs.json', "r") as fb:
                    data = json.load(fb)
                    target_R2_pearsonr_max[emotion][CV] = max(data['val_R2_pearsonr'])
                    print('target_R2_pearsonr_max: ' + str(target_R2_pearsonr_max[emotion][CV]))
            for root, subdirs, files in os.walk(transfer_execute_name + '/' + emotion + '/CV' + str(CV)):
                for f in files:
                    if os.path.splitext(f)[1] == '.h5' and 'R2pr_' + format(target_R2_pearsonr_max[emotion][CV], '.5f') in f:
                        print(f)
                        target_discriminator_model.load_weights(os.path.join(root, f))
                        target_classifier_model.set_weights(target_discriminator_model.get_weights()[:-7])
                        Y_predict = target_classifier_model.predict(Train_X)
                        Y_true = Y_true.reshape(-1, 1)
                        CH818_R2_pearsonr_max[emotion][CV] = np.square(pearsonr(Y_true, Y_predict)[0][0])
                        print(target_dataset_name + ': ' + str(CH818_R2_pearsonr_max[emotion][CV]))
print('CH818 maximum:')
print('R2_pearsonr_max: ' + str(CH818_R2_pearsonr_max))
print('Averaging Valence R2_pearsonr_max: ' + str(np.mean(CH818_R2_pearsonr_max['valence'])))
print('Averaging Arousal R2_pearsonr_max: ' + str(np.mean(CH818_R2_pearsonr_max['arousal'])))
data = OrderedDict()
data.update({"R2_pearsonr_max": [[emotions[0]], CH818_R2_pearsonr_max[emotions[0]],
                                 ['average', np.mean(CH818_R2_pearsonr_max[emotions[0]])],
                                 [emotions[1]], CH818_R2_pearsonr_max[emotions[1]],
                                 ['average', np.mean(CH818_R2_pearsonr_max[emotions[1]])]]})
save_data(transfer_execute_name + '/' + target_dataset_name + '_transfer_regressor.xls', data)
