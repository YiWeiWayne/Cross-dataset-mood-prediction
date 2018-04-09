import numpy as np
import keras.backend.tensorflow_backend as KTF
import os
import tensorflow as tf
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from functions import model_structure
import json
import csv
from pyexcel_xls import save_data
from collections import OrderedDict


# GPU speed limit
def get_session(gpu_fraction=0.3):
    # Assume that you have 6GB of GPU memory and want to allocate ~2GB
    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
KTF.set_session(get_session())

# Parameters
dataset_name = 'AMG_1608'
save_path = '/data/Wayne'
execute_name = save_path + '/' + 'AMG_1608_1522208082.13'
model_name = 'R2_0.3440237831_Loss_0.0656005718.h5'
loss = 'mean_squared_error'
fold = 10

# load log
emotions = ['valence', 'arousal']
R2_pearsonr_max = dict(zip(emotions, np.zeros((len(emotions), fold))))
R2_max = dict(zip(emotions, np.zeros((len(emotions), fold))))
for emotion in emotions:
    for CV in range(0, fold):
        if os.path.exists(execute_name + '/' + emotion + '/CV' + str(CV) + '/log_CV_' + str(CV).zfill(2) + '_logs.json'):
            with open(execute_name + '/' + emotion + '/CV' + str(CV) + '/log_CV_' + str(CV).zfill(2) + '_logs.json', "r") as fb:
                data = json.load(fb)
            R2_pearsonr_max[emotion][CV] = max(data['val_R2_pearsonr'])
            R2_max[emotion][CV] = max(data['val_R2'])
print('Log maximum:')
print('R2_pearsonr_max: ' + str(R2_pearsonr_max))
print('R2_max: ' + str(R2_max))
print('Averaging Valence R2_pearsonr_max: ' + str(np.mean(R2_pearsonr_max['valence'])) +
      ',R2_max: ' + str(np.mean(R2_max['valence'])))
print('Averaging Arousal R2_pearsonr_max: ' + str(np.mean(R2_pearsonr_max['arousal'])) +
      ',R2_max: ' + str(np.mean(R2_max['arousal'])))

# load model
model = model_structure.compact_cnn(loss=loss)

# test AMG1608
test_dataset_name = 'AMG_1608'
print('Testing: ' + test_dataset_name)
AMG1608_R2_pearsonr_max = dict(zip(emotions, np.zeros((len(emotions), fold))))
AMG1608_R2_max = dict(zip(emotions, np.zeros((len(emotions), fold))))
Train_Y_valence = np.load(save_path + '/' + test_dataset_name + '/Train_Y_valence.npy')
Train_Y_arousal = np.load(save_path + '/' + test_dataset_name + '/Train_Y_arousal.npy')
Train_X = np.load(save_path + '/' + test_dataset_name + '/Train_X.npy')
print('Train_Y_valence: ' + str(Train_Y_valence.shape))
print('Train_Y_arousal: ' + str(Train_Y_arousal.shape))
print('Train_X: ' + str(Train_X.shape))
if True:
    for emotion in emotions:
        for CV in range(0, fold):
            print(emotion + '/CV' + str(CV))
            test = np.load(execute_name + '/' + emotion + '/CV' + str(CV) + '/test.npy')
            if emotion == 'valence':
                Y_true = Train_Y_valence
            elif emotion == 'arousal':
                Y_true = Train_Y_arousal
            Y_true = Y_true[test]
            for root, subdirs, files in os.walk(execute_name + '/' + emotion + '/CV' + str(CV)):
                for f in files:
                    if os.path.splitext(f)[1] == '.h5' and 'R2pr_'+format(R2_pearsonr_max[emotion][CV], '.5f') in f:
                        print(f)
                        model.load_weights(os.path.join(root, f), by_name=True)
                        Y_predict = model.predict(Train_X[test, :, :])
                        Y_true = Y_true.reshape(-1, 1)
                        AMG1608_R2_pearsonr_max[emotion][CV] = np.square(pearsonr(Y_true, Y_predict)[0][0])
                        print(AMG1608_R2_pearsonr_max[emotion][CV])
                    if os.path.splitext(f)[1] == '.h5' and 'R2_'+format(R2_max[emotion][CV], '.5f') in f:
                        model.load_weights(os.path.join(root, f), by_name=True)
                        Y_predict = model.predict(Train_X[test, :, :])
                        Y_true = Y_true.reshape(-1, 1)
                        AMG1608_R2_max[emotion][CV] = r2_score(Y_true, Y_predict)
print('AMG1608 maximum:')
print('R2_pearsonr_max: ' + str(AMG1608_R2_pearsonr_max))
print('R2_max: ' + str(AMG1608_R2_max))
print('Averaging Valence R2_pearsonr_max: ' + str(np.mean(AMG1608_R2_pearsonr_max['valence'])) +
      ',R2_max: ' + str(np.mean(AMG1608_R2_max['valence'])))
print('Averaging Arousal R2_pearsonr_max: ' + str(np.mean(AMG1608_R2_pearsonr_max['arousal'])) +
      ',R2_max: ' + str(np.mean(AMG1608_R2_max['arousal'])))

data = OrderedDict()
data.update({"R2_pearsonr_max": [[emotions[0]], AMG1608_R2_pearsonr_max[emotions[0]],
                                 ['average', np.mean(AMG1608_R2_pearsonr_max[emotions[0]])],
                                 [emotions[1]], AMG1608_R2_pearsonr_max[emotions[1]],
                                 ['average', np.mean(AMG1608_R2_pearsonr_max[emotions[1]])]]})
data.update({"R2_max": [[emotions[0]], AMG1608_R2_max[emotions[0]],
                        ['average', np.mean(AMG1608_R2_max[emotions[0]])],
                        [emotions[1]], AMG1608_R2_max[emotions[1]],
                        ['average', np.mean(AMG1608_R2_max[emotions[1]])]]})
save_data(execute_name + '/' + test_dataset_name + '.xls', data)

# test CH818
test_dataset_name = 'CH_818'
print('Testing: ' + test_dataset_name)
CH818_R2_pearsonr_max = dict(zip(emotions, np.zeros((len(emotions), fold))))
CH818_R2_max = dict(zip(emotions, np.zeros((len(emotions), fold))))
Train_Y_valence = np.load(save_path + '/' + test_dataset_name + '/Train_Y_valence.npy')
Train_Y_arousal = np.load(save_path + '/' + test_dataset_name + '/Train_Y_arousal.npy')
Train_X = np.load(save_path + '/' + test_dataset_name + '/Train_X.npy')
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
            for root, subdirs, files in os.walk(execute_name + '/' + emotion + '/CV' + str(CV)):
                for f in files:
                    if os.path.splitext(f)[1] == '.h5' and 'R2pr_'+format(R2_pearsonr_max[emotion][CV], '.5f') in f:
                        print(f)
                        model.load_weights(os.path.join(root, f), by_name=True)
                        Y_predict = model.predict(Train_X)
                        Y_true = Y_true.reshape(-1, 1)
                        CH818_R2_pearsonr_max[emotion][CV] = np.square(pearsonr(Y_true, Y_predict)[0][0])
                        print(CH818_R2_pearsonr_max[emotion][CV])
                    if os.path.splitext(f)[1] == '.h5' and 'R2_'+format(R2_max[emotion][CV], '.5f') in f:
                        model.load_weights(os.path.join(root, f), by_name=True)
                        Y_predict = model.predict(Train_X)
                        Y_true = Y_true.reshape(-1, 1)
                        CH818_R2_max[emotion][CV] = r2_score(Y_true, Y_predict)
print('CH818 maximum:')
print('R2_pearsonr_max: ' + str(CH818_R2_pearsonr_max))
print('R2_max: ' + str(CH818_R2_max))
print('Averaging Valence R2_pearsonr_max: ' + str(np.mean(CH818_R2_pearsonr_max['valence'])) +
      ',R2_max: ' + str(np.mean(CH818_R2_max['valence'])))
print('Averaging Arousal R2_pearsonr_max: ' + str(np.mean(CH818_R2_pearsonr_max['arousal'])) +
      ',R2_max: ' + str(np.mean(CH818_R2_max['arousal'])))
data = OrderedDict()
data.update({"R2_pearsonr_max": [[emotions[0]], CH818_R2_pearsonr_max[emotions[0]],
                                 ['average', np.mean(CH818_R2_pearsonr_max[emotions[0]])],
                                 [emotions[1]], CH818_R2_pearsonr_max[emotions[1]],
                                 ['average', np.mean(CH818_R2_pearsonr_max[emotions[1]])]]})
data.update({"R2_max": [[emotions[0]], CH818_R2_max[emotions[0]],
                        ['average', np.mean(CH818_R2_max[emotions[0]])],
                        [emotions[1]], CH818_R2_max[emotions[1]],
                        ['average', np.mean(CH818_R2_max[emotions[1]])]]})
save_data(execute_name + '/' + test_dataset_name + '.xls', data)


# test MER60
test_dataset_name = 'MER_60'
print('Testing: ' + test_dataset_name)
MER60_R2_pearsonr_max = dict(zip(emotions, np.zeros((len(emotions), fold))))
MER60_R2_max = dict(zip(emotions, np.zeros((len(emotions), fold))))
Train_Y_valence = np.load(save_path + '/' + test_dataset_name + '/Train_Y_valence.npy')
Train_Y_arousal = np.load(save_path + '/' + test_dataset_name + '/Train_Y_arousal.npy')
Train_X = np.load(save_path + '/' + test_dataset_name + '/Train_X.npy')
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
            for root, subdirs, files in os.walk(execute_name + '/' + emotion + '/CV' + str(CV)):
                for f in files:
                    if os.path.splitext(f)[1] == '.h5' and 'R2pr_'+format(R2_pearsonr_max[emotion][CV], '.5f') in f:
                        print(f)
                        model.load_weights(os.path.join(root, f), by_name=True)
                        Y_predict = model.predict(Train_X)
                        Y_true = Y_true.reshape(-1, 1)
                        MER60_R2_pearsonr_max[emotion][CV] = np.square(pearsonr(Y_true, Y_predict)[0][0])
                        print(MER60_R2_pearsonr_max[emotion][CV])
                    if os.path.splitext(f)[1] == '.h5' and 'R2_'+format(R2_max[emotion][CV], '.5f') in f:
                        model.load_weights(os.path.join(root, f), by_name=True)
                        Y_predict = model.predict(Train_X)
                        Y_true = Y_true.reshape(-1, 1)
                        MER60_R2_max[emotion][CV] = r2_score(Y_true, Y_predict)
print('MER60 maximum:')
print('R2_pearsonr_max: ' + str(MER60_R2_pearsonr_max))
print('R2_max: ' + str(MER60_R2_max))
print('Averaging Valence R2_pearsonr_max: ' + str(np.mean(MER60_R2_pearsonr_max['valence'])) +
      ',R2_max: ' + str(np.mean(MER60_R2_max['valence'])))
print('Averaging Arousal R2_pearsonr_max: ' + str(np.mean(MER60_R2_pearsonr_max['arousal'])) +
      ',R2_max: ' + str(np.mean(MER60_R2_max['arousal'])))
data = OrderedDict()
data.update({"R2_pearsonr_max": [[emotions[0]], MER60_R2_pearsonr_max[emotions[0]],
                                 ['average', np.mean(MER60_R2_pearsonr_max[emotions[0]])],
                                 [emotions[1]], MER60_R2_pearsonr_max[emotions[1]],
                                 ['average', np.mean(MER60_R2_pearsonr_max[emotions[1]])]]})
data.update({"R2_max": [[emotions[0]], MER60_R2_max[emotions[0]],
                        ['average', np.mean(MER60_R2_max[emotions[0]])],
                        [emotions[1]], MER60_R2_max[emotions[1]],
                        ['average', np.mean(MER60_R2_max[emotions[1]])]]})
save_data(execute_name + '/' + test_dataset_name + '.xls', data)
