import numpy as np
import keras.backend.tensorflow_backend as KTF
import os
import tensorflow as tf
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from functions import model_structure, metric, Transfer_funcs
from keras.models import Model, load_model
from keras.layers import Input
from scipy.io import loadmat

# GPU speed limit
def get_session(gpu_fraction=0.6):
    # Assume that you have 6GB of GPU memory and want to allocate ~2GB
    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
KTF.set_session(get_session())

# Parameters
dataset_name = 'AMG_1608'
save_path = '/data/Wayne'
sec_length = 29
output_sample_rate = 22050


# load Mel spectrum features
print('load Mel spectrum features')
Train_X = np.load(save_path + '/' + dataset_name + '/Train_X@' + str(output_sample_rate) + 'Hz.npy')
Train_Y_valence = np.load(save_path + '/' + dataset_name + '/Train_Y_valence.npy')
Train_Y_arousal = np.load(save_path + '/' + dataset_name + '/Train_Y_arousal.npy')

input_tensor = Input(shape=(1, output_sample_rate*sec_length))
feature_tensor = model_structure.extract_melspec(input_tensor=input_tensor, sr=output_sample_rate)
feature_extractor = Model(inputs=input_tensor, outputs=feature_tensor)
Train_X_feat = feature_extractor.predict(Train_X)
print('Train_X_feat: ' + str(Train_X_feat.shape))
np.save(save_path + '/' + dataset_name + '/Train_X@' + str(output_sample_rate) + 'Hz_melSpec.npy', 'Train_X_feat')

# load rhythm features
print('load rhythm features')
if os.path.exists(save_path+'/'+dataset_name+'/Train_X@'+str(output_sample_rate)+'Hz_rCTA.mat'):
    Train_X = loadmat(save_path+'/'+dataset_name+'/Train_X@'+str(output_sample_rate)+'Hz_rCTA.mat')
Train_X = Train_X['Train_X_feature']
Train_X_feat = Train_X.reshape((Train_X.shape[0], Train_X.shape[1], Train_X.shape[2], 1))
print('Train_X_feat: ' + str(Train_X_feat.shape))

# feature_tensor = Input(shape=(Train_X_feat.shape[1], Train_X_feat.shape[2], 1))
# extractor = model_structure.compact_cnn_extractor(feature_tensor=feature_tensor)
# regressor = model_structure.regression_classifier(encoded_audio_tensor=extractor)
# feature_regressor = Model(inputs=feature_tensor, outputs=regressor)
# emotions = ['valence', 'arousal']
# CH818_R2_pearsonr_max = dict(zip(emotions, np.zeros((len(emotions)))))
# R2_pearsonr_max = dict(zip(emotions, np.zeros((len(emotions)))))
# for emotion in emotions:
#     if emotion == 'valence':
#         Y_true = Train_Y_valence
#     elif emotion == 'arousal':
#         Y_true = Train_Y_arousal
#     print(emotion)
#     if os.path.exists(source_execute_name + '/' + emotion + '/log_0_logs.json'):
#         with open(source_execute_name + '/' + emotion + '/log_0_logs.json', "r") as fb:
#             data = json.load(fb)
#             R2_pearsonr_max[emotion] = max(data['train_R2_pearsonr'])
#     for root, subdirs, files in os.walk(source_execute_name + '/' + emotion):
#         for f in files:
#             if os.path.splitext(f)[1] == '.h5' and 'train_R2pr_'+format(R2_pearsonr_max[emotion], '.5f') in f:
#                 print(f)
#                 model = load_model(os.path.join(root, f), custom_objects={'Melspectrogram': Melspectrogram,
#                                                                           'R2': metric.R2})
#                 print(str(len(feature_regressor.get_weights())))
#                 print(str(len(model.get_weights())))
#                 feature_regressor.set_weights(model.get_weights()[3:])
#                 print('load weights finished...')
#                 Y_predict = feature_regressor.predict(Train_X_feat)
#                 Y_true = Y_true.reshape(-1, 1)
#                 CH818_R2_pearsonr_max[emotion] = np.square(pearsonr(Y_true, Y_predict)[0][0])
#                 print(CH818_R2_pearsonr_max[emotion])
