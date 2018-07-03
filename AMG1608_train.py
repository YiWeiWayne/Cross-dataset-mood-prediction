import numpy as np
import keras.backend.tensorflow_backend as KTF
import os
import tensorflow as tf
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from functions import model_structure, callback_wayne, Transfer_funcs, metric
import datetime
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam


# GPU speed limit
def get_session(gpu_fraction=0.3):
    # Assume that you have 6GB of GPU memory and want to allocate ~2GB
    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
KTF.set_session(get_session())

action = 'melSpec_lw'
action_description = 'Change regressor to cnn \n' \
                     'and save model pearsonr \n' \
                     'train DCGAN'
feature = action  # 1.melSpec 2.rCTA 3.melSpec_lw 4.pitch+lw
source_dataset_name = 'AMG_1608'
target_dataset_name = 'CH_818'
save_path = '/mnt/data/Wayne'
emotions = ['valence']
source_dataset_path = save_path + '/Dataset/AMG1838_original/amg1838_mp3_original'
source_label_path = save_path + '/Dataset/AMG1838_original/AMG1608/amg1608_v2.xls'
target_dataset_path = save_path + '/Dataset/CH818/mp3'
target_label_path = save_path + '/Dataset/CH818/label/CH818_Annotations.xlsx'
sec_length = 29
output_sample_rate = 22050
patience = []
batch_size = 16
epochs = 300
now = datetime.datetime.now()
localtime = str(now.year) + str(now.month).zfill(2) + str(now.day).zfill(2) + '.' +\
            str(now.hour).zfill(2) + str(now.minute).zfill(2) + '.' + str(now.second).zfill(2)
execute_name = save_path + '/(' + action + ')' + source_dataset_name + 'DCGAN_' + localtime
save_best_only = False
save_weights_only = False
monitor = 'train_pearsonr'
mode = 'max'
load_pretrained_weights = False
regressor_net = 'cnn'
use_pooling = True
use_drop_out = False
regressor_bn = True
poolings = []
dr_rate = []

# regressor parameters
if regressor_net == 'nn':
    regressor_units = [128, 64, 1]
    regressor_activations = ['elu', 'elu', 'tanh']
elif regressor_net == 'cnn':
    regressor_units = [64, 128, 256, 1]
    regressor_activations = ['elu', 'elu', 'elu', 'tanh']
    regressor_kernels = [8, 4, 4]
    regressor_strides = [4, 3, 2]
    regressor_paddings = ['valid', 'valid', 'valid']
regressor_optimizer = 'adam'
regressor_loss = 'mean_squared_error'
use_mp=False
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

para_line = []
para_line.append('# setting Parameters \n')
para_line.append('action:' + str(action) + '\n')
para_line.append('feature:' + feature + '\n')
para_line.append('action_description:' + str(action_description) + '\n')
para_line.append('source_dataset_name:' + source_dataset_name + '\n')
para_line.append('target_dataset_name:' + target_dataset_name + '\n')
para_line.append('source_dataset_path:' + source_dataset_path + '\n')
para_line.append('source_label_path:' + source_label_path + '\n')
para_line.append('target_dataset_path:' + target_dataset_path + '\n')
para_line.append('target_label_path:' + target_label_path + '\n')
para_line.append('save_path:' + save_path + '\n')
para_line.append('execute_name:' + execute_name + '\n')
para_line.append('emotions :' + str(emotions) + '\n')
para_line.append('sec_length:' + str(sec_length) + '\n')
para_line.append('output_sample_rate:' + str(output_sample_rate) + '\n')
para_line.append('load_pretrained_weights:' + str(load_pretrained_weights) + '\n')
para_line.append('patience:' + str(patience) + '\n')
para_line.append('save_best_only:' + str(save_best_only) + '\n')
para_line.append('save_weights_only:' + str(save_weights_only) + '\n')
para_line.append('monitor:' + str(monitor) + '\n')
para_line.append('mode:' + str(mode) + '\n')

# network
para_line.append('\n# network Parameters \n')
para_line.append('batch_size:' + str(batch_size) + '\n')
para_line.append('epochs:' + str(epochs) + '\n')

# Feature extractor
para_line.append('\n# Feature extractor Parameters \n')
para_line.append('filters:' + str(filters) + '\n')
para_line.append('kernels:' + str(kernels) + '\n')
para_line.append('paddings:' + str(paddings) + '\n')
para_line.append('strides:' + str(strides) + '\n')
para_line.append('use_pooling:' + str(use_pooling) + '\n')
para_line.append('use_drop_out:' + str(use_drop_out) + '\n')
if use_pooling:
    para_line.append('poolings:' + str(poolings) + '\n')
if use_drop_out:
    para_line.append('dr_rate:' + str(dr_rate) + '\n')


# regressor
para_line.append('\n# regressor Parameters \n')
para_line.append('regressor_units:' + str(regressor_units) + '\n')
para_line.append('regressor_activations :' + str(regressor_activations ) + '\n')
if 'cnn' in regressor_net:
    para_line.append('regressor_kernels :' + str(regressor_kernels ) + '\n')
    para_line.append('regressor_strides :' + str(regressor_strides ) + '\n')
    para_line.append('regressor_paddings :' + str(regressor_paddings ) + '\n')
    para_line.append('regressor_bn :' + str(regressor_bn) + '\n')
para_line.append('regressor_loss:' + str(regressor_loss) + '\n')
para_line.append('regressor_optimizer:' + str(regressor_optimizer) + '\n')
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
Train_Y_valence = np.load(save_path + '/' + source_dataset_name + '/Train_Y_valence.npy')
Train_Y_arousal = np.load(save_path + '/' + source_dataset_name + '/Train_Y_arousal.npy')
Train_X = np.load(save_path + '/' + source_dataset_name +
                  '/Train_X@' + str(output_sample_rate) + 'Hz_' + feature + '.npy')

Val_Y_valence = np.load(save_path + '/' + target_dataset_name + '/Train_Y_valence.npy')
Val_Y_arousal = np.load(save_path + '/' + target_dataset_name + '/Train_Y_arousal.npy')
Val_X = np.load(save_path + '/' + target_dataset_name +
                '/Train_X@' + str(output_sample_rate) + 'Hz_' + feature + '.npy')
print('Train_X: ' + str(Train_X.shape))
#
# Training
for emotion_axis in emotions:
    if KTF._SESSION:
        print('Reset session.')
        KTF.clear_session()
        KTF.set_session(get_session())
    if emotion_axis == 'valence':
        Train_Y = Train_Y_valence
        Val_Y = Val_Y_valence
    else:
        Train_Y = Train_Y_arousal
        Val_Y = Val_Y_arousal
    # Generator for source and target data
    # train_data_generator = ADDA_funcs.data_generator(Train_X, Train_Y, batch_size)
    # val_data_generator = ADDA_funcs.data_generator(Val_X, Val_Y, batch_size)
    feature_tensor = Input(shape=(Train_X.shape[1], Train_X.shape[2], 1))
    extractor = model_structure.compact_cnn_extractor(x=feature_tensor,
                                                      filters=filters, kernels=kernels, poolings=poolings,
                                                      paddings=paddings, dr_rate=dr_rate, strides=strides,
                                                      use_pooling=use_pooling, use_drop_out=use_drop_out, use_mp=use_mp)
    if regressor_net == 'nn':
        regressor = model_structure.nn_classifier(x=extractor,
                                                  units=regressor_units,
                                                  activations=regressor_activations)
    elif regressor_net == 'cnn':
        regressor = model_structure.cnn_classifier(x=extractor,
                                                   input_channel=2,
                                                   units=regressor_units,
                                                   activations=regressor_activations,
                                                   kernels=regressor_kernels,
                                                   strides=regressor_strides,
                                                   paddings=regressor_paddings,
                                                   bn=regressor_bn)

    model = Model(inputs=feature_tensor, outputs=regressor)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=regressor_optimizer, loss=regressor_loss, metrics=['accuracy'])
    with open(os.path.join(execute_name, 'Model_summary.txt'), 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    model.summary()
    if load_pretrained_weights:
        model.load_weights(save_path + '/compact_cnn_weights.h5', by_name=True)
    model_path = execute_name + '/' + emotion_axis + '/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model.save(model_path + '/init_run.h5')
    LossR2Logger_ModelCheckPoint = callback_wayne.LossR2Logger_ModelCheckPoint(
        train_data=(Train_X, Train_Y), val_data=(Val_X, Val_Y),
        file_name=model_path + '/log', run_num=0,
        filepath=model_path, monitor=monitor, verbose=0,
        save_best_only=save_best_only, save_weights_only=save_weights_only,
        mode=mode, period=1)
    model.fit(Train_X, Train_Y, batch_size=batch_size, epochs=epochs,
              callbacks=[LossR2Logger_ModelCheckPoint],
              verbose=1, shuffle=True, validation_data=(Val_X, Val_Y))