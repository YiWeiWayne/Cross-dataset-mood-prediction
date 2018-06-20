import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from functions import model_structure, ADDA_funcs
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import SGD, Adam, RMSprop
import os, json
import numpy as np
import datetime
from keras.utils import generic_utils
from keras import backend as K
from functions.Custom_layers import Std2DLayer
import random
from keras.utils.vis_utils import plot_model


# GPU speed limit
def get_session(gpu_fraction=0.3):
    # Assume that you have 6GB of GPU memory and want to allocate ~2GB
    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
KTF.set_session(get_session())


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


# setting Parameters
algorithm = 'WADDA'
action = 'rCTA'
feature = action
action_description = 'Change regressor to CNN'
# 0.melSpec_lw _20180511.1153.51
# 1.pitch+lw 20180514.0016.35
# 2.rCTA 20180513.2344.55
source_dataset_name = 'CH_818'
target_dataset_name = 'AMG_1608'
save_path = '/mnt/data/Wayne'
now = datetime.datetime.now()
localtime = str(now.year) + str(now.month).zfill(2) + str(now.day).zfill(2) + '.' +\
            str(now.hour).zfill(2) + str(now.minute).zfill(2) + '.' + str(now.second).zfill(2)
execute_name = save_path + '/(' + action + ')' + algorithm + '_S_' + source_dataset_name + \
               '_T_' + target_dataset_name + '_' + localtime
emotions = ['valence', 'arousal']
if action == 'melSpec_lw':
    source_execute_name = save_path + '/(' + action + ')' + source_dataset_name + '_20180621.0028.13'
elif action == 'pitch+lw':
    source_execute_name = save_path + '/(' + action + ')' + source_dataset_name + '_20180621.0038.25'
elif action == 'rCTA':
    source_execute_name = save_path + '/(' + action + ')' + source_dataset_name + '_20180621.0044.30'
sec_length = 29
output_sample_rate = 22050
load_weights_source_feature_extractor = True
load_weights_source_classifier = True
load_weights_target_feature_extractor = True
save_best_only = True
save_weights_only = False
save_source_model = False
source_epoch_th = 2
target_epoch_th = 2
save_key = 'pearsonr'  # 1.R2 2.pearsonr

# network parameters
batch_size = 16
encoded_size = 384
epochs = 4000
k_d = 5
k_g = 1
use_shared_dis_reg = False
if use_shared_dis_reg:
    reg_output_activation = 'tanh'
soft_noise = 0.1
regressor_net = 'cnn'
discriminator_net = 'cnn'

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
regressor_optimizer = 'adam'
regressor_loss = 'mean_squared_error'

# discriminator parameters
if discriminator_net == 'nn':
    discriminator_units = [128, 64, 1]
    discriminator_activations = ['elu', 'elu', 'sigmoid']
elif discriminator_net == 'cnn':
    discriminator_units = [64, 128, 256, 1]
    discriminator_activations = ['elu', 'elu', 'elu', 'sigmoid']
    discriminator_kernels = [8, 4, 2]
    discriminator_strides = [4, 2, 1]
    discriminator_paddings = ['valid', 'valid', 'valid']
discriminator_optimizer = 'rms'  # 2.rms 3. adam 4.sgd_no_decay
target_optimizer = 'rms'  # 2.rms 3. adam 4.sgd_no_decay
use_wloss = True
if use_wloss:
    discriminator_activations[-1] = 'linear'
    use_clip_weights = True
    discriminator_loss = 'wloss'
else:
    discriminator_activations[-1] = 'sigmoid'
    use_clip_weights = False
    discriminator_loss = 'binary_crossentropy'
if use_clip_weights:
    clip_value = 0.01

#  Feature extractor parameters
if feature == 'melSpec':  # dim(96, 2498, 1)
    filters = [128, 128, 128]
    kernels = [(96, 10), (1, 6), (1, 4)]
    strides = [(1, 8), (1, 6), (1, 4)]
    paddings = ['valid', 'valid', 'valid']
    poolings = [(1, 1), (1, 1), (1, 1), (1, 13)]
    dr_rate = [0, 0, 0, 0]
elif feature == 'melSpec_lw':  # dim(96, 1249, 1)
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
elif feature == 'pitch':
    filters = [128, 128, 128, 128]
    kernels = [(360, 6), (1, 4), (1, 4), (1, 3)]
    strides = [(1, 4), (1, 4), (1, 4), (1, 3)]
    paddings = ['valid', 'valid', 'valid', 'valid']
    poolings = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 13)]
    dr_rate = [0, 0, 0, 0, 0]
elif feature == 'pitch+lw':  # dim(360, 1249, 1)
    filters = [128, 128, 128, 128, 128]
    kernels = [(360, 4), (1, 4), (1, 3), (1, 3), (1, 3)]
    strides = [(1, 3), (1, 2), (1, 3), (1, 3), (1, 2)]
    paddings = ['valid', 'valid', 'valid', 'valid', 'valid']
    poolings = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 11)]
    dr_rate = [0, 0, 0, 0, 0, 0]

# Add regularization term to classification loss for updating source feature extractor (fail)
use_regularization = False
if use_regularization:
    loss_weights = [1., 1., 1.]

# Parameters saved
para_line = []
para_line.append('# setting Parameters \n')
para_line.append('algorithm:' + algorithm + '\n')
para_line.append('action:' + str(action) + '\n')
para_line.append('feature:' + feature + '\n')
para_line.append('action_description:' + str(action_description) + '\n')
para_line.append('source_dataset_name:' + source_dataset_name + '\n')
para_line.append('target_dataset_name:' + target_dataset_name + '\n')
para_line.append('save_path:' + save_path + '\n')
para_line.append('source_execute_name:' + source_execute_name + '\n')
para_line.append('execute_name:' + execute_name + '\n')
para_line.append('emotions :' + str(emotions) + '\n')
para_line.append('sec_length:' + str(sec_length) + '\n')
para_line.append('output_sample_rate:' + str(output_sample_rate) + '\n')
para_line.append('load_weights_source_feature_extractor:' + str(load_weights_source_feature_extractor) + '\n')
para_line.append('load_weights_source_classifier:' + str(load_weights_source_classifier) + '\n')
para_line.append('load_weights_target_feature_extractor:' + str(load_weights_target_feature_extractor) + '\n')
para_line.append('save_best_only:' + str(save_best_only) + '\n')
para_line.append('save_weights_only:' + str(save_weights_only) + '\n')
para_line.append('save_source_model:' + str(save_source_model) + '\n')
para_line.append('source_epoch_th:' + str(source_epoch_th) + '\n')
para_line.append('target_epoch_th:' + str(target_epoch_th) + '\n')
para_line.append('save_key:' + str(save_key) + '\n')

# network
para_line.append('\n# network Parameters \n')
para_line.append('batch_size:' + str(batch_size) + '\n')
para_line.append('encoded_size:' + str(encoded_size) + '\n')
para_line.append('epochs:' + str(epochs) + '\n')
para_line.append('k_d:' + str(k_d) + '\n')
para_line.append('k_g:' + str(k_g) + '\n')
para_line.append('use_shared_dis_reg:' + str(use_shared_dis_reg) + '\n')
if use_shared_dis_reg:
    para_line.append('reg_output_activation:' + str(reg_output_activation) + '\n')
para_line.append('soft_noise:' + str(soft_noise) + '\n')
para_line.append('regressor_net:' + str(regressor_net) + '\n')
para_line.append('discriminator_net:' + str(discriminator_net) + '\n')
# regressor
para_line.append('\n# regressor Parameters \n')
para_line.append('regressor_units:' + str(regressor_units) + '\n')
para_line.append('regressor_activations :' + str(regressor_activations ) + '\n')
if regressor_net == 'cnn':
    para_line.append('regressor_kernels :' + str(regressor_kernels ) + '\n')
    para_line.append('regressor_strides :' + str(regressor_strides ) + '\n')
    para_line.append('regressor_paddings :' + str(regressor_paddings ) + '\n')
para_line.append('regressor_loss:' + str(regressor_loss) + '\n')
para_line.append('regressor_optimizer:' + str(regressor_optimizer) + '\n')
# discriminator
para_line.append('\n# discriminator Parameters \n')
para_line.append('discriminator_units:' + str(discriminator_units) + '\n')
para_line.append('discriminator_activations :' + str(discriminator_activations ) + '\n')
if discriminator_net == 'cnn':
    para_line.append('discriminator_kernels:' + str(discriminator_kernels) + '\n')
    para_line.append('discriminator_strides:' + str(discriminator_strides) + '\n')
    para_line.append('discriminator_paddings:' + str(discriminator_paddings) + '\n')
para_line.append('use_wloss:' + str(use_wloss) + '\n')
para_line.append('use_clip_weights:' + str(use_clip_weights) + '\n')
if use_clip_weights:
    para_line.append('clip_value:' + str(clip_value) + '\n')
para_line.append('discriminator_loss:' + str(discriminator_loss) + '\n')
para_line.append('discriminator_optimizer:' + str(discriminator_optimizer) + '\n')
para_line.append('target_optimizer:' + str(target_optimizer) + '\n')
# Feature extractor
para_line.append('\n# Feature extractor Parameters \n')
para_line.append('filters:' + str(filters) + '\n')
para_line.append('kernels:' + str(kernels) + '\n')
para_line.append('paddings:' + str(paddings) + '\n')
para_line.append('strides:' + str(strides) + '\n')
para_line.append('poolings:' + str(poolings) + '\n')
para_line.append('dr_rate:' + str(dr_rate) + '\n')
# regularization
para_line.append('use_regularization:' + str(use_regularization) + '\n')
if use_regularization:
    para_line.append('loss_weights:' + str(loss_weights) + '\n')
# save
if not os.path.exists(execute_name):
    os.makedirs(execute_name)
paraFile = open(os.path.join(execute_name, 'Parameters.txt'), 'w')
paraFile.writelines(para_line)
paraFile.close()

# Start to choose emotion for training
for emotion in emotions:
    # 0
    # Data load
    Source_Train_Y = np.load(save_path + '/' + source_dataset_name + '/Train_Y_' + emotion + '.npy')
    Target_Train_Y = np.load(save_path + '/' + target_dataset_name + '/Train_Y_' + emotion + '.npy')
    Source_Train_X = np.load(save_path + '/' + source_dataset_name +
                             '/Train_X' + '@' + str(output_sample_rate) + 'Hz_' + feature + '.npy')
    Target_Train_X = np.load(save_path + '/' + target_dataset_name +
                             '/Train_X' + '@' + str(output_sample_rate) + 'Hz_' + feature + '.npy')
    print("Source_Train_X shape:" + str(Source_Train_X.shape))
    print("Target_Train_X shape:" + str(Target_Train_X.shape))
    if KTF._SESSION:
        print('Reset session.')
        KTF.clear_session()
        KTF.set_session(get_session())

    # sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    sgd_no_decay = SGD(lr=0.001, decay=0, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None)
    rms = RMSprop(lr=0.001)
    # regressor
    if regressor_optimizer == 'rms':
        reg_opt = rms
    elif regressor_optimizer == 'adam':
        reg_opt = adam
    elif regressor_optimizer == 'sgd_no_decay':
        reg_opt = sgd_no_decay
    # discriminator
    if discriminator_optimizer == 'rms':
        dis_opt = rms
    elif discriminator_optimizer == 'adam':
        dis_opt = adam
    elif discriminator_optimizer == 'sgd_no_decay':
        dis_opt = sgd_no_decay
    # target
    if target_optimizer == 'rms':
        tar_opt = rms
    elif target_optimizer == 'adam':
        tar_opt = adam
    elif target_optimizer == 'sgd_no_decay':
        tar_opt = sgd_no_decay

    # 1
    # Discriminator generated
    source_or_target_tensor = Input(shape=(encoded_size, ))
    if discriminator_net == 'cnn':
        discriminator_model = Model(inputs=source_or_target_tensor,
                                    outputs=model_structure.cnn_classifier(x=source_or_target_tensor, input_channel=3,
                                                                           units=discriminator_units,
                                                                           activations=discriminator_activations,
                                                                           kernels=discriminator_kernels,
                                                                           strides=discriminator_strides,
                                                                           paddings=discriminator_paddings))

    else:
        discriminator_model = Model(inputs=source_or_target_tensor,
                                    outputs=model_structure.nn_classifier(x=source_or_target_tensor,
                                                                          units=discriminator_units,
                                                                          activations=discriminator_activations))
    if use_wloss:
        print('wloss!')
        discriminator_model.compile(loss=wasserstein_loss, optimizer=dis_opt, metrics=['accuracy'])
    else:
        discriminator_model.compile(loss=discriminator_loss, optimizer=dis_opt, metrics=['accuracy'])
    print("discriminator_model summary:")
    with open(os.path.join(execute_name, feature+'$discriminator_model_summary.txt'), 'w') as fh:
        discriminator_model.summary(print_fn=lambda x: fh.write(x + '\n'))
        # plot_model(discriminator_model, to_file=execute_name + '/' + algorithm + '@' + feature +
        #                                         'discriminator_model.png', show_shapes=True)

    # 2
    # Regressor generated
    if use_shared_dis_reg:
        target_tensor = discriminator_model.input
        # get the last feature map
        x = discriminator_model.layers[-2].output
        # connect regressor activation
        x = Dense(1, activation=reg_output_activation,
                  kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')(x)
        regressor_model = Model(inputs=target_tensor, outputs=x)
    else:
        target_tensor = Input(shape=(encoded_size,))
        if regressor_net == 'nn':
            regressor_model = Model(inputs=target_tensor,
                                    outputs=model_structure.nn_classifier(x=target_tensor,
                                                                          units=regressor_units,
                                                                          activations=regressor_activations))
        elif regressor_net == 'cnn':
            regressor_model = Model(inputs=target_tensor,
                                    outputs=model_structure.cnn_classifier(x=target_tensor,
                                                                           input_channel=3,
                                                                           units=regressor_units,
                                                                           activations=regressor_activations,
                                                                           kernels=regressor_kernels,
                                                                           strides=regressor_strides,
                                                                           paddings=regressor_paddings))
    print("regressor_model summary:")
    with open(os.path.join(execute_name, feature + '$regressor_model_summary.txt'), 'w') as fh:
        regressor_model.summary(print_fn=lambda x: fh.write(x + '\n'))
        # plot_model(classifier_model, to_file=execute_name + '/' + algorithm + '@' + feature +
        #                                      'classifier_model.png', show_shapes=True)

    # 3
    # Source & Target feature extraction generated
    source_feature_tensor = Input(shape=(Source_Train_X.shape[1], Source_Train_X.shape[2], 1))
    source_feature_extractor = model_structure.compact_cnn_extractor(x=source_feature_tensor,
                                                                     filters=filters, kernels=kernels, strides=strides,
                                                                     paddings=paddings, poolings=poolings,
                                                                     dr_rate=dr_rate)
    source_extractor = Model(inputs=source_feature_tensor, outputs=source_feature_extractor)

    target_feature_tensor = Input(shape=(Target_Train_X.shape[1], Target_Train_X.shape[2], 1))
    target_feature_extractor = model_structure.compact_cnn_extractor(x=target_feature_tensor,
                                                                     filters=filters, kernels=kernels, strides=strides,
                                                                     paddings=paddings, poolings=poolings,
                                                                     dr_rate=dr_rate)
    target_extractor = Model(inputs=target_feature_tensor, outputs=target_feature_extractor)

    # 4
    # Combine Target and discriminator
    target_discriminator_model = Model(inputs=target_feature_tensor,
                                       outputs=discriminator_model(target_feature_extractor))
    if use_wloss:
        print('wloss!')
        target_discriminator_model.compile(loss=wasserstein_loss, optimizer=tar_opt, metrics=['accuracy'])
    else:
        target_discriminator_model.compile(loss=discriminator_loss, optimizer=tar_opt, metrics=['accuracy'])
    print("target_discriminator_model summary:")
    with open(os.path.join(execute_name, feature+'$target_discriminator_model_summary.txt'), 'w') as fh:
        target_discriminator_model.summary(print_fn=lambda x: fh.write(x + '\n'))
        # plot_model(target_discriminator_model, to_file=execute_name + '/' + algorithm + '@' + feature +
        #                                                'target_discriminator_model.png', show_shapes=True)

    # 5
    # Combine Target and regressor
    target_regressor_model = Model(inputs=target_feature_tensor,
                                   outputs=regressor_model(target_feature_extractor))
    target_regressor_model.compile(loss=regressor_loss, optimizer=regressor_optimizer, metrics=['accuracy'])
    print("target_regressor_model summary:")
    with open(os.path.join(execute_name, feature + '$target_regressor_model_summary.txt'), 'w') as fh:
        target_regressor_model.summary(print_fn=lambda x: fh.write(x + '\n'))
        # plot_model(target_regressor_model, to_file=execute_name + '/' + algorithm + '@' + feature +
        #                                             'target_regressor_model.png', show_shapes=True)

    # 5'
    # Combine Source and regressor
    source_regressor_model = Model(inputs=source_feature_tensor,
                                   outputs=regressor_model(source_feature_extractor))
    source_regressor_model.compile(loss=regressor_loss, optimizer=regressor_optimizer, metrics=['accuracy'])
    print("source_regressor_model summary:")
    with open(os.path.join(execute_name, feature + '$source_regressor_model_summary.txt'), 'w') as fh:
        source_regressor_model.summary(print_fn=lambda x: fh.write(x + '\n'))
    # 5''
    # Combine Source and regressor and discriminator

    # 6
    # Load weights
    if os.path.exists(source_execute_name + '/' + emotion + '/log_0_logs.json'):
        with open(source_execute_name + '/' + emotion + '/log_0_logs.json', "r") as fb:
            data = json.load(fb)
            if save_key == 'pearsonr':
                train_max_temp = np.square(max(data['train_pearsonr']))
            else:
                train_max_temp = max(data['train_R2_pearsonr'])
            print('train_max_temp: ' + str(train_max_temp))
            for root, subdirs, files in os.walk(source_execute_name + '/' + emotion):
                for f in files:
                    if os.path.splitext(f)[1] == '.h5' and \
                                            'train_R2pr_' + format(train_max_temp, '.5f') in f:
                        print(f)
                        model = load_model(os.path.join(root, f), custom_objects={'Std2DLayer': Std2DLayer})
                        if load_weights_source_feature_extractor:
                            print('set source')
                            source_extractor.set_weights(model.get_weights()[:-2*len(regressor_units)])
                        # target_discriminator_model's weights will be set in the same time.
                        if load_weights_target_feature_extractor:
                            print('set target')
                            target_extractor.set_weights(model.get_weights()[:-2*len(regressor_units)])
                        if load_weights_source_classifier:
                            print('set classifier')
                            regressor_model.set_weights(model.get_weights()[-2*len(regressor_units):])

    # 7
    # Lock source extractor weights
    source_extractor.trainable = False
    # Lock classifier weights
    regressor_model .trainable = False

    # 8
    # Generator for source and target data
    source_data_generator = ADDA_funcs.data_generator(Source_Train_X, Source_Train_Y, batch_size)
    target_data_generator = ADDA_funcs.data_generator(Target_Train_X, Target_Train_Y, batch_size)

    # 9
    # training init
    total_training_steps = int(len(Target_Train_Y) / (batch_size * (k_g*2 + k_d)))
    model_path = execute_name + '/' + emotion + '/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    train_loss = np.zeros(shape=2)
    reg_loss = np.zeros(shape=len(source_regressor_model.metrics_names))
    val_loss = np.zeros(shape=len(target_regressor_model.metrics_names))
    loss_fake = np.zeros(shape=len(target_discriminator_model.metrics_names))
    loss_dis = np.zeros(shape=len(discriminator_model.metrics_names))
    log_data = {
                "train_MSE": [],
                "reg_MSE": [],
                "val_MSE": [],
                "train_loss_fake": [],
                "train_loss_dis": [],
                "train_R2_pearsonr": [],
                "source_val_R2_pearsonr": [],
                "val_R2_pearsonr": [],
                "train_pearsonr": [],
                "source_val_pearsonr": [],
                "val_pearsonr": [],
                }
    log_data = ADDA_funcs.log_dump_all(model_path=model_path, run_num=0,
                                       source_regressor_model=source_regressor_model,
                                       train_x=Source_Train_X, train_y=Source_Train_Y,
                                       target_regressor_model=target_regressor_model,
                                       val_x=Target_Train_X, val_y=Target_Train_Y,
                                       log_data=log_data,
                                       train_loss=train_loss, reg_loss=reg_loss, val_loss=val_loss,
                                       loss_fake=loss_fake, loss_dis=loss_dis,
                                       save_best_only=save_best_only, save_weights_only=save_weights_only,
                                       save_source_model=save_source_model,
                                       source_epoch_th=source_epoch_th, target_epoch_th=target_epoch_th)
    # 10
    # Goal: Make discriminator loss high and fake loss low.
    # Begin to train target feature extractor and discriminator alternatively
    for epoch in range(0, epochs):
        print('Epoch: ' + str(epoch).zfill(4) + '/' + str(epochs).zfill(4))
        # Generator for source and target data
        train_loss = np.zeros(shape=2)
        reg_loss = np.zeros(shape=len(source_regressor_model.metrics_names))
        val_loss = np.zeros(shape=len(target_regressor_model.metrics_names))
        loss_fake = np.zeros(shape=len(discriminator_model.metrics_names))
        loss_dis = np.zeros(shape=len(discriminator_model.metrics_names))
        progbar = generic_utils.Progbar(len(Target_Train_Y))
        for t in range(0, total_training_steps):
            # Train discriminator: taget = 1, source = -1, use discriminator model(To discriminate source and target)
            for i in range(k_d):
                sample_source_x, sample_source_y = next(source_data_generator)
                sample_target_x, sample_target_y = next(target_data_generator)
                if use_wloss:
                    source_y = -np.ones((len(sample_source_y), 1)) + random.uniform(-soft_noise, soft_noise)
                else:
                    source_y = np.zeros((len(sample_source_y), 1)) + random.uniform(-soft_noise, soft_noise)
                target_y = np.ones((len(sample_target_y), 1)) + random.uniform(-soft_noise, soft_noise)
                source_tensor_output = source_extractor.predict(sample_source_x)
                target_tensor_output = target_extractor.predict(sample_target_x)
                discriminator_model.trainable = True
                loss_s = discriminator_model.train_on_batch(source_tensor_output, source_y)
                loss_t = discriminator_model.train_on_batch(target_tensor_output, target_y)
                loss_dis = np.add((np.add(loss_s, loss_t) / 2), loss_dis)
                if use_clip_weights:
                    # Clip discriminator weights
                    for l in discriminator_model.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                        l.set_weights(weights)
            # Train taget feature extractor, use combined model: target label=-1
            # Trick: inverted target label, to make target similar to source)
            for i in range(k_g):
                sample_target_x, sample_target_y = next(target_data_generator)
                sample_target_x0, sample_target_y0 = next(target_data_generator)
                if use_wloss:
                    target_y = -np.ones(len(sample_target_y)) + random.uniform(-soft_noise, soft_noise)
                    target_y0 = -np.ones(len(sample_target_y0)) + random.uniform(-soft_noise, soft_noise)
                else:
                    target_y = np.zeros(len(sample_target_y)) + random.uniform(-soft_noise, soft_noise)
                    target_y0 = np.zeros(len(sample_target_y0)) + random.uniform(-soft_noise, soft_noise)
                discriminator_model.trainable = False
                loss_f = target_discriminator_model.train_on_batch(sample_target_x, target_y)
                loss_f0 = target_discriminator_model.train_on_batch(sample_target_x0, target_y0)
                loss_fake = np.add((np.add(loss_f, loss_f0) / 2), loss_fake)
            progbar.add(batch_size * (k_g*2 + k_d), values=[("loss_dis", loss_dis),
                                                            ("loss_fake", loss_fake)])
        loss_fake = loss_fake / (total_training_steps * k_g)
        loss_dis = loss_dis / (total_training_steps * k_d)
        log_data = ADDA_funcs.log_dump_all(model_path=model_path, run_num=0,
                                           source_regressor_model=source_regressor_model,
                                           train_x=Source_Train_X, train_y=Source_Train_Y,
                                           target_regressor_model=target_regressor_model,
                                           val_x=Target_Train_X, val_y=Target_Train_Y,
                                           log_data=log_data,
                                           train_loss=train_loss, reg_loss=reg_loss, val_loss=val_loss,
                                           loss_fake=loss_fake, loss_dis=loss_dis,
                                           save_best_only=save_best_only, save_weights_only=save_weights_only,
                                           save_source_model=save_source_model,
                                           source_epoch_th=source_epoch_th, target_epoch_th=target_epoch_th)