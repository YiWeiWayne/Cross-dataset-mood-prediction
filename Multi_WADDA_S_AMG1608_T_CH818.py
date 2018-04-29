import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from functions import model_structure, metric, ADDA_funcs
from keras.models import Model, load_model
from keras.layers import Input, concatenate, Reshape
from keras.optimizers import SGD, Adam, RMSprop
from kapre.time_frequency import Melspectrogram
import os, json
import numpy as np
import datetime
from keras.utils import generic_utils
from keras import backend as K


# GPU speed limit
def get_session(gpu_fraction=1):
    # Assume that you have 6GB of GPU memory and want to allocate ~2GB
    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
KTF.set_session(get_session())


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def make_trainable(model, trainable):
    model.trainable = trainable
    for l in model.layers:
        l.trainable = trainable


# Parameters
algorithm = 'WADDA'
action = '%ml%rc%pl'
action_description = 'Change features to melSpec_lw, rCTA, pitch+lw and enforced discriminator'
features = ['melSpec_lw', 'rCTA', 'pitch+lw']  # 1.melSpec 2.melSpec_lw 3.rCTA 4.rTA 5.pitch 6.pitch+lw
emotions = ['valence', 'arousal']
source_dataset_name = 'AMG_1608'
target_dataset_name = 'CH_818'
save_path = '/data/Wayne'
source_execute_name = save_path + '/(' + action + ')' + source_dataset_name + '_20180427.1613.34'
source_data_num = 1608
target_data_num = 818
para_File = open(source_execute_name + '/Parameters.txt', 'r')
parameters = para_File.readlines()
para_File.close()
for i in range(0, len(parameters)):
    if 'sec_length:' in parameters[i]:
        sec_length = int(parameters[i][len('sec_length:'):-1])
        print(str(sec_length))
    elif 'output_sample_rate:' in parameters[i]:
        output_sample_rate = int(parameters[i][len('output_sample_rate:'):-1])
        print(str(output_sample_rate))
    elif 'batch_size:' in parameters[i]:
        batch_size = int(parameters[i][len('batch_size:'):-1])
        print(str(batch_size))

# batch_size = 4
encoded_size = 32*3
epochs = 4000
k_d = 5
k_g = 1
load_weights_source_feature_extractor = True
load_weights_source_classifier = True
load_weights_target_feature_extractor = True
save_best_only = True
save_weights_only = False
now = datetime.datetime.now()
localtime = str(now.year) + str(now.month).zfill(2) + str(now.day).zfill(2) + '.' +\
            str(now.hour).zfill(2) + str(now.minute).zfill(2) + '.' + str(now.second).zfill(2)
execute_name = save_path + '/(' + action + ')' + algorithm + '_S_' + source_dataset_name + \
               '_T_' + target_dataset_name + '_' + localtime
classifier_loss = 'mean_squared_error'
discriminator_loss = 'binary_crossentropy'
# Following parameter and optimizer set as recommended in paper
clip_value = 0.1
# real_soft_label_min = 0
# real_soft_label_max = 0.3
# fake_soft_label_min = 0.7
# fake_soft_label_max = 1.2
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

# Parameters saved
para_line = []
para_line.append('algorithm:' + algorithm + '\n')
para_line.append('action:' + str(action) + '\n')
para_line.append('action_description:' + str(action_description) + '\n')
para_line.append('feature:' + feature + '\n')
para_line.append('save_path:' + save_path + '\n')
para_line.append('execute_name:' + execute_name + '\n')
para_line.append('source_dataset_name:' + source_dataset_name + '\n')
para_line.append('target_dataset_name:' + target_dataset_name + '\n')
para_line.append('source_execute_name:' + source_execute_name + '\n')
para_line.append('sec_length:' + str(sec_length) + '\n')
para_line.append('output_sample_rate:' + str(output_sample_rate) + '\n')
para_line.append('encoded_size:' + str(encoded_size) + '\n')
para_line.append('batch_size:' + str(batch_size) + '\n')
para_line.append('epochs:' + str(epochs) + '\n')
para_line.append('k_d:' + str(k_d) + '\n')
para_line.append('k_g:' + str(k_g) + '\n')
para_line.append('load_weights_source_feature_extractor:' + str(load_weights_source_feature_extractor) + '\n')
para_line.append('load_weights_source_classifier:' + str(load_weights_source_classifier) + '\n')
para_line.append('load_weights_target_feature_extractor:' + str(load_weights_target_feature_extractor) + '\n')
para_line.append('save_best_only:' + str(save_best_only) + '\n')
para_line.append('save_weights_only:' + str(save_weights_only) + '\n')
para_line.append('classifier_loss:' + str(classifier_loss) + '\n')
para_line.append('discriminator_loss:' + str(discriminator_loss) + '\n')
para_line.append('clip_value:' + str(clip_value) + '\n')
for feature in features:
    para_line.append('feature:' + str(feature) + '\n')
    para_line.append('filters:' + str(filters[feature]) + '\n')
    para_line.append('kernels:' + str(kernels[feature]) + '\n')
    para_line.append('poolings:' + str(poolings[feature]) + '\n')
    para_line.append('feature_sizes:' + str(feature_sizes[feature]) + '\n')
    para_line.append('pretrain_path:' + str(pretrain_path[feature]) + '\n')
# para_line.append('real_soft_label_min:' + str(real_soft_label_min) + '\n')
# para_line.append('real_soft_label_max:' + str(real_soft_label_max) + '\n')
# para_line.append('fake_soft_label_min:' + str(fake_soft_label_min) + '\n')
# para_line.append('fake_soft_label_max:' + str(fake_soft_label_max) + '\n')
if not os.path.exists(execute_name):
    os.makedirs(execute_name)
paraFile = open(os.path.join(execute_name, 'Parameters.txt'), 'w')
paraFile.writelines(para_line)
paraFile.close()

# load data
Source_Train_Y = dict(zip(emotions, np.zeros((source_data_num, 1))))
Target_Train_Y = dict(zip(emotions, np.zeros((target_data_num, 1))))
Source_Train_Y['valence'] = np.load(save_path + '/' + source_dataset_name + '/Train_Y_valence.npy')
Source_Train_Y['arousal'] = np.load(save_path + '/' + source_dataset_name + '/Train_Y_arousal.npy')
Target_Train_Y['valence'] = np.load(save_path + '/' + target_dataset_name + '/Train_Y_valence.npy')
Target_Train_Y['arousal'] = np.load(save_path + '/' + target_dataset_name + '/Train_Y_arousal.npy')
Source_Train_X = dict(zip(features, np.zeros((source_data_num, 1, 1, 1))))
Target_Train_X = dict(zip(features, np.zeros((target_data_num, 1, 1, 1))))
for feature in features:
    print(feature)
    Source_Train_X[feature] = np.load(save_path + '/' + source_dataset_name +
                               '/Train_X@' + str(output_sample_rate) + 'Hz_' + feature + '.npy')
    Target_Train_X[feature] = np.load(save_path + '/' + target_dataset_name +
                             '/Train_X@' + str(output_sample_rate) + 'Hz_' + feature + '.npy')
    print("Source_Train_X shape:" + str(Source_Train_X[feature].shape))
    print("Target_Train_X shape:" + str(Target_Train_X[feature].shape))

# Start to choose emotion for training
for emotion in emotions:
    # # 0
    # # Data load
    # if emotion == 'valence':
    #     Source_Train_Y = np.load(save_path + '/' + source_dataset_name + '/Train_Y_valence.npy')
    #     Target_Train_Y = np.load(save_path + '/' + target_dataset_name + '/Train_Y_valence.npy')
    # elif emotion == 'arousal':
    #     Source_Train_Y = np.load(save_path + '/' + source_dataset_name + '/Train_Y_arousal.npy')
    #     Target_Train_Y = np.load(save_path + '/' + target_dataset_name + '/Train_Y_arousal.npy')
    # Source_Train_X = np.load(save_path + '/' + source_dataset_name +
    #                          '/Train_X' + '@' + str(output_sample_rate) + 'Hz_' + feature + '.npy')
    # Target_Train_X = np.load(save_path + '/' + target_dataset_name +
    #                          '/Train_X' + '@' + str(output_sample_rate) + 'Hz_' + feature + '.npy')
    if KTF._SESSION:
        print('Reset session.')
        KTF.clear_session()
        KTF.set_session(get_session())

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    optimizer = RMSprop(lr=0.00005)
    # 1
    # Discriminator generated(layer_length: 6)
    #  Input: source or target feature extraction output tensor
    # Output: discriminator output tensor
    source_or_target_tensor = Input(shape=(encoded_size, ))
    discriminator_model = Model(inputs=source_or_target_tensor,
                                outputs=model_structure.enforced_domain_classifier(source_or_target_tensor, encoded_size))
    discriminator_model.compile(loss=wasserstein_loss, optimizer=adam, metrics=['accuracy'])
    print("discriminator_model summary:")
    discriminator_model.summary()

    # 2
    # Regression classifier generated(layer_length: 2)
    # Input: target feature extraction output tensor
    # Output: Regression classifier output tensor
    target_tensor = Input(shape=(encoded_size, ))
    classifier_model = Model(inputs=target_tensor,
                             outputs=model_structure.regression_classifier(target_tensor))
    # Lock classifier weights
    make_trainable(classifier_model, False)
    # classifier_model.trainable = False
    # 3
    # Source & Target feature extraction(layer_length: 33)
    # Input: Raw audio sized Input tensor
    # Output: feature extraction output tensor
    # generate source model
    source_feature_tensor = ['', '', '']
    source_extractor_tensor = ['', '', '']
    source_feature_tensor[0] = Input(shape=(feature_sizes[features[0]][0], feature_sizes[features[0]][1],
                                            feature_sizes[features[0]][2]))
    source_extractor_tensor[0] = model_structure.compact_cnn_extractor(feature_tensor=source_feature_tensor[0],
                                                                       filters=filters[features[0]],
                                                                       kernels=kernels[features[0]],
                                                                       poolings=poolings[features[0]])
    source_feature_tensor[1] = Input(shape=(feature_sizes[features[1]][0], feature_sizes[features[1]][1],
                                            feature_sizes[features[1]][2]))
    source_extractor_tensor[1] = model_structure.compact_cnn_extractor(feature_tensor=source_feature_tensor[1],
                                                                       filters=filters[features[1]],
                                                                       kernels=kernels[features[1]],
                                                                       poolings=poolings[features[1]])
    source_feature_tensor[2] = Input(shape=(feature_sizes[features[2]][0], feature_sizes[features[2]][1],
                                            feature_sizes[features[2]][2]))
    source_extractor_tensor[2] = model_structure.compact_cnn_extractor(feature_tensor=source_feature_tensor[2],
                                                                       filters=filters[features[2]],
                                                                       kernels=kernels[features[2]],
                                                                       poolings=poolings[features[2]])
    source_extractor_tensor = concatenate([source_extractor_tensor[0], source_extractor_tensor[1],
                                           source_extractor_tensor[2]])
    source_extractor = Model(inputs=source_feature_tensor, outputs=source_extractor_tensor)
    # Lock source extractor weights
    make_trainable(source_extractor, False)
    # source_extractor.trainable = False
    # generate target model
    target_feature_tensor = ['', '', '']
    target_extractor_tensor = ['', '', '']
    target_feature_tensor[0] = Input(shape=(feature_sizes[features[0]][0], feature_sizes[features[0]][1],
                                            feature_sizes[features[0]][2]))
    target_extractor_tensor[0] = model_structure.compact_cnn_extractor(feature_tensor=target_feature_tensor[0],
                                                                       filters=filters[features[0]],
                                                                       kernels=kernels[features[0]],
                                                                       poolings=poolings[features[0]])
    target_feature_tensor[1] = Input(shape=(feature_sizes[features[1]][0], feature_sizes[features[1]][1],
                                            feature_sizes[features[1]][2]))
    target_extractor_tensor[1] = model_structure.compact_cnn_extractor(feature_tensor=target_feature_tensor[1],
                                                                       filters=filters[features[1]],
                                                                       kernels=kernels[features[1]],
                                                                       poolings=poolings[features[1]])
    target_feature_tensor[2] = Input(shape=(feature_sizes[features[2]][0], feature_sizes[features[2]][1],
                                            feature_sizes[features[2]][2]))
    target_extractor_tensor[2] = model_structure.compact_cnn_extractor(feature_tensor=target_feature_tensor[2],
                                                                       filters=filters[features[2]],
                                                                       kernels=kernels[features[2]],
                                                                       poolings=poolings[features[2]])
    target_extractor_tensor = concatenate([target_extractor_tensor[0], target_extractor_tensor[1],
                                           target_extractor_tensor[2]])
    target_extractor_tensor = Reshape((encoded_size, ))(target_extractor_tensor)
    target_extractor = Model(inputs=target_feature_tensor, outputs=target_extractor_tensor)

    # 4
    # Combine Target(layer_length: 33) and discriminator(layer_length: 6)
    # Input: Raw audio sized Input tensor
    # Output: Discriminator output tensor(from target output)
    make_trainable(discriminator_model, False)
    # discriminator_model.trainable = False
    target_discriminator_model = Model(inputs=target_feature_tensor,
                                       outputs=discriminator_model(target_extractor_tensor))
    target_discriminator_model.compile(loss=wasserstein_loss, optimizer=adam, metrics=['accuracy'])
    print("target_discriminator_model summary:")
    target_discriminator_model.summary()

    # 5
    # Combine Target(layer_length: 33) and classifier(layer_length: 2)
    # Input: Raw audio sized Input tensor
    # Output: classifier output tensor(from target output)
    target_classifier_model = Model(inputs=target_feature_tensor,
                                    outputs=classifier_model(target_extractor_tensor))
    print("target_classifier_model summary:")
    target_classifier_model.summary()

    # 6
    # Load weights
    if os.path.exists(source_execute_name + '/' + emotion + '/log_0_logs.json'):
        with open(source_execute_name + '/' + emotion + '/log_0_logs.json', "r") as fb:
            data = json.load(fb)
            for root, subdirs, files in os.walk(source_execute_name + '/' + emotion):
                for f in files:
                    if os.path.splitext(f)[1] == '.h5' and \
                                            'train_R2pr_' + format(max(data['train_R2_pearsonr']), '.5f') in f:
                        print(f)
                        model = load_model(os.path.join(root, f))
                        if load_weights_source_feature_extractor:
                            print('set source')
                            source_extractor.set_weights(model.get_weights()[:-2])
                        # target_discriminator_model's weights will be set in the same time.
                        if load_weights_target_feature_extractor:
                            print('set target')
                            target_extractor.set_weights(model.get_weights()[:-2])
                        if load_weights_source_classifier:
                            print('set classifier')
                            classifier_model.set_weights(model.get_weights()[-2:])

    # 8
    # Generator for source and target data
    source_data_generator = ADDA_funcs.multi_data_generator(Source_Train_X, Source_Train_Y[emotion], batch_size,
                                                            features, source_data_num)
    target_data_generator = ADDA_funcs.multi_data_generator(Target_Train_X, Target_Train_Y[emotion], batch_size,
                                                            features, target_data_num)
    del Source_Train_X

    # 9
    # training init
    total_training_steps = int(target_data_num / (batch_size * (k_g*2 + k_d)))
    model_path = execute_name + '/' + emotion + '/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    loss_fake = np.zeros(shape=len(target_discriminator_model.metrics_names))
    loss_dis = np.zeros(shape=len(discriminator_model.metrics_names))
    log_data = {
                "train_loss_fake": [],
                "train_loss_dis": [],
                "val_R2_pearsonr": [],
                }
    log_data = ADDA_funcs.log_dump(model_path=model_path, run_num=0,
                                   target_classifier_model=target_classifier_model,
                                   val_x=[Target_Train_X[features[0]], Target_Train_X[features[1]],
                                          Target_Train_X[features[2]]], val_y=Target_Train_Y[emotion],
                                   log_data=log_data, loss_fake=loss_fake, loss_dis=loss_dis,
                                   save_best_only=save_best_only, save_weights_only=save_weights_only)

    # 10
    # Goal: Make discriminator loss high and fake loss low.
    # Begin to train target feature extractor and discriminator alternatively
    for epoch in range(0, epochs):
        print('Epoch: ' + str(epoch).zfill(4) + '/' + str(epochs).zfill(4))
        # Generator for source and target data
        loss_fake = np.zeros(shape=len(discriminator_model.metrics_names))
        loss_dis = np.zeros(shape=len(discriminator_model.metrics_names))
        # progbar = generic_utils.Progbar(len(Target_Train_Y))
        for t in range(0, total_training_steps):
            # Train discriminator: taget = 1, source = -1, use discriminator model(To discriminate source and target)
            for i in range(k_d):
                sample_source_x0, sample_source_x1, sample_source_x2, sample_source_y = next(source_data_generator)
                sample_target_x0, sample_target_x1, sample_target_x2, sample_target_y = next(target_data_generator)
                source_y = -np.ones((len(sample_source_y), 1))
                target_y = np.ones((len(sample_target_y), 1))
                # target_y = np.array([(np.random.randint(4)) / 10] * len(sample_target_y))
                source_tensor_output = source_extractor.predict([sample_source_x0, sample_source_x1, sample_source_x2])
                target_tensor_output = target_extractor.predict([sample_target_x0, sample_target_x1, sample_target_x2])
                make_trainable(discriminator_model, True)
                # discriminator_model.trainable = True
                loss_s = discriminator_model.train_on_batch(source_tensor_output, source_y)
                loss_t = discriminator_model.train_on_batch(target_tensor_output, target_y)
                loss_dis = np.add((np.add(loss_s, loss_t) / 2), loss_dis)
                # Clip discriminator weights
                for l in discriminator_model.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                    l.set_weights(weights)
            # Train taget feature extractor, use combined model: target label=-1
            # Trick: inverted target label, to make target similar to source)
            for i in range(k_g):
                sample_target_x10, sample_target_x11, sample_target_x12, sample_target_y1 = next(target_data_generator)
                sample_target_x20, sample_target_x21, sample_target_x22, sample_target_y2 = next(target_data_generator)
                target_y1 = -np.ones((len(sample_target_y1), 1))
                target_y2 = -np.ones((len(sample_target_y2), 1))
                # sample_target_x0 = np.concatenate((sample_target_x10, sample_target_x20), axis=0)
                # sample_target_x1 = np.concatenate((sample_target_x11, sample_target_x21), axis=0)
                # sample_target_x2 = np.concatenate((sample_target_x12, sample_target_x22), axis=0)
                # combine_y = np.concatenate((target_y1, target_y2), axis=0)
                make_trainable(discriminator_model, False)
                # discriminator_model.trainable = False
                loss_t1 = target_discriminator_model.train_on_batch(
                    [sample_target_x10, sample_target_x11, sample_target_x12], target_y1)
                loss_t2 = target_discriminator_model.train_on_batch(
                    [sample_target_x20, sample_target_x21, sample_target_x22], target_y2)
                loss_fake = np.add((np.add(loss_t1, loss_t2) / 2), loss_fake)
            # progbar.add(batch_size, values=[("loss_dis", loss_dis), ("loss_fake", loss_fake)])
        loss_fake = loss_fake / (total_training_steps * k_g*2)
        loss_dis = loss_dis / (total_training_steps * k_d)
        log_data = ADDA_funcs.log_dump(model_path=model_path, run_num=0,
                                       target_classifier_model=target_classifier_model,
                                       val_x=[Target_Train_X[features[0]], Target_Train_X[features[1]],
                                              Target_Train_X[features[2]]], val_y=Target_Train_Y[emotion],
                                       log_data=log_data, loss_fake=loss_fake, loss_dis=loss_dis,
                                       save_best_only=save_best_only, save_weights_only=save_weights_only)
