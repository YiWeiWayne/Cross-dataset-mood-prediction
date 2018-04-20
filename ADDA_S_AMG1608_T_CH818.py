import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from functions import model_structure, metric, ADDA_funcs
from keras.models import Model, load_model
from keras.layers import Input
from keras.optimizers import SGD, Adam
from kapre.time_frequency import Melspectrogram
import os, json
import numpy as np
import datetime
from keras.utils import generic_utils


# GPU speed limit
def get_session(gpu_fraction=0.3):
    # Assume that you have 6GB of GPU memory and want to allocate ~2GB
    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
KTF.set_session(get_session())


# Parameters
algorithm = 'ADDA'
feature = 'melSpec_lw'  # 1.melSpec 2.rCTA 3.melSpec_lw
source_dataset_name = 'AMG_1608'
target_dataset_name = 'CH_818'
save_path = '/mnt/data/Wayne'
source_execute_name = save_path + '/' + source_dataset_name + '_20180420.1305.38'
para_File = open(source_execute_name + '/Parameters.txt', 'r')
parameters = para_File.readlines()
para_File.close()
for i in range(0, len(parameters)):
    if 'feature' in parameters[i]:
        if feature in parameters[i]:
            print('load correct source model.')
        else:
            raise ValueError("Source model feature is not corresponding to ADDA.")

sec_length = 29
output_sample_rate = 22050
encoded_size = 32
batch_size = 16
epochs = 500
k_d = 1
k_g = 1
load_weights_source_feature_extractor = True
load_weights_source_classifier = True
load_weights_target_feature_extractor = True
save_best_only = True
save_weights_only = False
now = datetime.datetime.now()
localtime = str(now.year) + str(now.month).zfill(2) + str(now.day).zfill(2) + '.' +\
            str(now.hour).zfill(2) + str(now.minute).zfill(2) + '.' + str(now.second).zfill(2)
execute_name = save_path + '/' + algorithm + '_S_' + source_dataset_name + \
               '_T_' + target_dataset_name + '_' + localtime
classifier_loss = 'mean_squared_error'
discriminator_loss = 'binary_crossentropy'
# real_soft_label_min = 0
# real_soft_label_max = 0.3
# fake_soft_label_min = 0.7
# fake_soft_label_max = 1.2
action = 'Change sampling rate to 22KHz and enlarge the FFT filter size'
if feature == 'melSpec':
    poolings = [(2, 4), (3, 4), (2, 5), (2, 4), (4, 4)]
elif feature == 'melSpec_lw':
    poolings = [(2, 4), (3, 4), (2, 5), (2, 4), (4, 3)]
elif feature == 'rCTA':
    poolings = [(2, 4), (3, 4), (2, 5), (2, 4), (4, 3)]


# Parameters saved
para_line = []
para_line.append('algorithm:' + algorithm + '\n')
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
para_line.append('poolings:' + str(poolings) + '\n')
# para_line.append('real_soft_label_min:' + str(real_soft_label_min) + '\n')
# para_line.append('real_soft_label_max:' + str(real_soft_label_max) + '\n')
# para_line.append('fake_soft_label_min:' + str(fake_soft_label_min) + '\n')
# para_line.append('fake_soft_label_max:' + str(fake_soft_label_max) + '\n')
para_line.append('action:' + action + '\n')
if not os.path.exists(execute_name):
    os.makedirs(execute_name)
paraFile = open(os.path.join(execute_name, 'Parameters.txt'), 'w')
paraFile.writelines(para_line)
paraFile.close()

# Start to choose emotion for training
for emotion in ['valence']:
    # 0
    # Data load
    if emotion == 'valence':
        Source_Train_Y = np.load(save_path + '/' + source_dataset_name + '/Train_Y_valence.npy')
        Target_Train_Y = np.load(save_path + '/' + target_dataset_name + '/Train_Y_valence.npy')
    elif emotion == 'arousal':
        Source_Train_Y = np.load(save_path + '/' + source_dataset_name + '/Train_Y_arousal.npy')
        Target_Train_Y = np.load(save_path + '/' + target_dataset_name + '/Train_Y_arousal.npy')
    Source_Train_X = np.load(save_path + '/' + source_dataset_name +
                             '/Train_X' + '@' + str(output_sample_rate) + 'Hz_' + feature + '.npy')
    Target_Train_X = np.load(save_path + '/' + target_dataset_name +
                             '/Train_X' + '@' + str(output_sample_rate) + 'Hz_' + feature + '.npy')
    if KTF._SESSION:
        print('Reset session.')
        KTF.clear_session()
        KTF.set_session(get_session())

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # 1
    # Discriminator generated(layer_length: 6)
    #  Input: source or target feature extraction output tensor
    # Output: discriminator output tensor
    source_or_target_tensor = Input(shape=(encoded_size, ))
    discriminator_model = Model(inputs=source_or_target_tensor,
                                outputs=model_structure.domain_classifier(source_or_target_tensor))
    discriminator_model.compile(loss=discriminator_loss, optimizer=adam, metrics=['accuracy'])

    # 2
    # Regression classifier generated(layer_length: 2)
    # Input: target feature extraction output tensor
    # Output: Regression classifier output tensor
    target_tensor = Input(shape=(encoded_size, ))
    classifier_model = Model(inputs=target_tensor,
                             outputs=model_structure.regression_classifier(target_tensor))

    # 3
    # Source & Target feature extraction(layer_length: 33)
    # Input: Raw audio sized Input tensor
    # Output: feature extraction output tensor
    source_feature_tensor = Input(shape=(Source_Train_X.shape[1], Source_Train_X.shape[2], 1))
    source_feature_extractor = model_structure.compact_cnn_extractor(feature_tensor=source_feature_tensor,
                                                                     poolings=poolings)
    source_extractor = Model(inputs=source_feature_tensor, outputs=source_feature_extractor)

    target_feature_tensor = Input(shape=(Target_Train_X.shape[1], Target_Train_X.shape[2], 1))
    target_feature_extractor = model_structure.compact_cnn_extractor(feature_tensor=target_feature_tensor,
                                                                     poolings=poolings)
    target_extractor = Model(inputs=target_feature_tensor, outputs=target_feature_extractor)

    # 4
    # Combine Target(layer_length: 33) and discriminator(layer_length: 6)
    # Input: Raw audio sized Input tensor
    # Output: Discriminator output tensor(from target output)
    target_discriminator_model = Model(inputs=target_feature_tensor,
                                       outputs=discriminator_model(target_feature_extractor))
    target_discriminator_model.compile(loss=discriminator_loss, optimizer=adam, metrics=['accuracy'])
    print("target_discriminator_model summary:")
    target_discriminator_model.summary()

    # 5
    # Combine Target(layer_length: 33) and classifier(layer_length: 2)
    # Input: Raw audio sized Input tensor
    # Output: classifier output tensor(from target output)
    target_classifier_model = Model(inputs=target_feature_tensor,
                                    outputs=classifier_model(target_feature_extractor))
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
                                            'val_R2pr_' + format(max(data['val_R2_pearsonr'][50:]), '.5f') in f:
                        print(f)
                        model = load_model(os.path.join(root, f), custom_objects={'Melspectrogram': Melspectrogram,
                                                                                  'R2': metric.R2})
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

    # 7
    # Lock source extractor weights
    source_extractor.trainable = False
    # Lock classifier weights
    classifier_model.trainable = False

    # 8
    # Generator for source and target data
    source_data_generator = ADDA_funcs.data_generator(Source_Train_X, Source_Train_Y, batch_size)
    target_data_generator = ADDA_funcs.data_generator(Target_Train_X, Target_Train_Y, batch_size)

    # 9
    # training init
    total_training_steps = int(len(Target_Train_Y) / (batch_size * (k_g + k_d)))
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
    log_data = ADDA_funcs.log_dump(model_path=model_path, run_num=0, target_classifier_model=target_classifier_model,
                                   val_x=Target_Train_X, val_y=Target_Train_Y,
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
        progbar = generic_utils.Progbar(len(Target_Train_Y))
        for t in range(0, total_training_steps):
            # Train discriminator: taget = 0, source = 1, use discriminator model(To discriminate source and target)
            for i in range(k_d):
                sample_source_x, sample_source_y = next(source_data_generator)
                sample_target_x, sample_target_y = next(target_data_generator)
                source_y = np.ones(len(sample_source_y))
                target_y = np.zeros(len(sample_target_y))
                # target_y = np.array([(np.random.randint(4)) / 10] * len(sample_target_y))
                source_tensor_output = source_extractor.predict(sample_source_x)
                target_tensor_output = target_extractor.predict(sample_target_x)
                combine_source_target = np.concatenate((source_tensor_output, target_tensor_output), axis=0)
                combine_y = np.concatenate((source_y, target_y), axis=0)
                discriminator_model.trainable = True
                discriminator_model.compile(loss=discriminator_loss, optimizer=adam, metrics=['accuracy'])
                loss_dis = np.add(discriminator_model.train_on_batch(combine_source_target, combine_y), loss_dis)
            # Train taget feature extractor, use combined model: target label=1
            # Trick: inverted target label, to make target similar to source)
            for i in range(k_g):
                sample_target_x, sample_target_y = next(target_data_generator)
                target_y = np.ones(len(sample_target_y))
                # target_y = np.array([(7 + np.random.randint(6)) / 10] * len(sample_target_y))
                discriminator_model.trainable = False
                discriminator_model.compile(loss=discriminator_loss, optimizer=adam, metrics=['accuracy'])
                loss_fake = np.add(target_discriminator_model.train_on_batch(sample_target_x, target_y), loss_fake)
            progbar.add(batch_size * (k_g + k_d), values=[("loss_dis", loss_dis),
                                                          ("loss_fake", loss_fake)])
        loss_fake = loss_fake / (total_training_steps * k_g)
        loss_dis = loss_dis / (total_training_steps * k_d)
        log_data = ADDA_funcs.log_dump(model_path=model_path, run_num=0, target_classifier_model=target_classifier_model,
                                       val_x=Target_Train_X, val_y=Target_Train_Y,
                                       log_data=log_data, loss_fake=loss_fake, loss_dis=loss_dis,
                                       save_best_only=save_best_only, save_weights_only=save_weights_only)