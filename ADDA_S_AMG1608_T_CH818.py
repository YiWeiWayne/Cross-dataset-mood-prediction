import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from functions import model_structure, callback_wayne, metric
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.optimizers import SGD, Adam
import os, json
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime
from scipy.stats import pearsonr


# GPU speed limit
def get_session(gpu_fraction=0.6):
    # Assume that you have 6GB of GPU memory and want to allocate ~2GB
    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
KTF.set_session(get_session())


def data_generator(X_train, y_train, batch_size):
    idx = 0
    total = len(X_train)
    while 1:
        p = np.random.permutation(len(X_train))  # shuffle each time
        X_train = X_train[p]
        y_train = y_train[p]
        for i in range(total / batch_size):
            yield X_train[i * batch_size:(i + 1) * batch_size], y_train[i * batch_size:(i + 1) * batch_size]


def log_dump(model_path, CV, target_classifier_model, val_x, val_y, log_data, loss_fake, loss_dis,
             target_discriminator_model, save_best_only, save_weights_only):
    file_name = model_path + '/log_CV' + str(CV)
    val_y_pred = target_classifier_model.predict(val_x, verbose=0)
    val_y_pred = val_y_pred[:, 0]
    val_R2_pearsonr = np.square(pearsonr(val_y, val_y_pred)[0])
    log_data["train_loss_fake"].append(loss_fake[0])
    log_data["train_loss_dis"].append(loss_dis[0])
    log_data["val_R2_pearsonr"].append(val_R2_pearsonr)
    with open(file_name + "_logs.json", "w") as fb:
        json.dump(log_data, fb)
    print("loss_fake: ", loss_fake)
    print("loss_dis: ", loss_dis)
    print("val_R2_pearsonr: ", val_R2_pearsonr)

    # summarize history for train_loss_fake
    plt.plot(log_data['train_loss_fake'])
    plt.title('model train_loss_fake')
    plt.ylabel('loss_fake')
    plt.xlabel('epoch')
    plt.savefig(file_name + "_train_loss_fake.png")
    plt.close()

    # summarize history for train_loss_dis
    plt.plot(log_data['train_loss_dis'])
    plt.title('model train_loss_dis')
    plt.ylabel('loss_dis')
    plt.xlabel('epoch')
    plt.savefig(file_name + "_train_loss_dis.png")
    plt.close()

    # summarize history for R2_pearsonr
    plt.plot(log_data['val_R2_pearsonr'])
    plt.title('model pearson R square')
    plt.ylabel('R2_pearsonr')
    plt.xlabel('epoch')
    plt.savefig(file_name + "_R2_pearsonr.png")
    plt.close()

    # save model
    if save_best_only:
        print('val_R2_pearsonr: ' + str(val_R2_pearsonr))
        print('val_R2_pearsonr_log_max: ' + str(max(log_data['val_R2_pearsonr'])))
        if val_R2_pearsonr >= max(log_data['val_R2_pearsonr']):
            print('yes!')
            model_save(model=target_discriminator_model,
                       save_weights_only=save_weights_only,
                       filepath=model_path +
                                'loss_fake_' + format(loss_fake[0], '.5f') +
                                '_loss_dis_' + format(loss_dis[0], '.5f') +
                                '_R2pr_' + format(val_R2_pearsonr, '.5f'))
        else:
            print('no!')
    else:
        model_save(model=target_discriminator_model,
                   save_weights_only=save_weights_only,
                   filepath=model_path +
                            'loss_fake_' + format(loss_fake[0], '.5f') +
                            '_loss_dis_' + format(loss_dis[0], '.5f') +
                            '_R2pr_' + format(val_R2_pearsonr, '.5f'))
    return log_data


def model_save(model, filepath, save_weights_only):
    if save_weights_only:
        model.save_weights(filepath + '.h5', overwrite=True)
    else:
        model.save(filepath + '.h5', overwrite=True)


# Parameters
algorithm = 'ADDA'
source_dataset_name = 'AMG_1608'
target_dataset_name = 'CH_818'
save_path = '/data/Wayne'
source_execute_name = save_path + '/' + source_dataset_name + '_1522208082.13'
encoded_size = 32
batch_size = 16
epochs = 500
k_d = 1
k_g = 1
load_weights_source_feature_extractor = True
load_weights_source_classifier = True
load_weights_target_feature_extractor = True
save_best_only = True
save_weights_only = True
now = datetime.datetime.now()
localtime = str(now.year) + str(now.month).zfill(2) + str(now.day).zfill(2) + '.' +\
            str(now.hour).zfill(2) + str(now.minute).zfill(2) + '.' + str(now.second).zfill(2)
execute_name = save_path + '/' + algorithm + '_S_' + source_dataset_name + \
               '_T_' + target_dataset_name + '_' + localtime

# Parameters
para_line = []
para_line.append('algorithm:' + algorithm + '\n')
para_line.append('save_path:' + save_path + '\n')
para_line.append('execute_name:' + execute_name + '\n')
para_line.append('source_dataset_name:' + source_dataset_name + '\n')
para_line.append('target_dataset_name:' + target_dataset_name + '\n')
para_line.append('source_execute_name:' + source_execute_name + '\n')
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
if not os.path.exists(execute_name):
    os.makedirs(execute_name)
paraFile = open(os.path.join(execute_name, 'Parameters.txt'), 'w')
paraFile.writelines(para_line)
paraFile.close()

for emotion in ['valence', 'arousal']:
    # 0
    # Data load
    if emotion == 'valence':
        Source_Train_Y = np.load(save_path + '/' + source_dataset_name + '/Train_Y_valence.npy')
        Target_Train_Y = np.load(save_path + '/' + target_dataset_name + '/Train_Y_valence.npy')
    elif emotion == 'arousal':
        Source_Train_Y = np.load(save_path + '/' + source_dataset_name + '/Train_Y_arousal.npy')
        Target_Train_Y = np.load(save_path + '/' + target_dataset_name + '/Train_Y_arousal.npy')
    Source_Train_X = np.load(save_path + '/' + source_dataset_name + '/Train_X.npy')
    Target_Train_X = np.load(save_path + '/' + target_dataset_name + '/Train_X.npy')
    for CV in range(0, 10):
        if KTF._SESSION:
            print('Reset session.')
            KTF.clear_session()
            KTF.set_session(get_session())
        # 1
        # Discriminator generated(layer_length: 6)
        # Input: source or target feature extraction output tensor
        # Output: discriminator output tensor
        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        source_or_target_tensor = Input(shape=(encoded_size, ))
        discriminator_model = Model(inputs=source_or_target_tensor,
                                    outputs=model_structure.domain_classifier(source_or_target_tensor))
        print("discriminator model summary")
        discriminator_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        discriminator_model.summary()

        # 1'
        # Regression classifier generated(layer_length: 2)
        # Input: target feature extraction output tensor
        # Output: Regression classifier output tensor
        target_tensor = Input(shape=(encoded_size, ))
        classifier_model = Model(inputs=target_tensor,
                                 outputs=model_structure.regression_classifier(target_tensor))
        print("classifier_model summary")
        # classifier_model.compile(optimizer=adam, loss='mean_squared_error', metrics=[metric.R2])
        # classifier_model.summary()

        # 2
        # Source & Target feature extraction(layer_length: 33)
        # Input: Raw audio sized Input tensor
        # Output: feature extraction output tensor

        # Combine Target(layer_length: 33) and discriminator(layer_length: 6)
        # Input: Raw audio sized Input tensor
        # Output: Discriminator output tensor(from target output)

        # Combine Target(layer_length: 33) and classifier(layer_length: 2)
        # Input: Raw audio sized Input tensor
        # Output: classifier output tensor(from target output)

        # Load source clssifier weights and set to non-trainable
        _, source_extractor, source_audio_input, source_encoded_audio = model_structure.functional_compact_cnn(
            model_type="source_extractor")
        _, target_extractor, target_audio_input, target_encoded_audio = model_structure.functional_compact_cnn(
            model_type="target_extractor")
        source_extractor.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        target_extractor.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        discriminator_model.trainable = False
        # Combine target model and discriminator
        target_discriminator_model = Model(inputs=target_audio_input, outputs=discriminator_model(target_encoded_audio))
        target_discriminator_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        # print("target_discriminator_model summary:")
        # target_discriminator_model.summary()

        # Combine target model and classifier
        target_classifier_model = Model(inputs=target_audio_input, outputs=classifier_model(target_encoded_audio))
        target_classifier_model.compile(loss='mean_squared_error', optimizer=adam, metrics=[metric.R2])
        # print("target_classifier_model summary:")
        # target_classifier_model.summary()
        model = model_structure.compact_cnn()
        if os.path.exists(source_execute_name + '/' + emotion + '/CV' + str(CV) + '/log_CV_' + str(CV).zfill(2) + '_logs.json'):
            with open(source_execute_name + '/' + emotion + '/CV' + str(CV) + '/log_CV_' + str(CV).zfill(2) + '_logs.json', "r") as fb:
                data = json.load(fb)
                for root, subdirs, files in os.walk(source_execute_name + '/' + emotion + '/CV' + str(CV)):
                    for f in files:
                        if os.path.splitext(f)[1] == '.h5' and 'R2pr_' + format(max(data['val_R2_pearsonr']), '.5f') in f:
                            print(f)
                            model.load_weights(os.path.join(root, f))
                            if load_weights_source_feature_extractor:
                                source_extractor.set_weights(model.get_weights()[:-2])
                            # target_discriminator_model's weights will be set in the same time.
                            if load_weights_target_feature_extractor:
                                target_extractor.set_weights(model.get_weights()[:-2])
                            if load_weights_source_classifier:
                                classifier_model.set_weights(model.get_weights()[-2:])
        # Lock source extractor weights
        for layer in source_extractor.layers:
            layer.trainable = False
        print('Source extractor non-trainable summary:')
        source_extractor.summary()
        # Lock classifier weights
        for layer in classifier_model.layers:
            layer.trainable = False
        print('classifier_model non-trainable summary:')
        classifier_model.summary()

        # Pre-train combined model(SKIP, load source weights)

        # 6
        # Generator for source and target data
        # begin to train alternatively
        # Train taget feature extractor, use combined model: target label=1 (To make target similar to source)
        # Train discriminator: taget = 0, source = 1, use discriminator model(To discriminate source and target)
        # Trick here: inverted target labels
        # Goal: Make discriminator accuracy low.
        # Train generator and discriminator
        total_training_steps = int(len(Target_Train_Y) / (batch_size * k_g))
        #     init
        source_data_generator = data_generator(Source_Train_X, Source_Train_Y, batch_size)
        target_data_generator = data_generator(Target_Train_X, Target_Train_Y, batch_size)
        model_path = execute_name + '/' + emotion + '/CV' + str(CV) + '/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        loss_fake = np.zeros(shape=len(target_discriminator_model.metrics_names))
        loss_dis = np.zeros(shape=len(discriminator_model.metrics_names))
        log_data = {
                    "train_loss_fake": [],
                    "train_loss_dis": [],
                    "val_R2_pearsonr": [],
                    }
        log_data = log_dump(model_path=model_path, CV=CV, target_classifier_model=target_classifier_model,
                            val_x=Target_Train_X, val_y=Target_Train_Y,
                            log_data=log_data, loss_fake=loss_fake, loss_dis=loss_dis,
                            target_discriminator_model=target_discriminator_model,
                            save_best_only=save_best_only, save_weights_only=save_weights_only)
        # dump log
        # init, compute, set
        for epoch in range(0, epochs):
            print('Epoch: ' + str(epoch).zfill(4) + '/' + str(epochs).zfill(4))
            # Generator for source and target data
            loss_fake = np.zeros(shape=len(discriminator_model.metrics_names))
            loss_dis = np.zeros(shape=len(discriminator_model.metrics_names))
            for t in range(0, total_training_steps):
                for i in range(k_d):
                    sample_source_x, sample_source_y = next(source_data_generator)
                    sample_target_x, sample_target_y = next(target_data_generator)
                    # source_y = to_categorical(np.ones(len(sample_source_y)), num_classes=2)
                    # target_y = to_categorical(np.zeros(len(sample_target_y)), num_classes=2)
                    source_y = np.ones(len(sample_source_y))
                    target_y = np.zeros(len(sample_target_y))
                    source_tensor_output = source_extractor.predict(sample_source_x)
                    target_tensor_output = target_extractor.predict(sample_target_x)
                    combine_source_target = np.concatenate((source_tensor_output, target_tensor_output), axis=0)
                    combine_y = np.concatenate((source_y, target_y), axis=0)
                    discriminator_model.trainable = True
                    loss_dis = np.add(discriminator_model.train_on_batch(combine_source_target, combine_y), loss_dis)
                for i in range(k_g):
                    sample_target_x, sample_target_y = next(target_data_generator)
                    # target_y = to_categorical(np.ones(len(sample_target_y)), num_classes=2)
                    target_y = np.ones(len(sample_target_y))
                    # sample_target_x2, sample_target_y = next(target_data_generator)
                    # target_y2 = to_categorical(np.ones(len(sample_target_y)), num_classes=2)
                    # combine_x = np.concatenate((sample_target_x, sample_target_x2), axis=0)
                    # combine_y = np.concatenate((target_y, target_y2), axis=0)
                    discriminator_model.trainable = False
                    loss_fake = np.add(target_discriminator_model.train_on_batch(sample_target_x, target_y), loss_fake)
            loss_fake = loss_fake / (total_training_steps * k_g)
            loss_dis = loss_dis / (total_training_steps * k_d)
            log_data = log_dump(model_path=model_path, CV=CV, target_classifier_model=target_classifier_model,
                                val_x=Target_Train_X, val_y=Target_Train_Y,
                                log_data=log_data, loss_fake=loss_fake, loss_dis=loss_dis,
                                target_discriminator_model=target_discriminator_model,
                                save_best_only=save_best_only, save_weights_only=save_weights_only)
