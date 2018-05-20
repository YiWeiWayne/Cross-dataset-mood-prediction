import json
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import math


def data_generator(X_train, y_train, batch_size):
    idx = 0
    total = len(X_train)
    while 1:
        p = np.random.permutation(len(X_train))  # shuffle each time
        X_train = X_train[p]
        y_train = y_train[p]
        for i in range(0, int(total / batch_size)):
            yield X_train[i * batch_size:(i + 1) * batch_size], y_train[i * batch_size:(i + 1) * batch_size]


def multi_data_generator(X_train, y_train, batch_size, features, data_num):
    while 1:
        p = np.random.permutation(data_num)  # shuffle each time
        for feature in features:
            X_train[feature] = X_train[feature][p]
        y_train = y_train[p]
        for i in range(0, int(data_num / batch_size)):
            yield X_train[features[0]][i * batch_size:(i + 1) * batch_size], X_train[features[1]][i * batch_size:(i + 1) * batch_size], X_train[features[2]][i * batch_size:(i + 1) * batch_size], y_train[i * batch_size:(i + 1) * batch_size]


def data_load_generator(save_path, dataset_name, emotion, output_sample_rate, feature, batch_size):
    # load data
    Train_Y = np.load(save_path + '/' + dataset_name + '/Train_Y_' + emotion + '.npy')
    Train_X = np.load(save_path + '/' + dataset_name +
                      '/Train_X@' + str(output_sample_rate) + 'Hz_' + feature + '.npy')
    total = len(Train_X)
    while 1:
        p = np.random.permutation(len(Train_X))  # shuffle each time
        X_train = Train_X[p]
        y_train = Train_Y[p]
        for i in range(0, int(total / batch_size)):
            yield X_train[i * batch_size:(i + 1) * batch_size], y_train[i * batch_size:(i + 1) * batch_size]


def log_dump(model_path, run_num, target_classifier_model, val_x, val_y, log_data, loss_fake, loss_dis,
             save_best_only, save_weights_only):
    file_name = model_path + '/log_' + str(run_num)
    val_y_pred = target_classifier_model.predict(val_x, batch_size=4, verbose=0)
    val_y_pred = val_y_pred[:, 0]
    with np.errstate(divide='ignore'):
        val_R2_pearsonr = np.square(pearsonr(val_y, val_y_pred)[0])
    if math.isnan(val_R2_pearsonr):
        val_R2_pearsonr = 0
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
    if save_best_only and len(log_data['val_R2_pearsonr']) > 2:
        print('val_R2_pearsonr: ' + str(val_R2_pearsonr))
        print('val_R2_pearsonr_log_max: ' + str(max(log_data['val_R2_pearsonr'][1:])))
        if val_R2_pearsonr >= max(log_data['val_R2_pearsonr'][1:]):
            print('improved!')
            model_save(model=target_classifier_model,
                       save_weights_only=save_weights_only,
                       filepath=model_path +
                                'loss_fake_' + format(loss_fake[0], '.5f') +
                                '_loss_dis_' + format(loss_dis[0], '.5f') +
                                '_R2pr_' + format(val_R2_pearsonr, '.5f'))
        else:
            print('No improved!')
    else:
        model_save(model=target_classifier_model,
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


def log_dump_reg(model_path, run_num, target_classifier_model, val_x, val_y, log_data, loss_reg, loss_fake, loss_dis,
                 save_best_only, save_weights_only):
    file_name = model_path + '/log_' + str(run_num)
    val_y_pred = target_classifier_model.predict(val_x, batch_size=4, verbose=0)
    val_y_pred = val_y_pred[:, 0]
    with np.errstate(divide='ignore'):
        val_R2_pearsonr = np.square(pearsonr(val_y, val_y_pred)[0])
    if math.isnan(val_R2_pearsonr):
        val_R2_pearsonr = 0
    log_data["train_loss_reg"].append(loss_reg[0])
    log_data["train_loss_fake"].append(loss_fake[0])
    log_data["train_loss_dis"].append(loss_dis[0])
    log_data["val_R2_pearsonr"].append(val_R2_pearsonr)
    with open(file_name + "_logs.json", "w") as fb:
        json.dump(log_data, fb)
    print("loss_reg: ", loss_reg)
    print("loss_fake: ", loss_fake)
    print("loss_dis: ", loss_dis)
    print("val_R2_pearsonr: ", val_R2_pearsonr)

    # summarize history for train_loss_fake
    plt.plot(log_data['train_loss_reg'])
    plt.title('model train_loss_reg')
    plt.ylabel('loss_reg')
    plt.xlabel('epoch')
    plt.savefig(file_name + "_train_loss_reg.png")
    plt.close()

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
    if save_best_only and len(log_data['val_R2_pearsonr']) > 2:
        print('val_R2_pearsonr: ' + str(val_R2_pearsonr))
        print('val_R2_pearsonr_log_max: ' + str(max(log_data['val_R2_pearsonr'][1:])))
        if val_R2_pearsonr >= max(log_data['val_R2_pearsonr'][1:]):
            print('improved!')
            model_save(model=target_classifier_model,
                       save_weights_only=save_weights_only,
                       filepath=model_path +
                                'loss_fake_' + format(loss_fake[0], '.5f') +
                                '_loss_dis_' + format(loss_dis[0], '.5f') +
                                '_R2pr_' + format(val_R2_pearsonr, '.5f'))
        else:
            print('No improved!')
    else:
        model_save(model=target_classifier_model,
                   save_weights_only=save_weights_only,
                   filepath=model_path +
                            'loss_fake_' + format(loss_fake[0], '.5f') +
                            '_loss_dis_' + format(loss_dis[0], '.5f') +
                            '_R2pr_' + format(val_R2_pearsonr, '.5f'))
    return log_data
