import json
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import math
from sklearn.metrics import mean_squared_error


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


def log_dump(model_path, run_num, target_regressor_model, val_x, val_y, log_data, loss_fake, loss_dis,
             save_best_only, save_weights_only):
    file_name = model_path + '/log_' + str(run_num)
    val_y_pred = target_regressor_model.predict(val_x, batch_size=4, verbose=0)
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
            model_save(model=target_regressor_model,
                       save_weights_only=save_weights_only,
                       filepath=model_path +
                                'loss_fake_' + format(loss_fake[0], '.5f') +
                                '_loss_dis_' + format(loss_dis[0], '.5f') +
                                '_R2pr_' + format(val_R2_pearsonr, '.5f'))
        else:
            print('No improved!')
    else:
        model_save(model=target_regressor_model,
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


def log_dump_reg(model_path, run_num, target_regressor_model, val_x, val_y, log_data, loss_reg, loss_fake, loss_dis,
                 save_best_only, save_weights_only):
    file_name = model_path + '/log_' + str(run_num)
    val_y_pred = target_regressor_model.predict(val_x, batch_size=4, verbose=0)
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
            model_save(model=target_regressor_model,
                       save_weights_only=save_weights_only,
                       filepath=model_path +
                                'loss_fake_' + format(loss_fake[0], '.5f') +
                                '_loss_dis_' + format(loss_dis[0], '.5f') +
                                '_R2pr_' + format(val_R2_pearsonr, '.5f'))
        else:
            print('No improved!')
    else:
        model_save(model=target_regressor_model,
                   save_weights_only=save_weights_only,
                   filepath=model_path +
                            'loss_fake_' + format(loss_fake[0], '.5f') +
                            '_loss_dis_' + format(loss_dis[0], '.5f') +
                            '_R2pr_' + format(val_R2_pearsonr, '.5f'))
    return log_data


def log_dump_all(model_path, run_num, source_regressor_model, train_x, train_y,
                 target_regressor_model, val_x, val_y, log_data, train_loss, reg_loss, val_loss, loss_fake, loss_dis,
                 save_best_only, save_weights_only, save_source_model, source_epoch_th, target_epoch_th):
    file_name = model_path + '/log_' + str(run_num)
    train_y_pred = source_regressor_model.predict(train_x, batch_size=4, verbose=0)
    train_y_pred = train_y_pred[:, 0]
    source_val_y_pred = source_regressor_model.predict(val_x, batch_size=4, verbose=0)
    source_val_y_pred = source_val_y_pred[:, 0]
    val_y_pred = target_regressor_model.predict(val_x, batch_size=4, verbose=0)
    val_y_pred = val_y_pred[:, 0]
    target_train_y_pred = target_regressor_model.predict(train_x, batch_size=4, verbose=0)
    target_train_y_pred = target_train_y_pred[:, 0]
    train_MSE = mean_squared_error(train_y, train_y_pred)
    source_val_MSE = mean_squared_error(val_y, source_val_y_pred)
    val_MSE = mean_squared_error(val_y, val_y_pred)
    target_train_MSE = mean_squared_error(train_y, target_train_y_pred)
    with np.errstate(divide='ignore'):
        train_R2_pearsonr = np.square(pearsonr(train_y, train_y_pred)[0])
        source_val_R2_pearsonr = np.square(pearsonr(val_y, source_val_y_pred)[0])
        val_R2_pearsonr = np.square(pearsonr(val_y, val_y_pred)[0])
        target_train_R2_pearsonr = np.square(pearsonr(train_y, target_train_y_pred)[0])
        train_pearsonr = pearsonr(train_y, train_y_pred)[0]
        source_val_pearsonr = pearsonr(val_y, source_val_y_pred)[0]
        val_pearsonr = pearsonr(val_y, val_y_pred)[0]
        target_train_pearsonr = pearsonr(train_y, target_train_y_pred)[0]
    if math.isnan(source_val_R2_pearsonr):
        source_val_R2_pearsonr = 0
        source_val_pearsonr = 0
    if math.isnan(val_R2_pearsonr):
        val_R2_pearsonr = 0
        val_pearsonr = 0
    if math.isnan(train_R2_pearsonr):
        train_R2_pearsonr = 0
        train_pearsonr = 0
    if math.isnan(target_train_R2_pearsonr):
        target_train_R2_pearsonr = 0
        target_train_pearsonr = 0
    log_data["train_MSE"].append(train_MSE)
    log_data["source_val_MSE"].append(source_val_MSE)
    log_data["val_MSE"].append(val_MSE)
    log_data["target_train_MSE"].append(target_train_MSE)
    log_data["train_loss_fake"].append(loss_fake[0])
    log_data["train_loss_dis"].append(loss_dis[0])
    log_data["train_R2_pearsonr"].append(train_R2_pearsonr)
    log_data["source_val_R2_pearsonr"].append(source_val_R2_pearsonr)
    log_data["val_R2_pearsonr"].append(val_R2_pearsonr)
    log_data["target_train_R2_pearsonr"].append(target_train_R2_pearsonr)
    log_data["train_pearsonr"].append(train_pearsonr)
    log_data["source_val_pearsonr"].append(source_val_pearsonr)
    log_data["val_pearsonr"].append(val_pearsonr)
    log_data["target_train_pearsonr"].append(target_train_pearsonr)
    with open(file_name + "_logs.json", "w") as fb:
        json.dump(log_data, fb)
    print("train_MSE: ", train_loss)
    print("reg_MSE: ", reg_loss)
    print("val_MSE: ", val_loss)
    print("loss_fake: ", loss_fake)
    print("loss_dis: ", loss_dis)
    print("train_pearsonr: ", train_pearsonr)
    print("train_R2_pearsonr: ", train_R2_pearsonr)
    print("source_val_pearsonr: ", source_val_pearsonr)
    print("source_val_R2_pearsonr: ", source_val_R2_pearsonr)
    print("val_pearsonr: ", val_pearsonr)
    print("val_R2_pearsonr: ", val_R2_pearsonr)
    print("target_train_pearsonr: ", target_train_pearsonr)
    print("target_train_R2_pearsonr: ", target_train_R2_pearsonr)

    # summarize history for MSE
    plt.plot(log_data['train_MSE'])
    plt.plot(log_data['source_val_MSE'])
    plt.plot(log_data['val_MSE'])
    plt.plot(log_data['target_train_MSE'])
    plt.title('model MSE')
    plt.ylabel('MSE')
    plt.xlabel('epoch')
    plt.legend(['train', 'source_val', 'val', 'target_train'], loc='upper left')
    plt.savefig(file_name + "_MSE.png")
    plt.close()

    # summarize history for train_loss_dis&fake
    plt.plot(log_data['train_loss_fake'])
    plt.plot(log_data['train_loss_dis'])
    plt.title('model train_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['fake', 'dis'], loc='upper left')
    plt.savefig(file_name + "_train_loss.png")
    plt.close()

    # summarize history for R2_pearsonr
    plt.plot(log_data['train_R2_pearsonr'])
    plt.plot(log_data['source_val_R2_pearsonr'])
    plt.plot(log_data['val_R2_pearsonr'])
    plt.plot(log_data['target_train_R2_pearsonr'])
    plt.title('model pearson R square')
    plt.ylabel('R2_pearsonr')
    plt.xlabel('epoch')
    plt.legend(['train', 'source_val', 'val', 'target_train'], loc='upper left')
    plt.savefig(file_name + "_R2_pearsonr.png")
    plt.close()

    # summarize history for pearsonr
    plt.plot(log_data['train_pearsonr'])
    plt.plot(log_data['source_val_pearsonr'])
    plt.plot(log_data['val_pearsonr'])
    plt.plot(log_data['target_train_pearsonr'])
    plt.title('model pearson')
    plt.ylabel('pearsonr')
    plt.xlabel('epoch')
    plt.legend(['train', 'source_val', 'val', 'target_train'], loc='upper left')
    plt.savefig(file_name + "_pearsonr.png")
    plt.close()

    # save target model
    if save_best_only and len(log_data['val_pearsonr']) > target_epoch_th:
        print('val_pearsonr: ' + str(val_pearsonr))
        print('val_pearsonr_log_max: ' + str(max(log_data['val_pearsonr'][target_epoch_th-1:])))
        if val_pearsonr >= max(log_data['val_pearsonr'][target_epoch_th-1:]) or \
                        val_R2_pearsonr >= max(log_data['val_R2_pearsonr'][target_epoch_th-1:]):
            print('validation improved!')
            model_save(model=target_regressor_model,
                       save_weights_only=save_weights_only,
                       filepath=model_path + 'target_'
                                'train_R2pr_' + format(train_R2_pearsonr, '.5f') +
                                '_val_R2pr_' + format(val_R2_pearsonr, '.5f') + '.h5')
        else:
            print('No improved!')
    else:
        model_save(model=target_regressor_model,
                   save_weights_only=save_weights_only,
                   filepath=model_path + 'target_'
                            'train_R2pr_' + format(train_R2_pearsonr, '.5f') +
                            '_val_R2pr_' + format(val_R2_pearsonr, '.5f') + '.h5')
    # save source model
    if save_source_model:
        if save_best_only and len(log_data['train_pearsonr']) > source_epoch_th:
            print('train_pearsonr: ' + str(train_pearsonr))
            print('train_pearsonr_log_max: ' + str(max(log_data['train_pearsonr'][source_epoch_th-1:])))
            if train_pearsonr >= max(log_data['train_pearsonr'][source_epoch_th-1:]) or \
                            train_R2_pearsonr >= max(log_data['train_R2_pearsonr'][source_epoch_th - 1:]):
                print('training improved!')
                model_save(model=source_regressor_model,
                           save_weights_only=save_weights_only,
                           filepath=model_path + 'source_'
                                    'train_R2pr_' + format(train_R2_pearsonr, '.5f') +
                                    '_val_R2pr_' + format(source_val_R2_pearsonr, '.5f') + '.h5')
            else:
                print('No improved!')
        else:
            model_save(model=source_regressor_model,
                       save_weights_only=save_weights_only,
                       filepath=model_path + 'source_'
                                'train_R2pr_' + format(train_R2_pearsonr, '.5f') +
                                '_val_R2pr_' + format(source_val_R2_pearsonr, '.5f') + '.h5')
    return log_data
