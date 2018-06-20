from keras.callbacks import Callback
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import json
import numpy as np
import warnings
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr


class LossR2Logger_ModelCheckPoint(Callback):
    def __init__(self, train_data, val_data, file_name = "train", run_num=1,
                 filepath='model_tmp.h5', monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1
                 ):
        # Logger
        self.file_name = file_name + '_' + str(run_num)
        self.train_data = train_data
        self.val_data = val_data
        # ModelCheckPoint
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_train_begin(self, logs={}):
        self.log_data = {
            "train_loss": [],
            "train_R2_pearsonr": [],
            "train_pearsonr": [],
            "val_loss": [],
            "val_R2_pearsonr": [],
            "val_pearsonr": [],
            }

    def on_epoch_end(self, epoch, logs={}):
        train_x, train_y = self.train_data
        val_x, val_y = self.val_data
        train_y_pred = self.model.predict(train_x, batch_size=4, verbose=0)
        val_y_pred = self.model.predict(val_x, batch_size=4, verbose=0)
        train_y_pred = train_y_pred[:, 0]
        val_y_pred = val_y_pred[:, 0]
        # print('train_y_pred: ' + str(train_y_pred))
        # print('train_y: ' + str(train_y))
        # Statistics computation
        train_R2_pearsonr = np.square(pearsonr(train_y, train_y_pred)[0])
        val_R2_pearsonr = np.square(pearsonr(val_y, val_y_pred)[0])
        train_pearsonr = pearsonr(train_y, train_y_pred)[0]
        val_pearsonr = pearsonr(val_y, val_y_pred)[0]
        # train_R2 = r2_score(train_y, train_y_pred)
        # val_R2 = r2_score(val_y, val_y_pred)
        # Statistics print
        print('\nTraining pearsonr: {}, R2_pearsonr: {}\n'.format(train_pearsonr, train_R2_pearsonr))
        print('Validation pearsonr: {}, R2_pearsonr: {}\n'.format(val_pearsonr, val_R2_pearsonr))
        self.log_data["train_loss"].append(logs.get('loss'))
        self.log_data["train_R2_pearsonr"].append(train_R2_pearsonr)
        self.log_data["train_pearsonr"].append(train_pearsonr)
        self.log_data["val_loss"].append(logs.get('val_loss'))
        self.log_data["val_R2_pearsonr"].append(val_R2_pearsonr)
        self.log_data["val_pearsonr"].append(val_pearsonr)
        with open(self.file_name + "_logs.json", "w") as fb:
            json.dump(self.log_data, fb)

        # summarize history for loss
        plt.plot(self.log_data['train_loss'])
        plt.plot(self.log_data['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(self.file_name + "_loss.png")
        plt.close()

        # summarize history for pearsonr
        plt.plot(self.log_data['train_pearsonr'])
        plt.plot(self.log_data['val_pearsonr'])
        plt.title('model pearson')
        plt.ylabel('pearsonr')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(self.file_name + "_pearsonr.png")
        plt.close()

        # summarize history for R2_pearsonr
        plt.plot(self.log_data['train_R2_pearsonr'])
        plt.plot(self.log_data['val_R2_pearsonr'])
        plt.title('model pearson R square')
        plt.ylabel('R2_pearsonr')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(self.file_name + "_R2_pearsonr.png")
        plt.close()

        # ModelCheckPoint
        logs = logs or {}
        filepath = self.filepath + 'train_R2pr_' + format(train_R2_pearsonr, '.5f') + \
                   '_val_R2pr_' + format(val_R2_pearsonr, '.5f') + '.h5'
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = self.log_data[self.monitor][-1]
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


class Loss_Logger(Callback):
    def __init__(self, file_name = "train", run_num = 1):
        self.file_name = file_name + '_' + str(run_num).zfill(2)

    def on_train_begin(self, logs={}):
        self.log_data = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            }

    def on_epoch_end(self, epoch, logs={}):
        self.log_data["train_loss"].append(logs.get('loss'))
        self.log_data["train_acc"].append(logs.get('acc'))
        self.log_data["val_loss"].append(logs.get('val_loss'))
        self.log_data["val_acc"].append(logs.get('val_acc'))
        with open(self.file_name + "_logs.json", "w") as fb:
            json.dump(self.log_data, fb)

        # summarize history for loss
        plt.plot(self.log_data['train_loss'])
        plt.plot(self.log_data['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(self.file_name + "_loss.png")
        plt.close()

        # summarize history for acc
        plt.plot(self.log_data['train_acc'])
        plt.plot(self.log_data['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(self.file_name + "_acc.png")
        plt.close()


class LossLogger(Callback):
    def __init__(self, file_name = "train", run_num = 1):
        self.file_name = file_name+'_'+str(run_num).zfill(2)

    def on_train_begin(self, logs={}):
        self.log_data = {
            "train_loss": [],
            "train_acc": [],
            "train_ok_acc": [],
            "train_ng_acc": [],
            "train_new_metric": [],
            "val_loss": [],
            "val_acc": [],
            "val_ok_acc": [],
            "val_ng_acc": [],
            "val_new_metric": [],
        }

    def on_epoch_end(self, epoch, logs={}):
        self.log_data["train_loss"].append(logs.get('loss'))
        self.log_data["train_acc"].append(logs.get('acc'))
        self.log_data["train_ok_acc"].append(logs.get('ok_acc'))
        self.log_data["train_ng_acc"].append(logs.get('ng_acc'))
        self.log_data["train_new_metric"].append(logs.get('new_metric'))
        self.log_data["val_loss"].append(logs.get('val_loss'))
        self.log_data["val_acc"].append(logs.get('val_acc'))
        self.log_data["val_ok_acc"].append(logs.get('val_ok_acc'))
        self.log_data["val_ng_acc"].append(logs.get('val_ng_acc'))
        self.log_data["val_new_metric"].append(logs.get('val_new_metric'))

        with open(self.file_name+"_logs.json", "w") as fb:
            json.dump(self.log_data, fb)

        # summarize history for accuracy
        plt.plot(self.log_data['train_acc'])
        plt.plot(self.log_data['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(self.file_name+"_acc.png")
        plt.close()

        # summarize history for loss
        plt.plot(self.log_data['train_loss'])
        plt.plot(self.log_data['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(self.file_name+"_loss.png")
        plt.close()

        # summarize history for ok accuracy
        plt.plot(self.log_data['train_ok_acc'])
        plt.plot(self.log_data['val_ok_acc'])
        plt.title('model ok accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(self.file_name + "_ok_acc.png")
        plt.close()

        # summarize history for ng accuracy
        plt.plot(self.log_data['train_ng_acc'])
        plt.plot(self.log_data['val_ng_acc'])
        plt.title('model ng accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(self.file_name + "_ng_acc.png")
        plt.close()

        # summarize history for new_metric accuracy
        plt.plot(self.log_data['train_new_metric'])
        plt.plot(self.log_data['val_new_metric'])
        plt.title('model new_metric accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(self.file_name + "_new_metric.png")
        plt.close()


class ModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)