from keras import backend as K


def single_class_accuracy(interesting_class_id):
    def single_acc(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        # Replace class_id_preds with class_id_true for recall here
        accuracy_mask = K.cast(K.equal(class_id_true, interesting_class_id), 'int32')
        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
        class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
        return class_acc
    return single_acc


def new_metric_acc(ng_thr):
    def new_metric(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        # OK_acc
        ok_accuracy_mask = K.cast(K.equal(class_id_true, 0), 'int32')
        ok_class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * ok_accuracy_mask
        ok_class_acc = K.sum(ok_class_acc_tensor) / K.maximum(K.sum(ok_accuracy_mask), 1)
        # NG_acc
        ng_accuracy_mask = K.cast(K.equal(class_id_true, 1), 'int32')
        ng_class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * ng_accuracy_mask
        ng_class_acc = K.sum(ng_class_acc_tensor) / K.maximum(K.sum(ng_accuracy_mask), 1)
        return ok_class_acc * K.minimum(ng_class_acc+ng_thr, 1)
    return new_metric


def ok_acc(y_true, y_pred):
    interesting_class_id=0
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    # Replace class_id_preds with class_id_true for recall here
    accuracy_mask = K.cast(K.equal(class_id_true, interesting_class_id), 'int32')
    class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
    class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
    return class_acc


def ng_acc(y_true, y_pred):
    interesting_class_id = 1
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    # Replace class_id_preds with class_id_true for recall here
    accuracy_mask = K.cast(K.equal(class_id_true, interesting_class_id), 'int32')
    class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
    class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
    return class_acc


def R2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred), axis=-1)
    SS_tot = K.sum(K.square(y_true - K.mean(y_true, axis=-1)), axis=-1)
    return K.ones_like(SS_tot) - SS_res/(SS_tot + K.epsilon())


def R2_regression(y_true, y_pred):
    SS_reg = K.sum(K.square(y_pred - K.mean(y_true)))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (SS_reg/(SS_tot + K.epsilon()))

