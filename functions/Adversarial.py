from keras.engine.training import Model
from keras import backend as K
from functools import reduce

def _normalize_vector(x):
    z = K.sum(K.batch_flatten(K.square(x)), axis=1)
    while K.ndim(z) < K.ndim(x):
        z = K.expand_dims(z, axis=-1)
    return x / (K.sqrt(z) + K.epsilon())

def _kld(p, q):
    p = K.clip(p, K.epsilon(), 1)
    q = K.clip(q, K.epsilon(), 1)
    return K.sum(K.batch_flatten(p * K.log(p / q)), axis=1, keepdims=True)

def _to_list(x):
    if isinstance(x, list):
        return x
    return [x]


class ADVModel(Model):
    is_set_loss = False

    def setup_loss(self, eps = 0.01):
        self.eps = eps
        self.is_set_loss = True
        return self

    @property
    def losses(self):
        losses = super(self.__class__, self).losses
        if self.is_set_loss:
            losses += [self.adv_loss(self.eps)]
        return losses

    def adv_loss(self, eps):

        losses = [
            loss_weight*K.mean(loss_function(y_true,y_pred))
            for (y_true,y_pred,loss_weight,loss_function)
            in zip(_to_list(self.targets),
                   _to_list(self.outputs),
                   self.loss_weights,
                   self.loss_functions)
        ]

        loss = reduce(lambda t, x: t + x, losses, 0)
        r_adv = [eps*K.sign(K.stop_gradient(g)) for g in K.gradients(loss, self.inputs)]

        new_inputs = [x+r for (x, r) in zip(self.inputs, r_adv)]
        new_outputs = _to_list(self.call(new_inputs))
        new_loss = [
            loss_weight*K.mean(loss_function(y_true,y_pred))
            for (y_true,y_pred,loss_weight,loss_function)
            in zip(_to_list(self.targets),
                   new_outputs,
                   self.loss_weights,
                   self.loss_functions)
        ]

        loss = reduce(lambda t, x: t + x, new_loss, 0)
        return loss

class VATModel(Model):
    is_set_loss = False

    def setup_loss(self, eps=8, xi=1e-6, ip=1):
        self.eps = eps
        self.xi = xi
        self.ip = ip
        self.is_set_loss = True
        return self

    @property
    def losses(self):
        losses = super(self.__class__, self).losses
        if self.is_set_loss:
            losses += [self.adv_loss(self.eps, self.xi, self.ip)]
        return losses

    def adv_loss(self, eps, xi, ip):
        """
        :param eps: the epsilon (input variation parameter)
        :param ip: the number of iterations
        :param xi: the finite difference parameter
        """
        normal_outputs = [K.stop_gradient(x) for x in _to_list(self.outputs)]
        d_list = [K.random_normal(K.shape(x)) for x in self.inputs]

        for _ in range(ip):
            d_list = [xi * _normalize_vector(d) for d in d_list]
            new_inputs = [x + d for (x, d) in zip(self.inputs, d_list)]
            new_outputs = _to_list(self.call(new_inputs))
            klds = [K.mean(_kld(normal, new)) for normal, new in zip(normal_outputs, new_outputs)]
            kld = reduce(lambda t, x: t + x, klds, 0)
            d_list = [K.stop_gradient(d) for d in K.gradients(kld, d_list)]

        new_inputs = [x + eps * _normalize_vector(d) for (x, d) in zip(self.inputs, d_list)]
        y_perturbations = _to_list(self.call(new_inputs))
        klds = [K.mean(_kld(normal, new)) for normal, new in zip(normal_outputs, y_perturbations)]
        kld = reduce(lambda t, x: t + x, klds, 0)
        return kld

class JDVModel(Model):
    is_set_loss = False

    def setup_loss(self, eps=0.01):
        self.eps = eps
        self.is_set_loss = True
        return self

    @property
    def losses(self):
        losses = super(self.__class__, self).losses
        if self.is_set_loss:
            losses += [self.adv_loss(self.eps)]
        return losses

    def adv_loss(self, eps):
        outputs = [K.stop_gradient(x) for x in _to_list(self.outputs)]
        r_adv = [K.epsilon()*K.sign(K.random_normal(K.shape(x))) for x in self.inputs]

        # compute gradient of r_adv of negative kl-divergence
        new_inputs = [x + r for (x, r) in zip(self.inputs, r_adv)]
        new_outputs = _to_list(self.call(new_inputs))

        losses = [
            loss_weight * K.mean(loss_function(y_true, y_pred))
            for (y_true, y_pred, loss_weight, loss_function)
            in zip(outputs,
                   new_outputs,
                   self.loss_weights,
                   self.loss_functions)
        ]
        loss = reduce(lambda t, x: t + x, losses, 0)

        grads = [ K.stop_gradient(r) for r in K.gradients(loss, r_adv)]

        new_inputs = [x + eps * K.sign(g) for (x, g) in zip(self.inputs, grads)]
        new_outputs = _to_list(self.call(new_inputs))
        losses = [
            loss_weight * K.mean(loss_function(y_true, y_pred))
            for (y_true, y_pred, loss_weight, loss_function)
            in zip(outputs,
                   new_outputs,
                   self.loss_weights,
                   self.loss_functions)
        ]

        loss = reduce(lambda t, x: t + x, losses, 0)
        return loss