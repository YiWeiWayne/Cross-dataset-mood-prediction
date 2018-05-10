# -*- coding: utf-8 -*-
"""Pooling layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras.engine import Layer
from keras.engine import InputSpec
from keras.utils import conv_utils
from keras.legacy import interfaces


class Std2DLayer(Layer):

    def __init__(self, axis=2, **kwargs):
        super(Std2DLayer, self).__init__(**kwargs)
        self.axis = axis
        self.input_spec = InputSpec(ndim=4)

    def build(self, input_shape):
        super(Std2DLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        return K.std(inputs, axis=self.axis, keepdims=True)

    def compute_output_shape(self, input_shape):
        rows = input_shape[1]
        cols = input_shape[2]
        if self.axis == 1:
            rows = 1
        else:
            cols = 1
        return (input_shape[0], rows, cols, input_shape[3])
