from __future__ import absolute_import, division, print_function

import abc
import tensorflow as tf

__author__ = 'Otilia Stretcu'


class Model(object):
    """Superclass for models."""
    _metaclass__ = abc.ABCMeta

    def __init__(self, name):
        self.name = name

    @abc.abstractmethod
    def make_model(self, **kwargs):
        pass

    def __call__(self, **kwargs):
        return self.make_model(**kwargs)

    def get_callbacks(self):
        callbacks = []
        return callbacks


class CNN(Model):
    def __init__(self, inputs_shape, num_outputs, reg_weight=None, activation='relu', dropout=None,
                 name='CNN'):
        super().__init__(name=name)
        self.inputs_shape = inputs_shape
        self.num_outputs = num_outputs
        self.reg_weight = reg_weight
        self.activation = activation
        self.dropout = dropout

    def encoder(self, **kwargs):
        """Applied on inputs."""
        conv1 = tf.keras.layers.Conv2D(32,
                                       (3, 3),
                                       activation=self.activation,
                                       input_shape=self.inputs_shape,
                                       dtype=tf.float32,
                                       name='{}/Encoder/Conv2D_0'.format(self.name))
        layers = [
            conv1,
            tf.keras.layers.MaxPooling2D((2, 2))]
        if self.dropout:
            layers += [tf.keras.layers.Dropout(self.dropout)]
        layers += [
            tf.keras.layers.Conv2D(64,
                                   (3, 3),
                                   activation=self.activation,
                                   dtype=tf.float32,
                                   name='{}/Encoder/Conv2D_1'.format(self.name)),
            tf.keras.layers.MaxPooling2D((2, 2))]
        if self.dropout:
            layers += [tf.keras.layers.Dropout(self.dropout)]
        layers += [
            tf.keras.layers.Conv2D(64,
                                   (3, 3),
                                   activation=self.activation,
                                   dtype=tf.float32,
                                   name='{}/Encoder/Conv2D_2'.format(self.name))]
        if self.dropout:
            layers += [tf.keras.layers.Dropout(self.dropout)]
        layers += [tf.keras.layers.Flatten()]
        encoder = tf.keras.Sequential(layers, name='{}/Encoder/Sequential'.format(self.name))
        return encoder

    def predictor(self, num_outputs=None, **kwargs):
        """Applied on encoding."""
        num_outputs = self.num_outputs if num_outputs is None else num_outputs
        regularizer = tf.keras.regularizers.l2(self.reg_weight) \
            if self.reg_weight else None
        layers = [tf.keras.layers.Dense(
            num_outputs,
            kernel_regularizer=regularizer,
            name='{}/Predictor/Dense'.format(self.name))]
        predictor = tf.keras.Sequential(layers, name='{}/Predictor/Sequential'.format(self.name))
        return predictor
