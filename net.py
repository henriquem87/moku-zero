###############################################################################
#
# Copyright (c) 2018, Henrique Morimitsu,
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# #############################################################################

import tensorflow as tf
import numpy as np


class ResNet(tf.keras.Model):
    """ An implementation of a residual network [1] with an architecture based
    on the AlphaGo Zero network [2].

    Args:
      num_outputs: int: number of policy outputs to predict
      num_res_layers: int: number of residual blocks in the CNN.
      num_channels: int: number of channels in the CNN layers.

    [1] He, Kaiming, et al. "Deep residual learning for image recognition."
        CVPR. 2016.
    [2] Silver, David, et al. "Mastering the game of go without human
        knowledge." Nature 550.7676 (2017): 354.
    """

    def __init__(self, num_outputs, num_res_layers, num_channels):
        super(ResNet, self).__init__()

        self.num_res_layers = num_res_layers

        self.relu = tf.keras.layers.Activation('relu')
        self.tanh = tf.keras.layers.Activation('tanh')
        self.l2_reg = tf.keras.regularizers.l2(0.0001)

        self.conv1 = tf.keras.layers.Conv2D(
            num_channels, 3, padding='same', data_format='channels_last',
            kernel_regularizer=self.l2_reg)
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.9999)

        for i in range(num_res_layers):
            setattr(self, 'res%d_conv1' % (i+1),
                    tf.keras.layers.Conv2D(
                        num_channels, 3, padding='same',
                        data_format='channels_last',
                        kernel_regularizer=self.l2_reg))
            setattr(self, 'res%d_bn1' % (i+1),
                    tf.keras.layers.BatchNormalization(momentum=0.9999))
            setattr(self, 'res%d_conv2' % (i+1),
                    tf.keras.layers.Conv2D(
                        num_channels, 3, padding='same',
                        data_format='channels_last',
                        kernel_regularizer=self.l2_reg))
            setattr(self, 'res%d_bn2' % (i+1),
                    tf.keras.layers.BatchNormalization(momentum=0.9999))

        self.policy_conv = tf.keras.layers.Conv2D(
            2, 1, data_format='channels_last', kernel_regularizer=self.l2_reg)
        self.policy_bn = tf.keras.layers.BatchNormalization(momentum=0.9999)
        self.policy_pred = tf.keras.layers.Dense(
            num_outputs,
            bias_initializer=tf.keras.initializers.Constant(1.0/num_outputs))

        self.value_conv = tf.keras.layers.Conv2D(
            1, 1, data_format='channels_last', kernel_regularizer=self.l2_reg)
        self.value_bn = tf.keras.layers.BatchNormalization(momentum=0.9999)
        self.value_fc = tf.keras.layers.Dense(num_channels)
        self.value_pred = tf.keras.layers.Dense(1)

    def call(self, x, training):
        x = tf.transpose(x, [0, 2, 3, 1])
        x = self.relu(self.bn1(self.conv1(x), training=training))
        for i in range(self.num_res_layers):
            res = self.relu(getattr(self, 'res%d_bn1' % (i+1))(
                getattr(self, 'res%d_conv1' % (i+1))(x), training=training))
            res = self.relu(getattr(self, 'res%d_bn2' % (i+1))(
                getattr(self, 'res%d_conv2' % (i+1))(x), training=training))
            x = x + res
        x_policy = self.relu(self.policy_bn(self.policy_conv(x),
                             training=training))
        x_policy = tf.reshape(x_policy, [x_policy.get_shape()[0], -1])
        x_policy = self.policy_pred(x_policy)

        x_value = self.relu(self.value_bn(self.value_conv(x),
                            training=training))
        x_value = tf.reshape(x_value, [x_value.get_shape()[0], -1])
        x_value = self.relu(self.value_fc(x_value))
        x_value = self.tanh(self.value_pred(x_value))
        return [x_policy, x_value]


class NaiveNet(tf.keras.Model):
    """ This is actually not a network, but rather a constant returner. The
    value is always zero and the policy is a uniform distribution over the
    number of outputs. NaiveNet is used to simulate a player without any
    knowledge.

    Args:
      num_outputs: int: number of policy outputs to predict
    """
    def __init__(self, num_outputs):
        super(NaiveNet, self).__init__()
        self.num_outputs = num_outputs

    def call(self, x, training):
        x_policy = tf.ones(
            [x.get_shape()[0], self.num_outputs], tf.float32) / \
            self.num_outputs
        x_value = tf.zeros([x.get_shape()[0], 1], tf.float32)
        return [x_policy, x_value]
