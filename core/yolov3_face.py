#! /usr/bin/env python
# coding=utf-8

import numpy as np
import tensorflow as tf
import core.utils as utils
import core.common as common
import core.backbone_face as backbone
from core.config import cfg


class YOLOV3(object):
    """Implement tensoflow yolov3 here"""
    def __init__(self, input_data, trainable):

        self.trainable        = trainable
        self.classes          = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_class        = len(self.classes)
        self.strides          = np.array(cfg.YOLO.STRIDES)
        self.upsample_method  = cfg.YOLO.UPSAMPLE_METHOD

        try:
            self.conv_s = self.__build_nework(input_data)
        except:
            raise NotImplementedError("Can not build up yolov3 network!")

    def __build_nework(self, input_data):

        route_1, route_2, input_data = backbone.darknet53(input_data, self.trainable)

        # input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv52')
        # input_data = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, 'conv53')
        # input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv54')
        # input_data = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, 'conv55')
        # input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv56')
        #
        # conv_lobj_branch = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, name='conv_lobj_branch')
        # conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (self.num_class + 5)),
        #                                   trainable=self.trainable, name='conv_lbbox', activate=False, bn=False)
        #
        # input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv57')
        input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)

        with tf.variable_scope('route_1'):
            input_data = tf.concat([input_data, route_2], axis=-1)  # 1024+512=1536

        # input_data = common.convolutional(input_data, (1, 1, 768, 256), self.trainable, 'conv58')
        # input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv59')
        # input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv60')
        # input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv61')
        # input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv62')
        #
        # conv_mobj_branch = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, name='conv_mobj_branch')
        # conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3 * (self.num_class + 5)),
        #                                   trainable=self.trainable, name='conv_mbbox', activate=False, bn=False)
        #
        # input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv63')
        input_data = common.upsample(input_data, name='upsample1', method=self.upsample_method)

        with tf.variable_scope('route_2'):
            input_data = tf.concat([input_data, route_1], axis=-1)  # 1536+256=1792

        # input_data = common.convolutional(input_data, (1, 1, 384, 128), self.trainable, 'conv64')
        # input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv65')
        # input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv66')
        # input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv67')
        # input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv68')

        # conv_sobj_branch = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, name='conv_sobj_branch')
        # conv_s = common.convolutional(conv_sobj_branch, (1, 1, 256, 3 * (self.num_class + 5)),
        #                                   trainable=self.trainable, name='conv_sbbox', activate=False, bn=False)

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 1792,  256), trainable=True, name='conv_s_conv', downsample=True)  # 26*26*256
        input_data = common.convolutional(input_data, filters_shape=(1, 1, 256, self.num_class), trainable=True, name='conv_s_conv1', downsample=True)  # 13*13*19
        conv_s = tf.layers.flatten(input_data)
        conv_s = tf.reshape(conv_s, (-1, 13*13*self.num_class))
        conv_s = tf.layers.dense(conv_s, units=1024, activation=tf.nn.leaky_relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
        conv_s = tf.layers.dropout(conv_s, rate=0.5)
        conv_s = tf.layers.dense(conv_s, units=512, activation=tf.nn.leaky_relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
        conv_s = tf.layers.dropout(conv_s, rate=0.5)
        conv_s = tf.layers.dense(conv_s, units=self.num_class, name='conv_l', activation=tf.nn.softmax)

        return conv_s

    def loss_layer(self, conv, label):

        # label = tf.argmax(label, axis=1)
        # conv = tf.argmax(conv, axis=1)
        #
        # prob_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=conv)
        # prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1]))

        # prob_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=conv))

        prob_loss = tf.keras.losses.categorical_crossentropy(y_true=label, y_pred=conv)
        prob_loss = tf.reduce_mean(prob_loss)

        return prob_loss

    def compute_loss(self, label):

        # with tf.name_scope('smaller_loss'):
        #     loss_s = self.loss_layer(self.conv_s, label)
        #
        # with tf.name_scope('medium_loss'):
        #     loss_m = self.loss_layer(self.conv_m, label)
        #
        # with tf.name_scope('bigger_loss'):
        #     loss_l = self.loss_layer(self.conv_l, label)

        loss = self.loss_layer(self.conv_s, label)

        return loss


