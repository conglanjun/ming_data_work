#! /usr/bin/env python
# coding=utf-8

import numpy as np
import tensorflow as tf
import core.utils as utils
import core.common as common
import core.backbone as backbone
from core.config import cfg


class YOLOV3(object):
    """Implement tensoflow yolov3 here"""
    def __init__(self, input_data, trainable):

        self.trainable        = trainable
        self.classes          = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_class        = len(self.classes)
        self.strides          = np.array(cfg.YOLO.STRIDES)
        # self.anchors          = utils.get_anchors(cfg.YOLO.ANCHORS)
        # self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        # self.iou_loss_thresh  = cfg.YOLO.IOU_LOSS_THRESH
        self.upsample_method  = cfg.YOLO.UPSAMPLE_METHOD

        try:
            self.conv_l = self.__build_nework(input_data)
        except:
            raise NotImplementedError("Can not build up yolov3 network!")

    def __build_nework(self, input_data):

        input_data = backbone.darknet53(input_data, self.trainable)

        # input_data = common.convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv52')
        # input_data = common.convolutional(input_data, (3, 3,  512, 1024), self.trainable, 'conv53')
        # input_data = common.convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv54')
        # input_data = common.convolutional(input_data, (3, 3,  512, 1024), self.trainable, 'conv55')
        # input_data = common.convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv56')
        #
        # conv_lobj_branch = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, name='conv_lobj_branch')
        # conv_l = common.convolutional(input_data, (1, 1, 1024, self.num_class),
        #                                   trainable=self.trainable, name='conv_l', activate=False, bn=False)

        conv_l = tf.layers.flatten(input_data)
        conv_l = tf.reshape(conv_l, (-1, 8 * 8 * 128))
        conv_l = tf.layers.dense(conv_l, units=1024, activation=tf.nn.leaky_relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
        conv_l = tf.layers.dropout(conv_l, rate=0.4)
        conv_l = tf.layers.dense(conv_l, units=512, activation=tf.nn.leaky_relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
        conv_l = tf.layers.dropout(conv_l, rate=0.4)
        conv_l = tf.layers.dense(conv_l, units=self.num_class, name='conv_l', activation=tf.nn.softmax)

        # input_data = common.convolutional(input_data, (1, 1,  1024,  256), self.trainable, 'conv57')
        # input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)
        #
        # with tf.variable_scope('route_1'):
        #     input_data = tf.concat([input_data, route_2], axis=-1)

        # input_data = common.convolutional(input_data, (1, 1, 768, 256), self.trainable, 'conv58')
        # input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv59')
        # input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv60')
        # input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv61')
        # input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv62')
        #
        # conv_mobj_branch = common.convolutional(input_data, (3, 3, 256, 512),  self.trainable, name='conv_mobj_branch')
        # conv_m = common.convolutional(input_data, (1, 1, 768, self.num_class),
        #                                   trainable=self.trainable, name='conv_m', activate=False, bn=False)

        # conv_m = tf.layers.flatten(input_data)
        # conv_m = tf.reshape(conv_m, (-1, 26 * 26 * 768))
        # conv_m = tf.layers.dense(conv_m, units=self.num_class, name='conv_m', activation=tf.nn.relu)
        #
        # input_data = common.convolutional(input_data, (1, 1, 768, 128), self.trainable, 'conv63')
        # input_data = common.upsample(input_data, name='upsample1', method=self.upsample_method)
        #
        # with tf.variable_scope('route_2'):
        #     input_data = tf.concat([input_data, route_1], axis=-1)

        # input_data = common.convolutional(input_data, (1, 1, 384, 128), self.trainable, 'conv64')
        # input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv65')
        # input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv66')
        # input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv67')
        # input_data = common.convolutional(input_data, (1, 1, 384, 128), self.trainable, 'conv68')
        #
        # conv_sobj_branch = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, name='conv_sobj_branch')
        # conv_s = common.convolutional(input_data, (1, 1, 384, self.num_class),
        #                                   trainable=self.trainable, name='conv_s', activate=False, bn=False)

        # conv_s = tf.layers.flatten(input_data)
        # conv_s = tf.reshape(conv_s, (-1, 52 * 52 * 384))
        # conv_s = tf.layers.dense(conv_s, units=self.num_class, name='conv_s', activation=tf.nn.relu)

        return conv_l

    def decode(self, conv_output):
        """
        return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
               contains (x, y, w, h, score, probability)
        """

        conv_shape       = tf.shape(conv_output)
        batch_size       = conv_shape[0]
        output_size      = conv_shape[1]

        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, self.num_class))

        conv_raw_prob = conv_output[:, :, :, :]
        pred_prob = tf.sigmoid(conv_raw_prob)

        return pred_prob

    def focal(self, target, actual, alpha=1, gamma=2):
        focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss

    def bbox_giou(self, boxes1, boxes2):

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / union_area

        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]
        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

        return giou

    def bbox_iou(self, boxes1, boxes2):

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = 1.0 * inter_area / union_area

        return iou

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

        loss = self.loss_layer(self.conv_l, label)

        return loss


