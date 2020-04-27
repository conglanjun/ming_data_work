#! /usr/bin/env python
# coding=utf-8

import core.common as common
import tensorflow as tf


def darknet53(input_data, trainable):

    with tf.variable_scope('darknet'):

        input_data = common.convolutional(input_data, filters_shape=(3, 3,  3,  32), trainable=trainable, name='conv0')  # 416 x 416 x  32
        input_data = tf.layers.dropout(input_data, rate=0.5, training=trainable)
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 32,  64),
                                          trainable=trainable, name='conv1', downsample=True)  # 208 x 208 x  64
        input_data = tf.layers.dropout(input_data, rate=0.5, training=trainable)
        for i in range(1):
            residual_data = common.residual_block(input_data,  64,  32, 64, trainable=trainable, name='residual%d' %(i+0))  # 208 x 208 x  32, 208 x 208 x  64
            input_data = tf.layers.dropout(input_data, rate=0.5, training=trainable)
            input_data = common.attention_residual_block(input_data, residual_data, 208, 64, name='attention_residual%d' %(i+0))

        input_data = common.convolutional(input_data, filters_shape=(3, 3,  64, 128),
                                          trainable=trainable, name='conv4', downsample=True)  # 104 x 104 x 128
        input_data = tf.layers.dropout(input_data, rate=0.5, training=trainable)

        # 32*32*3->8*8*128

        # for i in range(1):
        #     residual_data = common.residual_block(input_data, 128,  64, 128, trainable=trainable, name='residual%d' %(i+1))  # 104 x 104 x  64, 104 x 104 x 128
        #     input_data = common.attention_residual_block(input_data, residual_data, 104, 128, name='attention_residual%d' % (i + 1))
        #
        # input_data = common.convolutional(input_data, filters_shape=(3, 3, 128, 256),
        #                                   trainable=trainable, name='conv9', downsample=True)  # 52 x  52 x 256
        #
        # for i in range(1):
        #     residual_data = common.residual_block(input_data, 256, 128, 256, trainable=trainable, name='residual%d' %(i+3))  # 52 x  52 x 128, 52 x  52 x 256
        #     input_data = common.attention_residual_block(input_data, residual_data, 52, 256, name='attention_residual%d' % (i + 3))
        #
        # route_1 = input_data
        # input_data = common.convolutional(input_data, filters_shape=(3, 3, 256, 512),
        #                                   trainable=trainable, name='conv26', downsample=True)  # 26 x  26 x 512
        #
        # for i in range(1):
        #     residual_data = common.residual_block(input_data, 512, 256, 512, trainable=trainable, name='residual%d' %(i+11))  # 26 x  26 x 256, 26 x  26 x 512
        #     input_data = common.attention_residual_block(input_data, residual_data, 26, 512, name='attention_residual%d' % (i + 11))
        #
        # route_2 = input_data
        # input_data = common.convolutional(input_data, filters_shape=(3, 3, 512, 1024),
        #                                   trainable=trainable, name='conv43', downsample=True)  # 13 x  13 x1024
        #
        # for i in range(1):
        #     residual_data = common.residual_block(input_data, 1024, 512, 1024, trainable=trainable, name='residual%d' %(i+19))  # 13 x  13 x 512, 13 x  13 x1024
        #     input_data = common.attention_residual_block(input_data, residual_data, 13, 1024, name='attention_residual%d' % (i + 19))

        # return route_1, route_2, input_data
        return input_data




"""
layer     filters    size              input                output
   0 conv     32  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  32 0.299 BF
   1 conv     64  3 x 3 / 2   416 x 416 x  32   ->   208 x 208 x  64 1.595 BF
   2 conv     32  1 x 1 / 1   208 x 208 x  64   ->   208 x 208 x  32 0.177 BF
   3 conv     64  3 x 3 / 1   208 x 208 x  32   ->   208 x 208 x  64 1.595 BF
   4 Shortcut Layer: 1
   5 conv    128  3 x 3 / 2   208 x 208 x  64   ->   104 x 104 x 128 1.595 BF
   6 conv     64  1 x 1 / 1   104 x 104 x 128   ->   104 x 104 x  64 0.177 BF
   7 conv    128  3 x 3 / 1   104 x 104 x  64   ->   104 x 104 x 128 1.595 BF
   8 Shortcut Layer: 5
   9 conv     64  1 x 1 / 1   104 x 104 x 128   ->   104 x 104 x  64 0.177 BF
  10 conv    128  3 x 3 / 1   104 x 104 x  64   ->   104 x 104 x 128 1.595 BF
  11 Shortcut Layer: 8
  12 conv    256  3 x 3 / 2   104 x 104 x 128   ->    52 x  52 x 256 1.595 BF
  13 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
  14 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
  15 Shortcut Layer: 12
  16 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
  17 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
  18 Shortcut Layer: 15
  19 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
  20 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
  21 Shortcut Layer: 18
  22 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
  23 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
  24 Shortcut Layer: 21
  25 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
  26 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
  27 Shortcut Layer: 24
  28 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
  29 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
  30 Shortcut Layer: 27
  31 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
  32 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
  33 Shortcut Layer: 30
  34 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
  35 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
  36 Shortcut Layer: 33
  37 conv    512  3 x 3 / 2    52 x  52 x 256   ->    26 x  26 x 512 1.595 BF
  38 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
  39 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
  40 Shortcut Layer: 37
  41 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
  42 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
  43 Shortcut Layer: 40
  44 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
  45 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
  46 Shortcut Layer: 43
  47 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
  48 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
  49 Shortcut Layer: 46
  50 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
  51 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
  52 Shortcut Layer: 49
  53 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
  54 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
  55 Shortcut Layer: 52
  56 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
  57 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
  58 Shortcut Layer: 55
  59 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
  60 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
  61 Shortcut Layer: 58
  62 conv   1024  3 x 3 / 2    26 x  26 x 512   ->    13 x  13 x1024 1.595 BF
  63 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512 0.177 BF
  64 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024 1.595 BF
  65 Shortcut Layer: 62
  66 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512 0.177 BF
  67 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024 1.595 BF
  68 Shortcut Layer: 65
  69 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512 0.177 BF
  70 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024 1.595 BF
  71 Shortcut Layer: 68
  72 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512 0.177 BF
  73 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024 1.595 BF
  74 Shortcut Layer: 71
  75 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512 0.177 BF
  76 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024 1.595 BF
  77 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512 0.177 BF
  78 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024 1.595 BF
  79 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512 0.177 BF
  80 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024 1.595 BF
  81 conv     18  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x  18 0.006 BF
  82 yolo
  83 route  79
  84 conv    256  1 x 1 / 1    13 x  13 x 512   ->    13 x  13 x 256 0.044 BF
  85 upsample            2x    13 x  13 x 256   ->    26 x  26 x 256
  86 route  85 61
  87 conv    256  1 x 1 / 1    26 x  26 x 768   ->    26 x  26 x 256 0.266 BF
  88 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
  89 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
  90 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
  91 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
  92 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
  93 conv     18  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x  18 0.012 BF
  94 yolo
  95 route  91
  96 conv    128  1 x 1 / 1    26 x  26 x 256   ->    26 x  26 x 128 0.044 BF
  97 upsample            2x    26 x  26 x 128   ->    52 x  52 x 128
  98 route  97 36
  99 conv    128  1 x 1 / 1    52 x  52 x 384   ->    52 x  52 x 128 0.266 BF
 100 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
 101 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
 102 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
 103 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
 104 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
 105 conv     18  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x  18 0.025 BF
 106 yolo

"""

"""
get pre-model parameters to new model
-------
from tensorflow.python import pywrap_tensorflow
reader = pywrap_tensorflow.NewCheckpointReader(pre_train_model_path)

# 获取当前图可训练变量
trainable_variables = tf.trainable_variables()
# 需要赋值的当前网络层变量，这里只是随便起的名字。
restore_v_target_name = "fc_target"
# 需要的预训练模型中的某层的名字
restore_v_source_name = "fc_source"
for v in trainable_variables:
    if restore_v_target_name == v.name:
"""
