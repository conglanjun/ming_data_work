#! /usr/bin/env python
# coding=utf-8


import tensorflow as tf
from core.yolov3_face import YOLOV3
from tensorflow.python import pywrap_tensorflow

pb_file = "./yolov3_cifar10.pb"
ckpt_file = "./checkpoint/yolov3_test_loss_face.ckpt"

reader = pywrap_tensorflow.NewCheckpointReader(ckpt_file)

with tf.Session() as sess:
    with tf.name_scope('input'):
        input_data = tf.placeholder(dtype=tf.float32, name='input_data')
    model = YOLOV3(input_data, trainable=False)
    trainable_variables = tf.trainable_variables()
    for v in trainable_variables:
        tensor = reader.get_tensor(v.name[:-2])
        print(tensor)





