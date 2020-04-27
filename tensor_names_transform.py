#! /usr/bin/env python
# coding=utf-8


import tensorflow as tf
from core.yolov3_face import YOLOV3
from tensorflow.python import pywrap_tensorflow

ckpt_file = "./checkpoint/yolov3_test_loss=0.5864.ckpt-1"

reader = pywrap_tensorflow.NewCheckpointReader(ckpt_file)

with open('tensor_names', 'r') as fr:
    lines = fr.readlines()
with tf.Session() as sess:
    with tf.name_scope('input'):
        input_data = tf.placeholder(dtype=tf.float32, name='input_data')
    model = YOLOV3(input_data, trainable=False)
    saver = tf.train.Saver(tf.global_variables())
    sess.run(tf.global_variables_initializer())
    trainable_variables = tf.trainable_variables()
    for v in trainable_variables:
        if v.name[:-2] + "\n" in lines:
            sess.run(tf.assign(v, reader.get_tensor(v.name[:-2])))
    ckpt_file1 = "./checkpoint/yolov3_test_loss_face.ckpt"
    saver.save(sess, ckpt_file1)





