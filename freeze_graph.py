#! /usr/bin/env python
# coding=utf-8


import tensorflow as tf
from core.yolov3_face import YOLOV3

pb_file = "./yolov3_face.pb"
ckpt_file = "./checkpoint/yolov3_test_loss=1.4231.ckpt-67"
output_node_names = ["conv_l/Softmax", "input/input_data"]

with tf.name_scope('input'):
    input_data = tf.placeholder(dtype=tf.float32, name='input_data')

model = YOLOV3(input_data, trainable=False)
print(model.conv_s)

sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver()
saver.restore(sess, ckpt_file)

operations = sess.graph.get_operations()
for oper in operations:
    print(oper.name)

converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                            input_graph_def  = sess.graph.as_graph_def(),
                            output_node_names = output_node_names)

with tf.gfile.GFile(pb_file, "wb") as f:
    f.write(converted_graph_def.SerializeToString())




