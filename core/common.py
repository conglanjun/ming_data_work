#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf
import numpy as np


def convolutional(input_data, filters_shape, trainable, name, downsample=False, activate=True, bn=True):

    with tf.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"

        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True, shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01), regularizer=tf.contrib.layers.l2_regularizer(0.1))
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True, dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)

        if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv


def residual_block(input_data, input_channel, filter_num1, filter_num2, trainable, name):

    short_cut = input_data

    with tf.variable_scope(name):
        input_data = convolutional(input_data, filters_shape=(1, 1, input_channel, filter_num1),
                                   trainable=trainable, name='conv1')
        input_data = tf.layers.dropout(input_data, rate=0.5, training=trainable)
        input_data = convolutional(input_data, filters_shape=(3, 3, filter_num1,   filter_num2),
                                   trainable=trainable, name='conv2')
        input_data = tf.layers.dropout(input_data, rate=0.5, training=trainable)

        residual_output = input_data + short_cut

    return residual_output


def attention_residual_block(conv, residual_output, kernel_size, feature_size, name):

    with tf.variable_scope(name):  # residual_output:6 * 208 * 208 * 64
        avg_pool = tf.nn.avg_pool(residual_output, strides=[1, kernel_size, kernel_size, 1], ksize=[1, kernel_size, kernel_size, 1], padding="SAME")  # feature map size avg_pool:6*1*1*64
        max_pool = tf.nn.max_pool(residual_output, strides=[1, kernel_size, kernel_size, 1], ksize=[1, kernel_size, kernel_size, 1], padding="SAME")  # feature map size max_pool:6*1*1*64

        avg_fc1 = tf.layers.dense(avg_pool, units=feature_size / 16, activation=tf.nn.leaky_relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))  # (6, 1, 1, 4)
        avg_fc2 = tf.layers.dense(avg_fc1, units=feature_size, activation=tf.nn.softmax, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))  # (6, 1, 1, 64)
        # sess.run(tf.global_variables_initializer())
        # result = sess.run(avg_fc2, feed_dict={input_data1: data[0], trainable: True})
        # print(result.shape)

        max_fc1 = tf.layers.dense(max_pool, units=feature_size / 16, activation=tf.nn.leaky_relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
        max_fc2 = tf.layers.dense(max_fc1, units=feature_size, activation=tf.nn.softmax, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))  # (6, 1, 1, 64)

        fc2 = (avg_fc2 + max_fc2) / 2.0  # (6, 1, 1, 64)

        # yc = tf.transpose(residual_output, perm=[3, 0, 1, 2])  # 64 size height width
        yc = residual_output  # residual_output:6 * 208 * 208 * 64

        output1 = yc * fc2  # (6, 208, 208, 64)
        # output1 = tf.transpose(output1, perm=[1, 2, 3, 0])  # size height width 64

        # spatial
        tf_max = tf.reduce_max(output1, axis=3, keepdims=True)  # (6, 208, 208, 1)
        tf_mean = tf.reduce_mean(output1, axis=3, keepdims=True)  # (6, 208, 208, 1)
        channel_concat = tf.concat([tf_max, tf_mean], 3)  # (6, 208, 208, 2)

        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True, shape=(3, 3, 2, 1), initializer=tf.random_normal_initializer(stddev=0.01))
        channel_conv = tf.nn.conv2d(input=channel_concat, filter=weight, strides=[1, 1, 1, 1], padding='SAME')  # (6, 208, 208, 1)
        channel_prob = tf.nn.softmax(channel_conv)  # (6, 208, 208, 1)
        output2 = output1 * channel_prob  # (6, 208, 208, 64)

        residual_attention = output2 + conv  # (6, 208, 208, 64)

    return residual_attention



def route(name, previous_output, current_output):

    with tf.variable_scope(name):
        output = tf.concat([current_output, previous_output], axis=-1)

    return output


def upsample(input_data, name, method="deconv"):
    assert method in ["resize", "deconv"]

    if method == "resize":
        with tf.variable_scope(name):
            input_shape = tf.shape(input_data)
            output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))

    if method == "deconv":
        # replace resize_nearest_neighbor with conv2d_transpose To support TensorRT optimization
        numm_filter = input_data.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(input_data, numm_filter, kernel_size=2, padding='same',
                                            strides=(2,2), kernel_initializer=tf.random_normal_initializer())

    return output


if __name__ == '__main__':
    # arr = np.arange(6).reshape((1, 2, 3))
    # print(arr.shape)
    # print(arr)
    # print('---------------')
    # t1 = arr.transpose((1, 0, 2))
    # print(t1.shape)
    # print(t1)
    # print('---------------')
    # t2 = arr.transpose((2, 0, 1))
    # print(t2.shape)
    # print(t2)
    # print('---------------')
    # t3 = t2.transpose((1, 0, 2)).transpose((0, 2, 1))
    # print(t3.shape)
    # print(t3)

    list_image = [
        [[[92, 18, 57],
          [66, 28, 38],
          [46, 29, 46],
          [25, 23, 90],
          [70, 13, 27],
          [56, 75, 20],
          [47, 21, 31],
          [12, 52, 98]],
        [[45, 18, 57],
         [66, 28, 38],
         [46, 29, 46],
         [25, 23, 90],
         [170, 13, 27],
         [56, 375, 20],
         [147, 21, 31],
         [12, 52, 98]],
        [[65, 18, 57],
         [66, 28, 211],
         [16, 29, 21],
         [25, 23, 90],
         [60, 13, 27],
         [56, 378, 20],
         [47, 201, 31],
         [45, 52, 98]],
        [[233, 34, 75],
         [15, 264, 52],
         [42, 62, 14],
         [31, 39, 59],
         [59, 20, 82],
         [61, 66, 22],
         [100, 34, 37],
         [15, 264, 52]],
        [[42, 62, 14],
         [31, 39, 59],
         [59, 20, 82],
         [61, 66, 22],
         [100, 34, 37],
         [15, 164, 52],
         [42, 62, 14],
         [31, 30, 54]],
        [[59, 20, 82],
         [61, 6, 22],
         [43, 12,90],
         [76,28,10],
         [87,21,199],
         [90,87,122],
         [98,124,211],
         [10,89,20]]
         ]
        ]

    array_image = np.squeeze(list_image, axis=0)
    print(array_image[:, :, 0])

    with tf.Session() as sess:
        # tf_shape = tf.reshape(array_image[0, 3, :], [3])
        # tf_max = tf.reduce_max(tf_shape)
        # position = sess.run(tf_max)
        # print(position)
        tensor_image = tf.convert_to_tensor(array_image)
        rmax = tf.reduce_max(tensor_image, axis=2, keepdims=True)
        tf_squeeze = tf.squeeze(rmax)  # 6 * 8 * 1  # todo channel avg pool and max pool
        concat = tf.concat([rmax, rmax], 2)
        result = sess.run(concat)  # 6 * 8
        print(result)

    image_np = np.array(list_image)
    print(image_np.shape)
    print(image_np[:, :, 1])


