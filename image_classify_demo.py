#! /usr/bin/env python
# coding=utf-8

import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
from tensorflow.python.keras.datasets import cifar10
import os

return_elements = ["conv_l/Softmax:0", "input/input_data:0"]
pb_file         = "./yolov3_cifar10.pb"

# (X_train_orig, Y_train_orig), (X_test_orig, Y_test_orig) = cifar10.load_data()
# for i in range(10):
#     img = X_test_orig[i]
#     label = Y_test_orig[i]
#     cv2.imwrite("./docs/test_images/" + str(i) + "_" + str(label[0]) + ".jpg", img)

image_path      = "./docs/images/road.jpeg"
num_classes     = 10
input_size      = 32
graph           = tf.Graph()

# original_image = cv2.imread(image_path)
# original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
# original_image_size = original_image.shape[:2]
# image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
with open('./data/classes/cifar10.names', 'r') as fr:
    lines = fr.readlines()
listdir = os.listdir("./docs/test_images/")
for dir in listdir:
    image_org = cv2.imread(os.path.join("./docs/test_images/", dir))
    image = image_org[np.newaxis, ...]

    return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)


    with tf.Session(graph=graph) as sess:
        pred = sess.run([return_tensors[0]], feed_dict={ return_tensors[1]: image})
        print(lines[np.argmax(pred[0])].strip() + ":" + dir)

    image_org = Image.fromarray(image_org)
    image_org.show()




