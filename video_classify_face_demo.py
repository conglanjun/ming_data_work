#! /usr/bin/env python
# coding=utf-8

import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
import os

return_elements = ["conv_l/Softmax:0", "input/input_data:0"]
pb_file         = "./yolov3_face.pb"

# (X_train_orig, Y_train_orig), (X_test_orig, Y_test_orig) = cifar10.load_data()
# for i in range(10):
#     img = X_test_orig[i]
#     label = Y_test_orig[i]
#     cv2.imwrite("./docs/test_images/" + str(i) + "_" + str(label[0]) + ".jpg", img)

num_classes     = 19
input_size      = 416
graph           = tf.Graph()

# original_image = cv2.imread(image_path)
# original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
# original_image_size = original_image.shape[:2]
# image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
with open('./core/train_face_mask_list.txt', 'r') as fr:
    lines = fr.readlines()

vs = cv2.VideoCapture(0)
while vs.isOpened():
    ret, frame = vs.read()
#     image = cv2.imwrite('test_clj.jpg', frame)
#     break
# image_org = cv2.imread("test_clj1.jpg")
    image_org1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image_org = image_org1 / 255.
# # image_org = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# # image_org.show()
#
#
# # image_org = frame
    image_org = cv2.resize(image_org, (input_size, input_size))
    image = image_org[np.newaxis, ...]

    return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

    with tf.Session(graph=graph) as sess:
        pred = sess.run([return_tensors[0]], feed_dict={ return_tensors[1]: image})
        clazz = lines[np.argmax(pred[0])].strip()
        print(pred)
        print(clazz)

    cv2.putText(image_org1, clazz, (0, 100), 0, 0.6, (255, 200, 100))
    image_org1 = Image.fromarray(image_org1)
    image_org1.show()

# import time
# vs = cv2.VideoCapture(1, cv2.CAP_DSHOW)
# i = 100
# while vs.isOpened():
#     print('read...', str(i))
#     time.sleep(1)
#     ret, frame = vs.read()
#     image = cv2.imwrite('data/test_clj' + str(i) + '.jpg', frame)
#     i += 1
#     if i >= 140:
#         break
# vs.release()


