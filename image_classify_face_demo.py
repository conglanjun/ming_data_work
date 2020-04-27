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
listdir = os.listdir("./docs/face_test_images/")
total = 0
accuracy = 0
for dir in listdir:
    face_mask_dir = os.path.join("./docs/face_test_images/", dir)
    face_mask_dirs = os.listdir(face_mask_dir)
    for face_mask_img_path in face_mask_dirs:
        image_org = cv2.imread(os.path.join(face_mask_dir, face_mask_img_path))
        image_org = cv2.resize(image_org, (input_size, input_size))
        image = image_org[np.newaxis, ...]

        return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

        total += 1.0

        with tf.Session(graph=graph) as sess:
            pred = sess.run([return_tensors[0]], feed_dict={ return_tensors[1]: image})
            clazz = lines[np.argmax(pred[0])].strip()
            print(clazz + ":" + face_mask_img_path)

        if lines[np.argmax(pred[0])].strip() in face_mask_img_path:
            accuracy += 1.0

        print('accuracy:%.4f, total:%d, acc:%d' % (accuracy / float(total), total, accuracy))

        # cv2.putText(image_org, clazz, (0, 10), 0, 0.6, (255, 200, 100))

        # image_org = Image.fromarray(cv2.cvtColor(image_org, cv2.COLOR_BGR2RGB))
        # image_org.show()




