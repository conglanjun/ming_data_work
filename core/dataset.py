#! /usr/bin/env python
# coding=utf-8

import os
import cv2
import random
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from tensorflow.python.keras.datasets import cifar10
import matplotlib.pyplot as plt



class Dataset(object):
    """implement Dataset here"""
    def __init__(self, dataset_type):
        self.annot_path  = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        self.batch_size  = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug    = cfg.TRAIN.DATA_AUG   if dataset_type == 'train' else cfg.TEST.DATA_AUG

        self.train_input_size = cfg.TRAIN.INPUT_SIZE[0]
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)

        (X_train_orig, Y_train_orig), (X_test_orig, Y_test_orig) = cifar10.load_data()

        X_train = X_train_orig / 255.
        X_test = X_test_orig / 255.

        Y_train = tf.keras.utils.to_categorical(Y_train_orig, 10)
        Y_test = tf.keras.utils.to_categorical(Y_test_orig, 10)
        self.trainset = X_train
        self.testset = X_test
        self.y_train = Y_train
        self.y_test = Y_test
        self.run_flag = 'train'
        # self.annotations = self.load_annotations(dataset_type)
        self.num_samples = len(self.trainset)
        self.num_samples_test = len(self.testset)
        self.num_batchs = int(np.floor(self.num_samples / self.batch_size))
        self.num_batchs_test = int(np.floor(self.num_samples_test / self.batch_size))
        self.batch_count = 0


    def load_annotations(self, dataset_type):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        return annotations

    def __iter__(self):
        return self

    def __next__(self):

        with tf.device('/cpu:0'):
            self.train_output_sizes = self.train_input_size // self.strides

            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3))

            batch_label = np.zeros((self.batch_size, self.num_classes))

            num = 0
            if self.run_flag == 'train':
                num_batchs = self.num_batchs
            else:
                num_batchs = self.num_batchs_test
            if self.batch_count < num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples: index -= self.num_samples
                    if self.run_flag == 'train':
                        image, label = self.random_horizontal_flip(self.trainset[index]), self.y_train[index]
                    else:
                        image, label = self.random_horizontal_flip(self.testset[index]), self.y_test[index]

                    batch_image[num, :, :, :] = cv2.resize(image, (self.train_input_size, self.train_input_size))
                    batch_label[num, :] = label
                    num += 1
                self.batch_count += 1
                return batch_image, batch_label
            else:
                self.batch_count = 0
                raise StopIteration

    def random_horizontal_flip(self, image):
        # if random.random() < 0.5:
        #     if random.random() < 0.3:
        #         image = cv2.flip(image, 1)
        #     elif random.random() < 0.6:
        #         image = cv2.flip(image, 0)
        #     else:
        #         image = cv2.flip(image, -1)

        return image

    def parse_annotation(self, annotation):

        line = annotation.split()
        image_path = line[0]
        label = line[1]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " %image_path)
        image = np.array(cv2.imread(image_path))

        if self.data_aug:
            image = self.random_horizontal_flip(np.copy(image))

        image = utils.image_preporcess(np.copy(image), [self.train_input_size, self.train_input_size])

        onehot = np.zeros(self.num_classes, dtype=np.float)
        onehot[label] = 1.0

        return image, onehot

    def __len__(self):
        return self.num_batchs




