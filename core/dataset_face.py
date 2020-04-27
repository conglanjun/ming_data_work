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

        with open('./core/data_train_path.txt', 'r') as fr:
            lines = fr.readlines()
            seed = 1
            np.random.seed(seed=seed)
            np.random.shuffle(lines)
            self.train_lines = lines

        with open('./core/data_test_path.txt', 'r') as fr:
            lines = fr.readlines()
            seed = 1
            np.random.seed(seed=seed)
            np.random.shuffle(lines)
            self.test_lines = lines

        self.run_flag = 'train'
        # self.annotations = self.load_annotations(dataset_type)
        self.num_samples = len(self.train_lines)
        self.num_samples_test = len(self.test_lines)
        self.num_batchs = int(np.floor(self.num_samples / self.batch_size))
        # self.num_batchs_test = int(np.floor(self.num_samples_test / self.batch_size))
        self.num_batchs_test = int(np.floor(self.num_samples_test / 1))
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
                        datas = self.train_lines[index].split(',')
                        original_image = cv2.imread(os.path.join('./core', datas[0]))
                        image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                        label = int(datas[1].strip())
                    else:
                        datas = self.test_lines[index].split(',')
                        original_image = cv2.imread(os.path.join('./core', datas[0]))
                        image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                        label = int(datas[1].strip())

                    image = image / 255.

                    label = tf.keras.utils.to_categorical(label, self.num_classes)

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




