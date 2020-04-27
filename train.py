#! /usr/bin/env python
# coding=utf-8

import os
import time
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from tqdm import tqdm
# from core.dataset import Dataset
from core.dataset_face import Dataset
from core.yolov3_face import YOLOV3
from core.config import cfg

from tensorflow.python import pywrap_tensorflow


class YoloTrain(object):
    def __init__(self):
        self.classes             = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes         = len(self.classes)
        self.learn_rate_init     = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end      = cfg.TRAIN.LEARN_RATE_END
        self.first_stage_epochs  = cfg.TRAIN.FISRT_STAGE_EPOCHS
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        self.warmup_periods      = cfg.TRAIN.WARMUP_EPOCHS
        self.initial_weight      = cfg.TRAIN.INITIAL_WEIGHT
        self.time                = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay    = cfg.YOLO.MOVING_AVE_DECAY
        self.train_logdir        = "./data/log/train"
        self.dataset = Dataset('train')
        self.run_flag = 'train'
        self.steps_per_period    = len(self.dataset.train_lines)
        self.sess                = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.train_op = None

        with tf.name_scope('define_input'):
            self.input_data   = tf.placeholder(dtype=tf.float32, name='input_data')
            self.label = tf.placeholder(dtype=tf.float32, name='label')
            self.trainable     = tf.placeholder(dtype=tf.bool, name='training')

            # pbar = tqdm(self.dataset)
            # for train_data in pbar:
            #     print(train_data)

        with tf.name_scope("define_loss"):
            self.model = YOLOV3(self.input_data, self.trainable)
            self.net_var = tf.global_variables()

            ckpt_file = self.initial_weight

            reader = pywrap_tensorflow.NewCheckpointReader(ckpt_file)

            with open('tensor_names', 'r') as fr:
                lines = fr.readlines()
                print(lines)

            self.sess.run(tf.global_variables_initializer())
            trainable_variables = tf.trainable_variables()
            for v in trainable_variables:
                if v.name[:-2] + "\n" in lines:
                    self.sess.run(tf.assign(v, reader.get_tensor(v.name[:-2])))
                    print('success assign model weights:%s', v.name[:-2])

            self.loss = self.model.compute_loss(self.label)
            correct_prediction = tf.equal(tf.argmax(self.model.conv_s, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
                                        dtype=tf.float64, name='warmup_steps')
            train_steps = tf.constant( (self.first_stage_epochs + self.second_stage_epochs)* self.steps_per_period,
                                        dtype=tf.float64, name='train_steps')
            self.learn_rate = tf.cond(
                pred=self.global_step < warmup_steps,
                true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
                false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) *
                                    (1 + tf.cos(
                                        (self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
            )
            global_step_update = tf.assign_add(self.global_step, 1.0)

        with tf.name_scope("define_weight_decay"):
            moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())

        with tf.name_scope("define_first_stage_train"):
            self.first_stage_trainable_var_list = []
            for var in tf.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                if var_name_mess[0] in ['conv_s', 'conv_m', 'conv_l']:
                    self.first_stage_trainable_var_list.append(var)

            first_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss, var_list=self.first_stage_trainable_var_list)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_frozen_variables = tf.no_op()

        with tf.name_scope("define_second_stage_train"):
            second_stage_trainable_var_list = tf.trainable_variables()
            second_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss, var_list=second_stage_trainable_var_list)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([second_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_all_variables = tf.no_op()

        with tf.name_scope('loader_and_saver'):
            self.loader = tf.train.Saver(self.net_var)
            self.saver  = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        with tf.name_scope('summary'):
            tf.summary.scalar("learn_rate",      self.learn_rate)
            tf.summary.scalar("total_loss", self.loss)

            logdir = "./data/log/"
            if os.path.exists(logdir): shutil.rmtree(logdir)
            os.mkdir(logdir)
            self.write_op = tf.summary.merge_all()
            self.summary_writer  = tf.summary.FileWriter(logdir, graph=self.sess.graph)


    def train(self):
        # self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        try:
            print('=> Restoring weights from: %s ... ' % self.initial_weight)
            self.loader.restore(self.sess, self.initial_weight)
        except:
            print('=> %s does not exist !!!' % self.initial_weight)
            print('=> Now it starts to train YOLOV3 from scratch ...')
            self.first_stage_epochs = 0
        for epoch in range(1, 1+self.first_stage_epochs+self.second_stage_epochs):
            if epoch <= self.first_stage_epochs:
                self.train_op = self.train_op_with_frozen_variables
            else:
                self.train_op = self.train_op_with_all_variables

            self.dataset.run_flag = 'train'
            pbar = tqdm(self.dataset)
            train_epoch_loss, test_epoch_loss, test_epoch_accuracy = [], [], []

            for train_data in pbar:
                _, summary, train_step_loss, global_step_val, accuracy = self.sess.run(
                    [self.train_op, self.write_op, self.loss, self.global_step, self.accuracy], feed_dict={
                                                self.input_data:   train_data[0],
                                                self.label:  train_data[1],
                                                self.trainable:    True,
                })

                train_epoch_loss.append(train_step_loss)
                self.summary_writer.add_summary(summary, global_step_val)
                pbar.set_description("train loss: %.2f accuracy: %.4f" % (train_step_loss, accuracy))

            self.dataset.run_flag = 'test'
            pbar1 = tqdm(self.dataset)
            for test_data in pbar1:
                test_step_loss, accuracy = self.sess.run([self.loss, self.accuracy], feed_dict={
                                                self.input_data:   test_data[0],
                                                self.label:  test_data[1],
                                                self.trainable:    False,
                })

                test_epoch_loss.append(test_step_loss)
                test_epoch_accuracy.append(accuracy)
                pbar1.set_description("test loss: %.2f accuracy: %.4f" % (test_step_loss, accuracy))

            train_epoch_loss, test_epoch_loss, test_epoch_accuracy = np.mean(train_epoch_loss), np.mean(test_epoch_loss), np.mean(test_epoch_accuracy)
            ckpt_file = "./checkpoint/yolov3_test_loss=%.4f.ckpt" % test_epoch_loss
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Accuracy : %.4f Saving %s ..."
                            % (epoch, log_time, train_epoch_loss, test_epoch_loss, test_epoch_accuracy, ckpt_file))
            self.saver.save(self.sess, ckpt_file, global_step=epoch)



if __name__ == '__main__': YoloTrain().train()

"""

fine tune https://github.com/YaoLing13/keras-yolo3-fine-tune/blob/master/train.py

def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model
"""


