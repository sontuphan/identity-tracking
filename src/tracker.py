from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2 as cv

IMAGE_SHAPE = (96, 96)
HISTORICAL_LENGTH = 4
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "../models/tpu/ohmnilabs_features_extractor_quant_postprocess.tflite")


class AppearanceExtractor(keras.Model):
    def __init__(self, tensor_length):
        super(AppearanceExtractor, self).__init__()
        self.fc_units = 128
        self.tensor_length = tensor_length
        self.conv = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.ft = tf.keras.layers.Flatten()
        self.fc = keras.layers.Dense(self.fc_units, activation='relu')

    def call(self, x):
        (batch_size, _, _, _, _) = x.shape
        imgs = tf.reshape(
            x, [batch_size*self.tensor_length, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3])
        conv_output = self.conv(imgs)
        ft_output = self.ft(conv_output)
        fc_output = self.fc(ft_output)
        y = tf.reshape(
            fc_output, [batch_size, self.tensor_length, self.fc_units])
        return y


# class MotionExtractor(keras.Model):
#     def __init__(self, tensor_length):
#         super(MotionExtractor, self).__init__()
#         self.fc_units = 128
#         self.tensor_length = tensor_length
#         self.fc = keras.layers.Dense(self.fc_units, activation='relu')

#     def call(self, x):
#         (batch_size, _, _) = x.shape
#         bbox_inputs = tf.reshape(x, [batch_size*self.tensor_length, 4])
#         fc_output = self.fc(bbox_inputs)
#         y = tf.reshape(
#             fc_output, [batch_size, self.tensor_length, self.fc_units])
#         return y


class Classification(keras.Model):
    def __init__(self, tensor_length):
        super(Classification, self).__init__()
        self.tensor_length = tensor_length
        self.fc1 = keras.layers.Dense(256, activation='relu')
        self.fc2 = keras.layers.Dense(64, activation='relu')
        self.fc3 = keras.layers.Dense(1, activation='sigmoid')

    # def call(self, app_features, mot_features):
    def call(self, app_features):
        features = app_features
        # features = tf.concat([app_features, mot_features], 2)
        (batch_size, _, _) = features.shape
        encode, decode = tf.split(features, [self.tensor_length-1, 1], axis=1)
        print("encode:", encode.shape)
        print("decode:", decode.shape)
        l_input = tf.reduce_mean(encode, 1)
        r_input = tf.reshape(decode, [batch_size, -1])
        x = tf.concat([l_input, r_input], 1)
        fc1_output = self.fc1(x)
        fc2_output = self.fc2(fc1_output)
        y = self.fc3(fc2_output)
        return y


class Tracker:
    def __init__(self):
        self.tensor_length = HISTORICAL_LENGTH
        self.batch_size = 64
        self.image_shape = IMAGE_SHAPE

        self.apex = AppearanceExtractor(self.tensor_length)
        # self.moex = MotionExtractor(self.tensor_length)
        self.clsf = Classification(self.tensor_length)

        self.optimizer = keras.optimizers.Adam()
        self.loss = keras.losses.BinaryCrossentropy()

        self.loss_metric = keras.metrics.Mean(name='train_loss')
        self.accuracy_metric = keras.metrics.BinaryAccuracy(
            name='train_accurary')

        self.checkpoint_dir = './models/idtr/training_checkpoints_' + \
            str(self.image_shape[0]) + '_' + str(self.tensor_length)
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              apex=self.apex,
                                              #   moex=self.moex,
                                              clsf=self.clsf)
        self.checkpoint.restore(
            tf.train.latest_checkpoint(self.checkpoint_dir))

    def formaliza_data(self, obj, frame):
        xmin = 0 if obj.bbox.xmin < 0 else obj.bbox.xmin
        xmax = 300 if obj.bbox.xmax > 300 else obj.bbox.xmax
        ymin = 0 if obj.bbox.ymin < 0 else obj.bbox.ymin
        ymax = 300 if obj.bbox.ymax > 300 else obj.bbox.ymax
        box = [xmin/300, ymin/300, xmax/300, ymax/300]
        if xmin == xmax:
            return np.zeros(self.image_shape)
        if ymin == ymax:
            return np.zeros(self.image_shape)
        cropped_obj_img = frame[ymin:ymax, xmin:xmax]
        resized_obj_img = cv.resize(cropped_obj_img, self.image_shape)
        obj_img = resized_obj_img/255.0
        return box, obj_img

    @tf.function
    def train_step(self, bboxes, imgs, labels):
        with tf.GradientTape() as tape:
            app_features = self.apex(imgs)
            # mot_features = self.moex(bboxes)
            output = self.clsf(app_features)
            # output = self.clsf(app_features, mot_features)
            predictions = tf.reshape(output, [-1])
            loss = self.loss(labels, predictions)

        variables = self.clsf.trainable_variables + self.apex.trainable_variables
        # variables = self.clsf.trainable_variables + \
        #     self.apex.trainable_variables + self.moex.trainable_variables
        print(variables)
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        self.loss_metric(loss)
        self.accuracy_metric(labels, predictions)
        return labels, predictions

    def train(self, dataset, epochs=10):
        for epoch in range(epochs):
            start = time.time()
            steps_per_epoch = 0
            iterator = iter(dataset)
            try:
                while True:
                    bboxes, imgs, labels = next(iterator)
                    steps_per_epoch += 1
                    self.train_step(bboxes, imgs, labels)
            except StopIteration:
                pass

            self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            end = time.time()
            print('Epoch {}'.format(epoch + 1))
            print('\tSteps per epoch: {}'.format(steps_per_epoch))
            print('\tLoss Metric {:.4f}'.format(self.loss_metric.result()))
            print('\tAccuracy Metric {:.4f}'.format(
                self.accuracy_metric.result()*100))
            print('\tTime taken for 1 epoch {:.4f} sec\n'.format(end - start))

            self.loss_metric.reset_states()
            self.accuracy_metric.reset_states()

    def predict(self, bboxes_batch, obj_imgs_batch):

        apexstart = time.time()
        app_features = self.apex(np.array(obj_imgs_batch))
        apexend = time.time()
        print('APEX estimated time {:.4f}'.format(apexend-apexstart))

        # moexstart = time.time()
        # mot_features = self.moex(np.array(bboxes_batch))
        # moexend = time.time()
        # print('MOEX estimated time {:.4f}'.format(moexend-moexstart))

        clsfstart = time.time()
        output = self.clsf(app_features)
        # output = self.clsf(app_features, mot_features)
        predictions = tf.reshape(output, [-1])
        clsfend = time.time()
        print('CLSF estimated time {:.4f}'.format(clsfend-clsfstart))

        return predictions, tf.math.argmax(predictions)
