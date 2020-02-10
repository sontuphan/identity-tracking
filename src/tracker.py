from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2 as cv

IMAGE_SHAPE = (96, 96)
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "../models/tpu/ohmnilabs_features_extractor_quant_postprocess.tflite")


class Extractor(keras.Model):
    def __init__(self):
        super(Extractor, self).__init__()
        self.conv = keras.applications.MobileNetV2(
            input_shape=(96, 96, 3), include_top=False, weights='imagenet')
        self.conv.trainable = False
        self.ga = keras.layers.GlobalAveragePooling2D()
        self.fc = keras.layers.Dense(256, activation='sigmoid')

    def call(self, imgs, bboxes):
        conv_output = self.conv(imgs)
        ga_output = self.ga(conv_output)
        fc_output = self.fc(ga_output)
        features = tf.concat([fc_output, bboxes], 1)
        return features


class Tracker:
    def __init__(self):
        self.batch_size = 64
        self.image_shape = IMAGE_SHAPE

        self.extractor = Extractor()
        self.optimizer = keras.optimizers.Adam()

        self.loss_metric = keras.metrics.Mean(name='train_loss')
        # self.accuracy_metric = keras.metrics.BinaryAccuracy(
        #     name='train_accurary')

        self.checkpoint_dir = './models/idtr/training_checkpoints_' + \
            str(self.image_shape[0]) + '_trilet'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              extractor=self.extractor)
        self.checkpoint.restore(
            tf.train.latest_checkpoint(self.checkpoint_dir))

    def loss_function(self, afs, pfs, nfs):
        # loss = ||afs-pfs||^2 - ||afs-nfs||^2 + a iff loss >= 0
        # loss = 0 iff loss < 0
        lloss = tf.linalg.normalize(afs - pfs, ord='euclidean', axis=1)
        rloss = tf.linalg.normalize(afs - nfs, ord='euclidean', axis=1)
        alpha = tf.fill([self.batch_size, 1], tf.constant(1, dtype=tf.float32))
        loss = lloss[1] - rloss[1] + alpha
        return tf.exp(loss)

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
    def train_step(self, anis, anbs, pis, pbs, nis, nbs):
        with tf.GradientTape() as tape:
            afs = self.extractor(anis, anbs)
            pfs = self.extractor(pis, pbs)
            nfs = self.extractor(nis, nbs)
            loss = self.loss_function(afs, pfs, nfs)

        variables = self.extractor.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        self.loss_metric(loss)
        # self.accuracy_metric(labels, predictions)
        return afs, pfs, nfs

    def train(self, dataset, epochs=10):
        for epoch in range(epochs):
            start = time.time()
            steps_per_epoch = 0
            iterator = iter(dataset)
            try:
                while True:
                    imgs, bboxes = next(iterator)
                    anis, pis, nis = tf.split(imgs, [1, 1, 1], axis=1)
                    anis = tf.reshape(
                        anis, [self.batch_size, self.image_shape[0], self.image_shape[1], 3])
                    pis = tf.reshape(
                        pis, [self.batch_size, self.image_shape[0], self.image_shape[1], 3])
                    nis = tf.reshape(
                        nis, [self.batch_size, self.image_shape[0], self.image_shape[1], 3])
                    anbs, pbs, nbs = tf.split(bboxes, [1, 1, 1], axis=1)
                    anbs = tf.reshape(anbs, [self.batch_size, 4])
                    pbs = tf.reshape(pbs, [self.batch_size, 4])
                    nbs = tf.reshape(nbs, [self.batch_size, 4])
                    steps_per_epoch += 1
                    self.train_step(anis, anbs, pis, pbs, nis, nbs)
            except StopIteration:
                pass

            self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            end = time.time()
            print('Epoch {}'.format(epoch + 1))
            print('\tSteps per epoch: {}'.format(steps_per_epoch))
            print('\tLoss Metric {:.4f}'.format(self.loss_metric.result()))
            # print('\tAccuracy Metric {:.4f}'.format(self.accuracy_metric.result()*100))
            print('\tTime taken for 1 epoch {:.4f} sec\n'.format(end - start))

            self.loss_metric.reset_states()
            # self.accuracy_metric.reset_states()

    def predict(self, imgs, bboxes):

        estart = time.time()
        vectors = self.extractor(np.array(imgs), np.array(bboxes))
        eend = time.time()
        print('Extractor estimated time {:.4f}'.format(eend-estart))

        return vectors
