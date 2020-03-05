from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import tensorflow as tf
from tensorflow import keras, lite
import tflite_runtime.interpreter as tflite
import numpy as np
import cv2 as cv

IMAGE_SHAPE = (96, 96)
EDGETPU_SHARED_LIB = 'libedgetpu.so.1'
CHECKPOINT = './models/idtr/training_checkpoints_' + \
    str(IMAGE_SHAPE[0]) + '_trilet'
MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "../models/tpu/ohmnilabs_features_extractor_quant_postprocess.tflite")
EDGE_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "../models/tpu/ohmnilabs_features_extractor_quant_postprocess_edgetpu.tflite")


class Extractor(keras.Model):
    def __init__(self):
        super(Extractor, self).__init__()
        self.conv = keras.applications.MobileNetV2(
            input_shape=(96, 96, 3), include_top=False, weights='imagenet')
        self.conv.trainable = False
        self.pool = keras.layers.GlobalMaxPool2D()
        self.fc = keras.layers.Dense(256, activation='sigmoid')

    def call(self, imgs):
        conv_output = self.conv(imgs)
        pool_output = self.pool(conv_output)
        fc_output = self.fc(pool_output)
        return fc_output


class Tracker:
    def __init__(self):
        self.batch_size = 64
        self.image_shape = IMAGE_SHAPE

        self.extractor = Extractor()
        self.optimizer = keras.optimizers.Adam()

        self.loss_metric = keras.metrics.Mean(name='train_loss')

        self.checkpoint_dir = CHECKPOINT
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              extractor=self.extractor)
        self.checkpoint.restore(
            tf.train.latest_checkpoint(self.checkpoint_dir))

    def loss_function(self, afs, pfs, nfs):
        # loss = (||afs-pfs||^2 + 1) / (||afs-nfs||^2 + 1)
        lloss = tf.linalg.normalize(afs - pfs, ord='euclidean', axis=1)
        rloss = tf.linalg.normalize(afs - nfs, ord='euclidean', axis=1)
        one = tf.fill([self.batch_size, 1], tf.constant(1, dtype=tf.float32))
        loss = (lloss[1] + one)/(rloss[1] + one)
        return loss

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
            _afs = self.extractor(anis)
            afs = tf.concat([_afs, anbs], 1)
            _pfs = self.extractor(pis)
            pfs = tf.concat([_pfs, pbs], 1)
            _nfs = self.extractor(nis)
            nfs = tf.concat([_nfs, nbs], 1)
            loss = self.loss_function(afs, pfs, nfs)

        variables = self.extractor.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        self.loss_metric(loss)
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
            print('\tTime taken for 1 epoch {:.4f} sec\n'.format(end - start))

            self.loss_metric.reset_states()

    def predict(self, imgs, bboxes):
        estart = time.time()
        imgs = np.array(imgs)
        bboxes = np.array(bboxes)
        features = self.extractor(imgs)
        vectors = np.concatenate((features, bboxes), axis=1)
        eend = time.time()
        print('Extractor estimated time {:.4f}'.format(eend-estart))
        return vectors

    def convert(self, pipeline):
        model = self.extractor

        # Define input shapes
        pseudo_imgs = tf.keras.Input(shape=(96, 96, 3))
        model(pseudo_imgs)
        model.summary()

        def representative_dataset_gen():
            for tensor in pipeline.take(100):
                imgs, _ = tensor
                img, _, _ = tf.split(imgs, [1, 1, 1], axis=0)
                yield [img]

        converter = lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [
            lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        tflite_quant_model = converter.convert()

        open(MODEL, 'wb').write(tflite_quant_model)


class Inference:
    def __init__(self):
        self.image_shape = IMAGE_SHAPE
        self.interpreter = tflite.Interpreter(
            model_path=EDGE_MODEL,
            experimental_delegates=[
                tflite.load_delegate(EDGETPU_SHARED_LIB)
            ])

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

    def predict(self, imgs, bboxes):
        estart = time.time()
        features = None
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        for img in imgs:
            img = np.array(img, dtype=np.float32)
            self.interpreter.allocate_tensors()
            self.interpreter.set_tensor(input_details[0]['index'], [img])
            self.interpreter.invoke()
            feature = self.interpreter.get_tensor(output_details[0]['index'])
            if features is None:
                features = feature
            else:
                features = np.append(features, feature, axis=0)

        bboxes = np.array(bboxes, dtype=np.float32)
        output = np.concatenate((features, bboxes), axis=1)
        eend = time.time()
        print('Extractor estimated time {:.4f}'.format(eend-estart))

        return output
