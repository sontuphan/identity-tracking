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

tf.debugging.set_log_device_placement(True)

class Extractor(keras.Model):
    def __init__(self):
        super(Extractor, self).__init__()
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.conv = keras.applications.MobileNetV2(
                input_shape=(96, 96, 3), include_top=False, weights='imagenet')
            self.conv.trainable = False
            self.pool = keras.layers.GlobalAveragePooling2D()
            self.fc = keras.layers.Dense(512, activation='sigmoid')

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

    @tf.function
    def loss_function(self, afs, pfs, nfs):
        lloss = tf.sqrt(tf.reduce_sum(tf.square(afs - pfs), 1))
        rloss = tf.sqrt(tf.reduce_sum(tf.square(afs - nfs), 1))
        loss = tf.reduce_mean(tf.maximum(lloss - rloss + 10, 0))
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
    def train_step(self, anis, pis, nis):
        with tf.GradientTape() as tape:
            afs = self.extractor(anis)
            pfs = self.extractor(pis)
            nfs = self.extractor(nis)
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

                    ais, pis, nis = tf.split(imgs, [1, 1, 1], axis=1)
                    ais = tf.reshape(
                        ais, [self.batch_size, self.image_shape[0], self.image_shape[1], 3])
                    pis = tf.reshape(
                        pis, [self.batch_size, self.image_shape[0], self.image_shape[1], 3])
                    nis = tf.reshape(
                        nis, [self.batch_size, self.image_shape[0], self.image_shape[1], 3])
                    self.train_step(ais, pis, nis)

                    steps_per_epoch += 1

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
        self.frame_shape = (300, 300)
        self.image_shape = IMAGE_SHAPE
        self.interpreter = tflite.Interpreter(
            model_path=EDGE_MODEL,
            experimental_delegates=[
                tflite.load_delegate(EDGETPU_SHARED_LIB)
            ])
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.confidence = 0.7
        self.threshold = 8
        self.tradeoff = 10  # Between encoding distance and bbox distance
        self.prev_encoding = None
        self.prev_bbox = None

    def reset(self):
        self.prev_encoding = None
        self.prev_bbox = None

    def formaliza_data(self, obj, frame):
        xmin = 0 if obj.bbox.xmin < 0 else obj.bbox.xmin
        xmax = self.frame_shape[0] if obj.bbox.xmax > self.frame_shape[0] else obj.bbox.xmax
        ymin = 0 if obj.bbox.ymin < 0 else obj.bbox.ymin
        ymax = self.frame_shape[1] if obj.bbox.ymax > self.frame_shape[1] else obj.bbox.ymax
        box = np.array([xmin/self.frame_shape[0], ymin/self.frame_shape[1],
                        xmax/self.frame_shape[0], ymax/self.frame_shape[1]])
        if xmin == xmax:
            return np.zeros(self.image_shape)
        if ymin == ymax:
            return np.zeros(self.image_shape)
        cropped_obj_img = frame[ymin:ymax, xmin:xmax]
        resized_obj_img = cv.resize(cropped_obj_img, self.image_shape)
        obj_img = resized_obj_img/255.0
        return box, obj_img

    def confidence_level(self, distances):
        deltas = (self.threshold - distances)/self.threshold
        zeros = np.zeros(deltas.shape, dtype=np.float32)
        logits = np.maximum(deltas, zeros)
        logits_sum = np.sum(logits)
        if logits_sum == 0:
            return zeros
        else:
            confidences = logits/logits_sum
            return confidences

    def infer(self, img):
        img = np.array(img, dtype=np.float32)
        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(self.input_details[0]['index'], [img])
        self.interpreter.invoke()
        feature = self.interpreter.get_tensor(self.output_details[0]['index'])
        return np.array(feature[0])

    def predict(self, imgs, bboxes, init=False):
        if init:
            if len(imgs) != 1 or len(bboxes) != 1:
                raise ValueError('You must initialize one object only.')
            encoding = self.infer(imgs[0])
            self.prev_encoding = encoding
            self.prev_bbox = bboxes[0]
            return np.array([.0]), 0
        else:
            estart = time.time()
            encodings = []
            features = np.array([])
            positions = np.array([])

            for index, img in enumerate(imgs):
                encoding = self.infer(img)
                encodings.append(encoding)
                feature = np.linalg.norm(self.prev_encoding - encoding)
                features = np.append(features, feature)
                bbox = bboxes[index]
                position = np.linalg.norm(
                    self.prev_bbox - bbox) * self.tradeoff
                positions = np.append(positions, position)

            distances = features + positions
            confidences = self.confidence_level(distances)
            argmax = np.argmax(confidences)

            eend = time.time()
            print('Features:', features)
            print('Positions:', positions)
            print('Distances:', distances)
            print('Extractor estimated time {:.4f}'.format(eend-estart))
            if confidences[argmax] > self.confidence:
                self.prev_encoding = encodings[argmax]
                self.prev_bbox = bboxes[argmax]
                return confidences, argmax
            else:
                return confidences, None
