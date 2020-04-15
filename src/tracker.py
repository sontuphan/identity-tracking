from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import datetime
import tensorflow as tf
from tensorflow import keras, lite
import tflite_runtime.interpreter as tflite
import numpy as np
import cv2 as cv

IMAGE_SHAPE = (160, 160)
EDGETPU_SHARED_LIB = 'libedgetpu.so.1'
CHECKPOINT = './models/idtr/training_checkpoints_' + \
    str(IMAGE_SHAPE[0]) + '_trilet'
MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "../models/tpu/ohmnilabs_features_extractor_quant_postprocess.tflite")
EDGE_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "../models/tpu/ohmnilabs_features_extractor_quant_postprocess_edgetpu.tflite")
LOGS_DIR = './logs/'


def formaliza_data(obj, frame):
    (height, width, _) = frame.shape

    xmin = int(obj[-4]*width)
    xmin = 0 if xmin < 0 else xmin
    xmin = width if xmin > width else xmin

    ymin = int(obj[-3]*height)
    ymin = 0 if ymin < 0 else ymin
    ymin = height if ymin > height else ymin

    xmax = int(obj[-2]*width)
    xmax = 0 if xmax < 0 else xmax
    xmax = width if xmax > width else xmax

    ymax = int(obj[-1]*height)
    ymax = 0 if ymax < 0 else ymax
    ymax = height if ymax > height else ymax

    box = np.array([xmin, ymin, xmax, ymax], dtype=np.int32)
    if xmin >= xmax or ymin >= ymax:
        return np.zeros(IMAGE_SHAPE), box

    cropped_obj_img = frame[ymin:ymax, xmin:xmax]
    resized_obj_img = cv.resize(cropped_obj_img, IMAGE_SHAPE)
    obj_img = np.array(resized_obj_img/127.5 - 1, dtype=np.float32)
    return obj_img, box


class Extractor(keras.Model):
    def __init__(self):
        super(Extractor, self).__init__()
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.conv = keras.applications.MobileNetV2(
                input_shape=(IMAGE_SHAPE + (3,)),
                include_top=False,
                weights='imagenet'
            )
            self.conv.trainable = False
            self.pool = keras.layers.GlobalAveragePooling2D()
            self.fc = keras.layers.Dense(512)

    def call(self, imgs):
        conv_output = self.conv(imgs)
        pool_output = self.pool(conv_output)
        fc_output = self.fc(pool_output)
        return fc_output


class Tracker:
    def __init__(self):
        self.batch_size = 128
        self.image_shape = IMAGE_SHAPE

        # Setup model
        self.extractor = Extractor()
        # self.optimizer = keras.optimizers.Adam()
        self.optimizer = keras.optimizers.RMSprop(learning_rate=0.0001)
        self.loss_metric = keras.metrics.Mean(name='train_loss')

        # Setup checkpoints
        self.checkpoint_dir = CHECKPOINT
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              extractor=self.extractor)
        self.checkpoint.restore(
            tf.train.latest_checkpoint(self.checkpoint_dir))

        # Setup logs (tensorboard)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = LOGS_DIR + 'triplets/' + current_time + '/train'
        self.train_log_writer = tf.summary.create_file_writer(train_log_dir)

    @tf.function
    def loss_function(self, afs, pfs, nfs):
        lloss = tf.sqrt(tf.reduce_sum(tf.square(afs - pfs), 1))
        rloss = tf.sqrt(tf.reduce_sum(tf.square(afs - nfs), 1))
        loss = tf.reduce_mean(tf.maximum(lloss - rloss + 30., 0.))
        return loss

    @tf.function
    def train_step(self, ais, pis, nis):
        with tf.GradientTape() as tape:
            afs = self.extractor(ais)
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

            # Training
            steps_per_epoch = 0
            iterator = iter(dataset)
            try:
                while True:
                    imgs, _ = next(iterator)
                    ais, pis, nis = tf.split(imgs, [1, 1, 1], axis=1)
                    ais = tf.reshape(
                        ais, ((self.batch_size,) + self.image_shape + (3,)))
                    pis = tf.reshape(
                        pis, ((self.batch_size,) + self.image_shape + (3,)))
                    nis = tf.reshape(
                        nis, ((self.batch_size,) + self.image_shape + (3,)))
                    self.train_step(ais, pis, nis)
                    # Logs
                    with self.train_log_writer.as_default():
                        tf.summary.scalar(
                            'epoch'+str(epoch),
                            self.loss_metric.result(),
                            step=steps_per_epoch
                        )
                    steps_per_epoch += 1
            except StopIteration:
                pass
            # Checkpoint
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            end = time.time()
            print('Epoch {}'.format(epoch + 1))
            print('\tSteps per epoch: {}'.format(steps_per_epoch))
            print('\tLoss Metric {:.4f}'.format(self.loss_metric.result()))
            print('\tTime taken for 1 epoch {:.4f} sec\n'.format(end - start))

            self.loss_metric.reset_states()

    def extract_features(self, imgs):
        estart = time.time()
        features = self.extractor(imgs)
        eend = time.time()
        print('Extractor estimated time {:.4f}'.format(eend-estart))
        return features

    def convert(self, pipeline):
        model = self.extractor

        # Define input shapes
        pseudo_imgs = tf.keras.Input(shape=(self.image_shape+(3,)))
        model(pseudo_imgs)
        model.summary()

        def representative_dataset_gen():
            for tensor in pipeline.take(1):
                imgs, _ = tensor
                imgs, _, _ = tf.split(imgs, [1, 1, 1], axis=1)
                imgs = tf.reshape(
                    imgs, ((self.batch_size,) + self.image_shape + (3,)))
                img = np.array([imgs[0]])
                yield [img]  # Shape (1, 1, height, width, channel)

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
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.confidence = 0.7
        self.threshold = 35
        self.prev_encoding = None
        self.prev_bbox = None

    def reset(self):
        self.prev_encoding = None
        self.prev_bbox = None

    def __area(self, box):
        return max(0, box[3]-box[1]+1)*max(0, box[2]-box[0]+1)

    def __confidence_level(self, distances):
        if len(distances) == 0:
            return np.array([]), None
        deltas = (self.threshold - distances)/self.threshold
        zeros = np.zeros(deltas.shape, dtype=np.float32)
        logits = np.maximum(deltas, zeros)
        logits_sum = np.sum(logits)
        if logits_sum == 0:
            return zeros, None
        else:
            confidences = logits/logits_sum
            return confidences, np.argmax(confidences)

    def iou(self, anchor_box, predicted_box):
        xmin = max(anchor_box[0], predicted_box[0])
        ymin = max(anchor_box[1], predicted_box[1])
        xmax = min(anchor_box[2], predicted_box[2])
        ymax = min(anchor_box[3], predicted_box[3])
        inter_box = np.array([xmin, ymin, xmax, ymax])
        inter_area = self.__area(inter_box)
        anchor_area = self.__area(anchor_box)
        predicted_area = self.__area(predicted_box)
        return inter_area/float(anchor_area+predicted_area-inter_area)

    def infer(self, img):
        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(self.input_details[0]['index'], [img])
        self.interpreter.invoke()
        feature = self.interpreter.get_tensor(self.output_details[0]['index'])
        return np.array(feature[0], dtype=np.float32)

    def set_anchor(self, img, bbox):
        encoding = self.infer(img)
        self.prev_encoding = encoding
        self.prev_bbox = bbox
        return np.array([.0]), 0

    def predict(self, imgs, bboxes):
        estart = time.time()

        differentials = np.array([])
        indice = []
        encodings = []
        ious = []
        for index, box in enumerate(bboxes):
            iou = self.iou(self.prev_bbox, box)
            ious.append(iou)
            if iou > 0.5:
                img = imgs[index]
                encoding = self.infer(img)
                differential = np.linalg.norm(
                    self.prev_encoding - encoding)

                differentials = np.append(differentials, differential)
                indice.append(index)
                encodings.append(encoding)

        print('Differentials:', differentials)

        confidences, argmax = self.__confidence_level(differentials)
        index = indice[argmax] if argmax is not None else None

        eend = time.time()
        print('Extractor estimated time {:.4f}'.format(eend-estart))
        if argmax is not None and confidences[argmax] > self.confidence:
            self.prev_encoding = encodings[argmax]
            self.prev_bbox = bboxes[index]
        return confidences, index
