from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import numpy as np

from utils import image

IMAGE_SHAPE = (96, 96)


class FeaturesExtractor(tf.keras.Model):
    def __init__(self, tensor_length):
        super(FeaturesExtractor, self).__init__()
        self.tensor_length = tensor_length
        self.model = tf.keras.Sequential([
            hub.KerasLayer(
                'https://tfhub.dev/google/imagenet/mobilenet_v2_050_96/feature_vector/4',
                trainable=False,
                input_shape=(IMAGE_SHAPE+(3,))
            ),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu')
        ])

    def call(self, x):
        (batch_size, _, _, _, _) = x.shape
        cnn_inputs = tf.reshape(
            x, [batch_size*self.tensor_length, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3])
        logits = self.model(cnn_inputs)
        features = tf.reshape(
            logits, [batch_size, self.tensor_length, 512])
        return features


class DimensionExtractor(tf.keras.Model):
    def __init__(self, tensor_length):
        super(DimensionExtractor, self).__init__()
        self.tensor_length = tensor_length
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(1024, activation='relu',
                                  input_shape=(self.tensor_length, 4)),
            tf.keras.layers.Dense(512, activation='relu')
        ])

    def call(self, x):
        (batch_size, _, _) = x.shape
        cnn_inputs = tf.reshape(
            x, [batch_size*self.tensor_length, 4])
        logits = self.model(cnn_inputs)
        features = tf.reshape(
            logits, [batch_size, self.tensor_length, 512])
        return features


class Encoder(tf.keras.Model):
    def __init__(self, units):
        super(Encoder, self).__init__()
        self.units = units
        # Recall: in gru cell, h = c
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, state):
        _, hidden_state = self.gru(x, initial_state=state)
        return hidden_state

    def initialize_hidden_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


class Decoder(tf.keras.Model):
    def __init__(self, units):
        super(Decoder, self).__init__()
        self.units = units
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.dense = tf.keras.layers.Dense(512, activation='relu')
        self.dense = tf.keras.layers.Dense(256, activation='relu')
        self.classifier = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x, state):
        gru_output, _ = self.gru(x, initial_state=state)
        dense_output = self.dense(gru_output)
        classifier_output = self.classifier(dense_output)
        return classifier_output


class IdentityTracking:
    def __init__(self):
        self.tensor_length = 8
        self.batch_size = 64
        self.image_shape = IMAGE_SHAPE
        self.encoder = Encoder(512)
        self.decoder = Decoder(512)
        self.fextractor = FeaturesExtractor(self.tensor_length)
        self.dextractor = DimensionExtractor(self.tensor_length)
        self.optimizer = keras.optimizers.Adam()
        self.loss = keras.losses.BinaryCrossentropy()

        self.loss_metric = tf.keras.metrics.Mean(name='train_loss')
        self.accuracy_metric = tf.keras.metrics.BinaryAccuracy(
            name='train_accurary')

        self.checkpoint_dir = './models/idtr/training_checkpoints_' + \
            str(self.image_shape[0]) + '_' + str(self.tensor_length)
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              fextractor=self.fextractor,
                                              dextractor=self.dextractor,
                                              encoder=self.encoder,
                                              decoder=self.decoder)
        self.checkpoint.restore(
            tf.train.latest_checkpoint(self.checkpoint_dir))

    def loss_function(self, real, pred):
        real = tf.reshape(real, [self.batch_size, 1])
        pred = tf.reshape(pred, [self.batch_size, 1])
        loss = self.loss(real, pred)
        return loss

    def metrics(self, loss, real, pred):
        real = tf.reshape(real, [self.batch_size, 1])
        pred = tf.reshape(pred, [self.batch_size, 1])
        self.loss_metric(loss)
        self.accuracy_metric(real, pred)

    @tf.function
    def train_step(self, bboxes, cnn_inputs, labels, encoder_state):
        with tf.GradientTape() as tape:
            dimension_features = self.dextractor(bboxes)
            cnn_features = self.fextractor(cnn_inputs)
            x = tf.concat([dimension_features, cnn_features], 2)
            encoder_input, decoder_input = tf.split(
                x, [self.tensor_length-1, 1], axis=1)
            encoder_state = self.encoder(encoder_input, encoder_state)
            decoder_state = encoder_state
            predictions = self.decoder(decoder_input, decoder_state)
            loss = self.loss_function(labels, predictions)
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        self.metrics(loss, labels, predictions)

    def train(self, dataset, epochs=10):
        for epoch in range(epochs):

            start = time.time()
            steps_per_epoch = 0

            iterator = iter(dataset)
            init_state = self.encoder.initialize_hidden_state(self.batch_size)

            try:
                while True:
                    bboxes, cnn_inputs, labels = next(iterator)
                    steps_per_epoch += 1
                    self.train_step(bboxes, cnn_inputs, labels, init_state)
            except StopIteration:
                pass

            if (epoch+1) % 2 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            end = time.time()
            print('Epoch {}'.format(epoch + 1))
            print('\tSteps per epoch: {}'.format(steps_per_epoch))
            print('\tLoss Metric {:.4f}'.format(self.loss_metric.result()))
            print('\tAccuracy Metric {:.4f}'.format(
                self.accuracy_metric.result()*100))
            print('\tTime taken for 1 epoch {} sec\n'.format(end - start))

    def predict(self, inputs):
        bboxes = []
        imgs = []
        for row in inputs:
            bbox_tensor = []
            img_tensor = []
            for (obj, img) in row:
                bbox = np.array(
                    [obj.bbox.xmin/640, obj.bbox.ymin/480, obj.bbox.xmax/640, obj.bbox.ymax/480])
                cropped_img = image.crop(img, obj)
                resized_img = image.resize(cropped_img, self.image_shape)
                img_arr = image.convert_pil_to_cv(resized_img)/255.0
                bbox_tensor.append(bbox)
                img_tensor.append(img_arr)
            bboxes.append(bbox_tensor)
            imgs.append(img_tensor)
        bboxes = tf.stack(bboxes)
        imgs = tf.stack(imgs)

        dimension_features = self.dextractor(bboxes)
        cnn_features = self.fextractor(imgs)
        x = tf.concat([dimension_features, cnn_features], 2)
        (batch_inputs, _, _) = x.shape
        init_state = self.encoder.initialize_hidden_state(batch_inputs)
        encoder_input, decoder_input = tf.split(
            x, [self.tensor_length-1, 1], axis=1)
        hidden_state = self.encoder(encoder_input, init_state)
        predictions = self.decoder(decoder_input, hidden_state)
        predictions = tf.reshape(predictions, [-1])
        return predictions, tf.math.argmax(predictions)
