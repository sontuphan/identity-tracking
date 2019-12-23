from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import numpy as np

from utils import image

IMAGE_SHAPE = (96, 96)


class FeaturesExtractor(keras.Model):
    def __init__(self, tensor_length, units):
        super(FeaturesExtractor, self).__init__()
        self.units = units
        self.tensor_length = tensor_length
        self.extractor = hub.KerasLayer(
            'https://tfhub.dev/google/imagenet/mobilenet_v2_050_96/feature_vector/4',
            trainable=False,
            input_shape=(IMAGE_SHAPE+(3,))
        )
        self.fc = keras.layers.Dense(self.units, activation='relu')

    def call(self, x):
        (batch_size, _, _, _, _) = x.shape
        cnn_inputs = tf.reshape(
            x, [batch_size*self.tensor_length, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3])
        logits = self.extractor(cnn_inputs)
        fc_output = self.fc(logits)
        features = tf.reshape(
            fc_output, [batch_size, self.tensor_length, self.units])
        return features


class MovementExtractor(keras.Model):
    def __init__(self, tensor_length, units):
        super(MovementExtractor, self).__init__()
        self.units = units
        self.tensor_length = tensor_length
        self.fc = keras.layers.Dense(self.units, activation='relu')

    def call(self, x):
        (input_size, _, _) = x.shape
        dim_inputs = tf.reshape(x, [input_size*self.tensor_length, 4])
        fc_output = self.fc(dim_inputs)
        features = tf.reshape(
            fc_output, [input_size, self.tensor_length, self.units])
        return features


class Encoder(keras.Model):
    def __init__(self, units):
        super(Encoder, self).__init__()
        self.units = units
        # Recall: in gru cell, h = c
        self.gru = keras.layers.GRU(self.units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')

    def call(self, x, state):
        _, hidden_state = self.gru(x, initial_state=state)
        return hidden_state

    def initialize_hidden_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


class Decoder(keras.Model):
    def __init__(self, units):
        super(Decoder, self).__init__()
        self.gru_units = units[0]
        self.dense_units = units[1]
        self.gru = keras.layers.GRU(self.gru_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        self.fc = keras.layers.Dense(self.dense_units, activation='relu')
        self.classifier = keras.layers.Dense(1, activation='sigmoid')

    def call(self, x, state):
        gru_output, _ = self.gru(x, initial_state=state)
        batch_size, _, _ = gru_output.shape
        gru_output = tf.reshape(gru_output, [batch_size, -1])
        fc_output = self.fc(gru_output)
        classifier_output = self.classifier(fc_output)
        classifier_output = tf.reshape(classifier_output, [-1])
        return classifier_output


class IdentityTracking:
    def __init__(self):
        self.tensor_length = 8
        self.batch_size = 64
        self.image_shape = IMAGE_SHAPE
        self.encoder = Encoder(256)
        self.decoder = Decoder([256, 128])
        self.fextractor = FeaturesExtractor(self.tensor_length, 256)
        self.mextractor = MovementExtractor(self.tensor_length, 128)

        self.optimizer = keras.optimizers.Adam()
        self.loss = keras.losses.BinaryCrossentropy()

        self.loss_metric = keras.metrics.Mean(name='train_loss')
        self.accuracy_metric = keras.metrics.BinaryAccuracy(
            name='train_accurary')

        self.checkpoint_dir = './models/idtr/training_checkpoints_' + \
            str(self.image_shape[0]) + '_' + str(self.tensor_length)
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              fextractor=self.fextractor,
                                              mextractor=self.mextractor,
                                              encoder=self.encoder,
                                              decoder=self.decoder)
        self.checkpoint.restore(
            tf.train.latest_checkpoint(self.checkpoint_dir))

    @tf.function
    def train_step(self, bboxes, cnn_inputs, labels, encoder_state):
        with tf.GradientTape() as tape:
            mov_features = self.mextractor(bboxes)
            cnn_features = self.fextractor(cnn_inputs)
            x = tf.concat([mov_features, cnn_features], 2)
            encoder_input, decoder_input = tf.split(
                x, [self.tensor_length-1, 1], axis=1)
            decoder_state = self.encoder(encoder_input, encoder_state)
            predictions = self.decoder(decoder_input, decoder_state)
            loss = self.loss(labels, predictions)
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables + \
            self.mextractor.trainable_variables + self.fextractor.trainable_variables
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
            init_state = self.encoder.initialize_hidden_state(self.batch_size)

            try:
                while True:
                    bboxes, cnn_inputs, labels = next(iterator)
                    steps_per_epoch += 1
                    self.train_step(bboxes, cnn_inputs, labels, init_state)
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

        mov_features = self.mextractor(bboxes)
        cnn_features = self.fextractor(imgs)
        x = tf.concat([mov_features, cnn_features], 2)
        encoder_input, decoder_input = tf.split(
            x, [self.tensor_length-1, 1], axis=1)
        batch_inputs, _, _ = encoder_input.shape
        init_state = self.encoder.initialize_hidden_state(batch_inputs)
        hidden_state = self.encoder(encoder_input, init_state)
        predictions = self.decoder(decoder_input, hidden_state)

        return predictions, tf.math.argmax(predictions)
