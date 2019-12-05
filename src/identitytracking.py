from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import os
import time
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import numpy as np
import cv2 as cv

from src.datamanufacture import DataManufacture
from src.humandetection import HumanDetection

IMAGE_SHAPE = (224, 224)


class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        self.model.reset_metrics()


class Encoder(tf.keras.Model):
    def __init__(self, units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.units = units
        # Recall: in gru cell, h = c
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, state):
        output, hidden_state = self.gru(x, initial_state=state)
        return output, hidden_state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.units))


class Decoder(tf.keras.Model):
    def __init__(self, units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.units = units
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.dense = tf.keras.layers.Dense(512, activation='relu')
        self.dense = tf.keras.layers.Dense(128, activation='relu')
        self.dense = tf.keras.layers.Dense(32, activation='relu')
        self.classifier = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x, state):
        gru_output, hidden_state = self.gru(x, initial_state=state)
        dense_output = self.dense(gru_output)
        classifier_output = self.classifier(dense_output)
        return classifier_output, hidden_state


class IdentityTracking:
    def __init__(self):
        self.tensor_length = 32
        self.batch_size = 512
        self.encoder = Encoder(64, self.batch_size)
        self.decoder = Decoder(64, self.batch_size)
        self.optimizer = keras.optimizers.Adam()
        self.loss = keras.losses.BinaryCrossentropy()

        self.data_dir = 'data/train/MOT17-05/'
        self.cnn_file = self.data_dir + 'cnn.txt'
        self.rnn_file = self.data_dir + 'rnn.txt'
        self.labels_file = self.data_dir + 'labels.txt'

        self.checkpoint_dir = './models/idtr/training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              encoder=self.encoder,
                                              decoder=self.decoder)

    def load_data(self):
        labels = tf.data.TextLineDataset(self.labels_file)
        cnn_inputs = tf.data.TextLineDataset(self.cnn_file)
        rnn_inputs = tf.data.TextLineDataset(self.rnn_file)

        dataset = tf.data.Dataset.zip((labels, cnn_inputs, rnn_inputs))
        return dataset

    def load_feature_extractor(self):
        feature_extractor_url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2'
        feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                                 input_shape=(224, 224, 3))
        feature_extractor_layer.trainable = False

        return tf.keras.Sequential([feature_extractor_layer])

    def loss_function(self, real, pred):
        loss = self.loss(real, pred)
        return tf.reduce_mean(loss)

    @tf.function
    def train_step(self, x, y, encoder_state):
        with tf.GradientTape() as tape:
            encoder_input, decoder_input = tf.split(
                x, [self.tensor_length-1, 1], axis=1)
            _, encoder_state = self.encoder(encoder_input, encoder_state)

            decoder_state = encoder_state
            predictions, _ = self.decoder(decoder_input, decoder_state)
            loss = self.loss_function(y, predictions)
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def train(self, dataset, epochs=10):
        (input_tensor, target_tensor) = dataset
        input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(
            input_tensor, target_tensor, test_size=0.2)
        steps_per_epoch = len(input_tensor_train)//self.batch_size

        train_data = tf.data.Dataset.from_tensor_slices(
            (input_tensor_train, target_tensor_train)).shuffle(len(input_tensor_train))
        train_data = train_data.batch(self.batch_size, drop_remainder=True)

        val_data = tf.data.Dataset.from_tensor_slices(
            (input_tensor_val, target_tensor_val)).shuffle(len(input_tensor_val))
        val_data = val_data.batch(self.batch_size, drop_remainder=True)

        for epoch in range(epochs):
            start = time.time()
            init_state = self.encoder.initialize_hidden_state()
            total_loss = 0
            for (batch, (x, y)) in enumerate(train_data.take(steps_per_epoch)):
                batch_loss = self.train_step(x, y, init_state)
                total_loss += batch_loss
                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(
                        epoch + 1, batch, batch_loss.numpy()))
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)
            print('Epoch {} Loss {:.4f}'.format(
                epoch + 1, total_loss / steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    def predict(self):
        pass
