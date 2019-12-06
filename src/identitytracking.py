from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import numpy as np

from utils import image

IMAGE_SHAPE = (224, 224)


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
        self.tensor_length = 16
        self.batch_size = 32
        self.encoder = Encoder(64, self.batch_size)
        self.decoder = Decoder(64, self.batch_size)
        self.optimizer = keras.optimizers.Adam()
        self.loss = keras.losses.BinaryCrossentropy()

        self.checkpoint_dir = './models/idtr/training_checkpoints_1'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              encoder=self.encoder,
                                              decoder=self.decoder)

        self.checkpoint.restore(
            tf.train.latest_checkpoint(self.checkpoint_dir))
        self.feature_extractor = self.load_feature_extractor()

    def load_feature_extractor(self):
        feature_extractor_url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2'
        feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                                 input_shape=(IMAGE_SHAPE+(3,)))
        feature_extractor_layer.trainable = False
        return tf.keras.Sequential([feature_extractor_layer])

    def loss_function(self, real, pred):
        real = tf.reshape(real, [self.batch_size, 1])
        pred = tf.reshape(pred, [self.batch_size, 1])
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

    def train(self, dataset, steps_per_epoch, epochs=10):  # 74
        for epoch in range(epochs):
            start = time.time()
            init_state = self.encoder.initialize_hidden_state()
            total_loss = 0
            for (batch, (bbox, cnn_inputs, y)) in enumerate(dataset.take(steps_per_epoch)):
                cnn_inputs = tf.reshape(
                    cnn_inputs, [self.batch_size*self.tensor_length, 224, 224, 3])
                logits = self.feature_extractor(cnn_inputs)
                logits = tf.reshape(
                    logits, [self.batch_size, self.tensor_length, 1280])
                x = tf.concat([bbox, logits], 2)

                batch_loss = self.train_step(x, y, init_state)
                total_loss += batch_loss
                if batch % self.batch_size == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(
                        epoch + 1, batch, batch_loss.numpy()))
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)
            end = time.time()
            print('Epoch {} Loss {:.4f}'.format(
                epoch + 1, total_loss / steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(end - start))

    def predict(self, inputs):
        x = []
        for (obj, img) in inputs:
            start = time.time()
            bbox = np.array(
                [obj.bbox.xmin/640, obj.bbox.ymin/480, obj.bbox.xmax/640, obj.bbox.ymax/480])
            cropped_img = image.crop(img, obj)
            resized_img = image.resize(cropped_img, IMAGE_SHAPE)
            img_arr = image.convert_pil_to_cv(resized_img)/255.0
            logits = self.feature_extractor(np.array([img_arr]))
            logits = tf.reshape(logits, [1280])
            rnn_cell_input = tf.concat([bbox, logits], 0)
            x.append(rnn_cell_input)
            end = time.time()
            print('Estimated time for a iteraction: {:.4f} sec'.format(end-start))
        x = tf.stack([x])

        (input_len, _, _) = x.shape
        encoder_input, decoder_input = tf.split(
            x, [self.tensor_length-1, 1], axis=1)
        encoder_state = tf.zeros((input_len, self.encoder.units))
        _, encoder_state = self.encoder(encoder_input, encoder_state)
        decoder_state = encoder_state
        predictions, _ = self.decoder(decoder_input, decoder_state)
        predictions = tf.reshape(predictions, [-1])
        return predictions
