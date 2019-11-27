from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import time
from sklearn.model_selection import train_test_split


class Encoder(tf.keras.Model):
    def __init__(self, units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.units = units
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
        self.dense = tf.keras.layers.Dense(1024, activation='relu')
        self.classifier = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x, state):
        gru_output, hidden_state = self.gru(x, initial_state=state)
        dense_output = self.dense(gru_output)
        classifier_output = self.classifier(dense_output)
        return classifier_output, hidden_state


class Seq2Seq:
    def __init__(self):
        self.batch_size = 64
        self.encoder = Encoder(10, self.batch_size)
        self.decoder = Decoder(1, self.batch_size)
        self.optimizer = keras.optimizers.Adam()
        self.loss = keras.losses.BinaryCrossentropy()

    def loss_function(self, real, pred):
        loss = self.loss(real, pred)
        return tf.reduce_mean(loss)

    @tf.function
    def train_step(self, x, y, encoder_state):
        loss = 0
        with tf.GradientTape() as tape:
            _, encoder_state = self.encoder(x[:-1], encoder_state)
            decoder_state = encoder_state
            decoder_input = x[-1:]
            for t in range(len(y)):
                predictions, decoder_state = self.decoder(
                    decoder_input, decoder_state)
                loss += self.loss_function(y[t], predictions)
        batch_loss = (loss / len(y))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss

    def train(self, dataset, epochs=10):
        (input_tensor, target_tensor) = dataset
        input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(
            input_tensor, target_tensor, test_size=0.2)
        steps_per_epoch = len(input_tensor_train)//self.batch_size

        for epoch in range(epochs):
            start = time.time()
            init_state = self.encoder.initialize_hidden_state()
            total_loss = 0
            for (batch, (x, y)) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss = self.train_step(x, y, init_state)
                total_loss += batch_loss
                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(
                        epoch + 1, batch, batch_loss.numpy()))
            print('Epoch {} Loss {:.4f}'.format(
                epoch + 1, total_loss / steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
