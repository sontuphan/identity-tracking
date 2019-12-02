from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import os
import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from src.datamanufacture import DataManufacture


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


class Seq2Seq:
    def __init__(self):
        self.tensor_length = 28+1
        self.batch_size = 512
        self.encoder = Encoder(64, self.batch_size)
        self.decoder = Decoder(64, self.batch_size)
        self.optimizer = keras.optimizers.Adam()
        self.loss = keras.losses.BinaryCrossentropy()

        self.checkpoint_dir = './models/seq2seq/training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              encoder=self.encoder,
                                              decoder=self.decoder)

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

    def predict(self, inputs):
        self.checkpoint.restore(
            tf.train.latest_checkpoint(self.checkpoint_dir))

        tensor_inputs = tf.convert_to_tensor(inputs)
        encoder_input, decoder_input = tf.split(
            tensor_inputs, [self.tensor_length-1, 1], axis=1)
        encoder_state = tf.zeros((len(inputs), self.encoder.units))
        _, encoder_state = self.encoder(encoder_input, encoder_state)

        decoder_state = encoder_state
        predictions, _ = self.decoder(decoder_input, decoder_state)
        predictions = tf.reshape(predictions, [-1])
        return predictions.numpy(), tf.argmax(predictions).numpy()

    def sub_preprocess_data(self, data):
        historical_data = []
        for frame in data:
            if data.get(str(int(frame)+self.tensor_length)) is None:
                break
            vector = []
            for i in range(self.tensor_length):
                vector.append(data.get(str(int(frame)+i)))
            historical_data.append(vector)

        filtered_data = []
        for tensor in historical_data:
            re = []
            for objs in tensor:
                check = False
                for obj in objs:
                    if obj[1] == 1:
                        check = True
                        re.append([obj[4]/640, obj[5]/640,
                                   obj[6]/480, obj[7]/480])
                if check is False:
                    re.append([0., 0., 0., 0.])
            re.append([0, 1, 0, 0])
            filtered_data.append(re.copy())
            for obj in objs:
                re[-2] = [obj[4]/640, obj[5]/640,
                          obj[6]/480, obj[7]/480]
                re[-1] = [1, 0, 0, 0]
                filtered_data.append(re.copy())

        reduced_data = np.unique(filtered_data, axis=0)

        features = []
        labels = []
        for tensor in reduced_data:
            _features = []
            for index, value in enumerate(tensor):
                if index == self.tensor_length:
                    labels.append(value)
                else:
                    _features.append(value)
            features.append(_features)

        return features, labels

    def preprocess_data(self, name, max_id):
        dm = DataManufacture(name)
        features = []
        labels = []

        for id in range(1, max_id):
            data = dm.process_data(id)
            _features, _labels = self.sub_preprocess_data(data)
            features.extend(_features)

            binary_labels = []
            for label in _labels:
                y = 0 if label[0] == 1 else 1
                binary_labels.append([y])
            labels.extend(np.array(binary_labels, dtype=np.int))

        features = np.array(features, dtype=np.float)
        labels = np.array(labels, dtype=np.int)

        dataset = (features, labels)
        return dataset
