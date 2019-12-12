from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import numpy as np

from utils import image

IMAGE_SHAPE = (96, 96)


class Classification():
    def __init__(self):
        self.batch_size = 32
        self.tensor_length = 1

        self.model = keras.Sequential([
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        self.feature_extractor = self.load_feature_extractor()

    def load_feature_extractor(self):
        feature_extractor_url = 'https://tfhub.dev/google/imagenet/mobilenet_v2_050_96/feature_vector/4'
        feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                                 input_shape=(IMAGE_SHAPE+(3,)))
        feature_extractor_layer.trainable = False
        return tf.keras.Sequential([feature_extractor_layer])

    def train(self, dataset):
        iterator = iter(dataset)

        try:
            while True:
                _, cnn_inputs, y = next(iterator)
                cnn_inputs = tf.reshape(
                    cnn_inputs, [self.batch_size*self.tensor_length, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3])
                logits = self.feature_extractor(cnn_inputs)
                x = tf.reshape(
                        logits, [self.batch_size, self.tensor_length, 1280])
                print(x)
        except StopIteration:
            pass
