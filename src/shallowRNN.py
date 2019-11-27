from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from utils.bbox import Object, BBox
from src.datamanufacture import DataManufacture

PADDING_OBJECT = Object(id=0, frame=0, label=0, score=1,
                        bbox=BBox(xmin=0, ymin=0, xmax=0, ymax=0))


class ShallowRNN():
    def __init__(self):
        self.tensor_length = 11
        self.buffer_size = 10000
        self.batch_size = 64

    def plot_graphs(self, history, string):
        plt.plot(history.history[string])
        plt.plot(history.history['val_'+string], '')
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_'+string])
        plt.show()

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.RNN(tf.keras.layers.LSTMCell(10)),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(2, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model

    def sub_preprocess_data(self, data):
        historical_data = []
        for frame in data:
            if data.get(frame+self.tensor_length) is None:
                break
            vector = []
            for i in range(self.tensor_length):
                vector.append(data.get(frame+i))
            historical_data.append(vector)

        def centroid_map(tensor):
            re = []
            for objs in tensor:
                check = False
                for obj in objs:
                    if obj.label == 1:
                        check = True
                        re.append([(obj.bbox.xmin+obj.bbox.xmax)/(2*640),
                                   (obj.bbox.ymin+obj.bbox.ymax)/(2*480)])
                if check is False:
                    re.append([0, 0])
            return re
        filtered_data = map(centroid_map, historical_data)

        features = []
        labels = []
        for tensor in filtered_data:
            _features = []
            for index, centroid in enumerate(tensor):
                if index == self.tensor_length-1:
                    labels.append(centroid)
                else:
                    _features.append(centroid)
            features.append(_features)

        return features, labels

    def preprocess_data(self, name):
        dm = DataManufacture(name)
        features = []
        labels = []

        for id in range(1, 100):
            data = dm.load_data(id)
            _features, _labels = self.sub_preprocess_data(data)
            features.extend(_features)
            labels.extend(_labels)

        features = np.array(features, dtype=np.float)
        labels = np.array(labels, dtype=np.float)

        dataset = tf.data.Dataset.from_tensor_slices(
            (features, labels))
        return dataset

        # (train_examples, train_labels) = (features[:250], labels[:250])
        # (test_examples, test_labels) = (features[250:], labels[250:])
        # train_dataset = tf.data.Dataset.from_tensor_slices(
        #     (train_examples, train_labels))
        # test_dataset = tf.data.Dataset.from_tensor_slices(
        #     (test_examples, test_labels))

        # return (train_dataset, test_dataset)
