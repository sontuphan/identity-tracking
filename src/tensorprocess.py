from __future__ import absolute_import, division, print_function, unicode_literals

import os
import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers

from src.datamanufacture import DataManufacture

IMAGE_SHAPE = (224, 224)
AUTOTUNE = tf.data.experimental.AUTOTUNE


class TensorProcess():
    def __init__(self):
        self.data_dir = 'data/train/'

    def process_data(self):
        sets = next(os.walk(self.data_dir))[1]
        for name in sets:
            dm = DataManufacture(name)
            dataset = dm.load_data()
            for frame in dataset:
                print('{} ================================='.format(frame))
                objs = dataset.get(frame)
                for obj in objs:
                    print(obj)
                    image, _, _ = self.process_path(
                        'data/train/{}/{}/{}.jpg'.format(name, frame, obj[0]))
                    print('Image shape:', image.numpy().shape)

    def get_frame_no(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        return parts[-3], parts[-2]

    def decode_img(self, img):
        (img_width, img_height) = IMAGE_SHAPE
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, [img_width, img_height])

    def process_path(self, file_path):
        source, frame = self.get_frame_no(file_path)
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, frame, source
