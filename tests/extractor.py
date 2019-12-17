from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from src.datamanufacture import DataManufacture
from utils import image

IMAGE_SHAPE = (96, 96)


class Extractor96(tf.keras.Model):
    def __init__(self):
        super(Extractor96, self).__init__()
        self.model = tf.keras.Sequential([
            hub.KerasLayer(
                'https://tfhub.dev/google/imagenet/mobilenet_v2_050_96/feature_vector/4',
                trainable=False,
                input_shape=(IMAGE_SHAPE+(3,))
            )
        ])

    def call(self, x):
        features = self.model(x)
        return features


def test_96():
    dm = DataManufacture()
    dataset = dm.gen_data_by_frame()
    objs = dataset[0]
    his_img = None
    for obj in objs:
        img = dm.load_frame(obj[2])
        obj = dm.convert_array_to_object(obj)
        cropped_img = image.crop(img, obj)
        resized_img = image.resize(cropped_img, IMAGE_SHAPE)
        img_arr = image.convert_pil_to_cv(resized_img)
        if his_img is None:
            his_img = img_arr
        else:
            his_img = np.concatenate((his_img, img_arr), axis=1)
    while True:
        cv.imshow('People', his_img)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    # extractor = Extractor96()
    # features = extractor()
