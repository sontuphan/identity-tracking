from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.factory import Factory
from utils import image


class Extractor96(tf.keras.Model):
    def __init__(self):
        super(Extractor96, self).__init__()
        self.model = tf.keras.Sequential([
            hub.KerasLayer(
                'https://tfhub.dev/google/imagenet/mobilenet_v2_050_96/feature_vector/4',
                trainable=False,
                input_shape=((96, 96, 3))
            )
        ])

    def call(self, x):
        return self.model(x)


class Extractor224(tf.keras.Model):
    def __init__(self):
        super(Extractor224, self).__init__()
        self.model = tf.keras.Sequential([
            hub.KerasLayer(
                'https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2',
                trainable=False,
                input_shape=((224, 224, 3))
            )
        ])

    def call(self, x):
        return self.model(x)


class ExtractorInception(tf.keras.Model):
    def __init__(self):
        super(ExtractorInception, self).__init__()
        image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                        weights='imagenet')
        new_input = image_model.input
        hidden_layer = image_model.layers[-1].output
        self.model = tf.keras.Model(new_input, hidden_layer)

    def call(self, x):
        return self.model(x)


def test_96():
    IMAGE_SHAPE = (96, 96)
    fac = Factory(img_shape=IMAGE_SHAPE)
    dataset = fac.gen_frames()
    objs = dataset[0]
    objs_img = []
    for obj in objs:
        img = fac.load_frame(obj[2])
        obj = fac.convert_array_to_object(obj)
        cropped_img = image.crop(img, obj)
        resized_img = image.resize(cropped_img, IMAGE_SHAPE)
        img_arr = image.convert_pil_to_cv(resized_img)
        objs_img.append(img_arr)

    objs_img = np.array(objs_img)/255.0
    extractor = Extractor96()
    features = extractor(objs_img)
    data = features.numpy()
    data = pd.DataFrame(data)
    data = data.transpose()
    plt.plot(data)
    plt.show()


def test_224():
    IMAGE_SHAPE = (224, 224)
    fac = Factory(img_shape=IMAGE_SHAPE)
    dataset = fac.gen_frames()
    objs = dataset[0]
    objs_img = []
    for obj in objs:
        img = fac.load_frame(obj[2])
        obj = fac.convert_array_to_object(obj)
        cropped_img = image.crop(img, obj)
        resized_img = image.resize(cropped_img, IMAGE_SHAPE)
        img_arr = image.convert_pil_to_cv(resized_img)
        objs_img.append(img_arr)

    objs_img = np.array(objs_img)/255.0
    extractor = Extractor224()
    features = extractor(objs_img)
    data = features.numpy()
    data = pd.DataFrame(data)
    data = data.transpose()
    plt.plot(data)
    plt.show()


def test_inception():
    IMAGE_SHAPE = (299, 299)
    fac = Factory(img_shape=IMAGE_SHAPE)
    dataset = fac.gen_frames()
    objs = dataset[0]
    objs_img = []
    for obj in objs:
        img = fac.load_frame(obj[2])
        obj = fac.convert_array_to_object(obj)
        cropped_img = image.crop(img, obj)
        resized_img = image.resize(cropped_img, IMAGE_SHAPE)
        img_arr = image.convert_pil_to_cv(resized_img)
        objs_img.append(img_arr)

    objs_img = np.array(objs_img)/255.0
    extractor = ExtractorInception()
    features = extractor(objs_img)
    features = tf.reshape(features, (features.shape[0], -1))
    data = features.numpy()
    data = pd.DataFrame(data)
    data = data.transpose()
    plt.plot(data)
    plt.show()
