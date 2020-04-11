from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv

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


class ExtractorSiamnet(tf.keras.Model):
    def __init__(self):
        super(ExtractorSiamnet, self).__init__()
        self.conv = tf.keras.applications.MobileNetV2(
            input_shape=(96, 96, 3), include_top=False, weights='imagenet')
        self.conv.trainable = False
        self.pool = tf.keras.layers.GlobalMaxPool2D()
        self.fc = tf.keras.layers.Dense(256, activation='relu')

    def call(self, imgs):
        print('IMGS:', imgs.shape)
        conv_output = self.conv(imgs)
        print('CONV:', conv_output.shape)
        pool_output = self.pool(conv_output)
        print('POOL:', pool_output.shape)
        fc_output = self.fc(pool_output)
        print('FC:', fc_output.shape)
        return fc_output


def test_96():
    factory = Factory('MOT17-05')
    dataset = factory.gen_frames()
    objs = dataset[0]
    objs_img = []
    for obj in objs:
        obj_id = obj[2]
        obj_box = [obj[4], obj[5], obj[6], obj[7]]
        img = factory.load_frame(obj_id)
        cropped_img = image.crop(img, obj_box)
        resized_img = image.resize(cropped_img, (96, 96))
        objs_img.append(resized_img)

    objs_img = np.array(objs_img)/255.0
    extractor = Extractor96()
    features = extractor(objs_img)
    data = features.numpy()
    data = pd.DataFrame(data)
    data = data.transpose()
    plt.plot(data)
    plt.show()


def test_224():
    factory = Factory('MOT17-05')
    dataset = factory.gen_frames()
    objs = dataset[0]
    objs_img = []
    for obj in objs:
        obj_id = obj[2]
        obj_box = [obj[4], obj[5], obj[6], obj[7]]
        img = factory.load_frame(obj_id)
        cropped_img = image.crop(img, obj_box)
        resized_img = image.resize(cropped_img, (224, 224))
        objs_img.append(resized_img)

    objs_img = np.array(objs_img)/255.0
    extractor = Extractor224()
    features = extractor(objs_img)
    data = features.numpy()
    data = pd.DataFrame(data)
    data = data.transpose()
    plt.plot(data)
    plt.show()


def test_inception():
    factory = Factory('MOT17-05')
    dataset = factory.gen_frames()
    objs = dataset[0]
    objs_img = []
    for obj in objs:
        obj_id = obj[2]
        obj_box = [obj[4], obj[5], obj[6], obj[7]]
        img = factory.load_frame(obj_id)
        cropped_img = image.crop(img, obj_box)
        resized_img = image.resize(cropped_img, (299, 299))
        objs_img.append(resized_img)

    objs_img = np.array(objs_img)/255.0
    extractor = ExtractorInception()
    features = extractor(objs_img)
    features = tf.reshape(features, (features.shape[0], -1))
    data = features.numpy()
    data = pd.DataFrame(data)
    data = data.transpose()
    plt.plot(data)
    plt.show()


def test_siamnet():
    extractor = ExtractorSiamnet()
    factory = Factory('MOT17-05')
    generator = iter(factory.generator())

    while True:
        imgs, bboxes = next(generator)
        imgs_tensor = []
        for img in imgs:
            imgs_tensor.append(image.resize(img, (96, 96)))
        imgs, bboxes = np.array(imgs_tensor)/255.0, np.array(bboxes)

        (anchor, positive, negative) = tf.concat([extractor(imgs), bboxes], 1)
        print('Anchor:', anchor.shape)
        print('Positive:', positive.shape)
        print('Negative:', negative.shape)

        # lloss = tf.linalg.normalize(anchor - positive, ord='euclidean', axis=0)
        lloss = tf.sqrt(tf.reduce_sum(tf.square(anchor - positive)))
        print('d(a,p):', lloss)
        # rloss = tf.linalg.normalize(anchor - negative, ord='euclidean', axis=0)
        rloss = tf.sqrt(tf.reduce_sum(tf.square(anchor - negative)))
        print('d(a,n):', rloss)
        one = tf.constant(1, dtype=tf.float32)
        proportion = (lloss + one)/(rloss + one)
        print('The proportion:', proportion)

        # Vizualization
        tensor = None
        for img in imgs:
            tensor = img if tensor is None else np.concatenate(
                (tensor, img), axis=1)
        cv.imshow('Video', tensor)
        if cv.waitKey(500) & 0xFF == ord('q'):
                break
        print("================================")

    cv.destroyAllWindows()
