from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import tensorflow as tf
from tensorflow import lite
import tflite_runtime.interpreter as tflite
import numpy as np

from src.mobilenet import Mobilenet


EDGETPU_SHARED_LIB = 'libedgetpu.so.1'
MODELS = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../models/tpu/mobilenet_v2_features_extractor.tflite")
IMAGE_SHAPE = (96, 96)


def convert():
    model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_SHAPE+(3,)))
    converter = lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    open(MODELS, 'wb').write(tflite_model)


def run():
    mbn = Mobilenet()

    tensor = None
    for _ in range(5):
        input_data = np.array(np.random.random_sample(
            (IMAGE_SHAPE+(3,))), dtype=np.float32)
        if tensor is None:
            tensor = np.array([input_data])
        else:
            tensor = np.append(tensor, np.array([input_data]), axis=0)

    tpustart = time.time()
    print('Input shape:', tensor.shape)
    features = mbn.predict(tensor)
    print('Output shape:', features.shape)
    tpuend = time.time()
    print('TPU estimated time {:.4f}'.format(tpuend-tpustart))
