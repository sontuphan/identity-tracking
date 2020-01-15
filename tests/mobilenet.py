from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import tensorflow as tf
from tensorflow import lite
import numpy as np

from src.mobilenet import Mobilenet
from src.datamanufacture import DataManufacture


MODELS = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../models/tpu/mobilenet_v2_features_extractor_quant_postprocess.tflite")
IMAGE_SHAPE = (96, 96)


def convert():

    model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_SHAPE+(3,)))
    model.trainable = False

    dm = DataManufacture(hist_len=4, img_shape=IMAGE_SHAPE)
    pipeline = dm.input_pipeline()

    def representative_dataset_gen():
        for tensor in pipeline.take(100):
            _, imgs, _ = tensor
            input_value = imgs[0]
            input_value = tf.reshape(
                input_value, [1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3])
            yield [input_value]

    converter = lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_quant_model = converter.convert()

    open(MODELS, 'wb').write(tflite_quant_model)


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
