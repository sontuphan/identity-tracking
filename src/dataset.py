import pathlib
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2 as cv

AUTOTUNE = tf.data.experimental.AUTOTUNE


class Dataset:
    def __init__(self, image_shape=(160, 160), batch_size=256):
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.data_dir = pathlib.Path('data/train')

    def print_dataset_info(self):
        dataset_names = np.array(
            [item.name for item in self.data_dir.glob('*')])
        num_triplets = len(list(self.data_dir.glob('*/*')))
        num_image = len(list(self.data_dir.glob('*/*/*.jpg')))
        print('*** Dataset names:', dataset_names)
        print('*** There are %d triplets over %d images' %
              (num_triplets, num_image))

    def decode_img(self, file_path):
        img = cv.imread(file_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, self.image_shape)
        return img/127.5 - 1

    def read_img(self, triplet_dir):
        anchor_dir = str(triplet_dir.joinpath('anchor.jpg'))
        anchor_img = self.decode_img(anchor_dir)
        positive_dir = str(triplet_dir.joinpath('positive.jpg'))
        positive_img = self.decode_img(positive_dir)
        negative_dir = str(triplet_dir.joinpath('negative.jpg'))
        negative_img = self.decode_img(negative_dir)
        return np.array([anchor_img, positive_img, negative_img])

    def read_csv(self, triplet_dir):
        csv_file = triplet_dir.joinpath('box.csv')
        df = pd.read_csv(csv_file)
        boxes = np.zeros((3, 4), dtype=np.float32)
        width = df.values[:, 4][0]
        height = df.values[:, 5][0]
        boxes[:, 0:4:2] = df.values[:, 0:4:2]/width
        boxes[:, 1:4:2] = df.values[:, 1:4:2]/height
        return boxes

    def generator(self):
        triplet_dirs = self.data_dir.glob('*/*')
        for triplet_dir in triplet_dirs:
            boxes = self.read_csv(triplet_dir)
            imgs = self.read_img(triplet_dir)
            yield imgs, boxes

    def prepare_for_training(self, ds):
        # ds = ds.cache()
        ds = ds.shuffle(512)
        ds = ds.batch(self.batch_size, drop_remainder=True)
        # ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def pipeline(self):
        ds = tf.data.Dataset.from_generator(
            self.generator,
            output_types=(tf.float32, tf.float32),
            output_shapes=(((3,)+self.image_shape+(3,)), (3, 4))
        )
        train_ds = self.prepare_for_training(ds)
        return train_ds
