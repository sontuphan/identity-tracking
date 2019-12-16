import time
import tensorflow as tf

from src.datamanufacture import DataManufacture


def generate_small_data():
    batch_size = 32
    dm = DataManufacture("MOT17-05", 8, batch_size)
    pipeline = dm.input_pipeline()

    dataset = pipeline.shuffle(128).batch(batch_size, drop_remainder=True)
    iteractor = iter(dataset)
    steps_per_epoch = 0

    try:
        while True:
            steps_per_epoch += 1
            data = next(iteractor)
            rnn_inputs, cnn_inputs, labels = data
    except StopIteration:
        pass

    print('Steps per epoch: {}'.format(steps_per_epoch))
    print('Output shapes:', rnn_inputs.shape, cnn_inputs.shape, labels.shape)


def generate_data():
    batch_size = 32
    dm = DataManufacture("MOT17-02", 32, batch_size)
    pipeline = dm.input_pipeline()
    next_dm = DataManufacture("MOT17-04", 32, batch_size)
    next_pipeline = next_dm.input_pipeline()

    pipeline.concatenate(next_pipeline)

    dataset = pipeline.shuffle(128).batch(batch_size, drop_remainder=True)
    rnn_inputs, cnn_inputs, labels = next(iter(dataset))
    print(rnn_inputs.shape, cnn_inputs.shape, labels.shape)


def review_source():
    dm = DataManufacture("MOT17-09", 16, 32)
    dm.review_source()
