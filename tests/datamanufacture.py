import cv2 as cv
import numpy as np

from utils import image
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


def review_hist_data():
    dm = DataManufacture("MOT17-05", 8, 32)
    frames = dm.gen_data_by_frame()
    hist_data = dm.gen_data_by_hist(frames, 8)

    for tensor in hist_data:
        imgs = None
        for obj in tensor:
            frame = dm.load_frame(obj[2])
            obj = dm.convert_array_to_object(obj)
            img = image.crop(frame, obj)
            img = image.resize(img, (96, 96))
            img = image.convert_pil_to_cv(img)
            if imgs is None:
                imgs = img
            else:
                imgs = np.concatenate((imgs, img), axis=1)
        cv.imshow('Video', imgs)
        if cv.waitKey(500) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()


def review_source():
    dm = DataManufacture("MOT17-05", 8, 32)
    dataset = dm.gen_data_by_frame()

    for index, frame in enumerate(dataset):
        objs = map(dm.convert_array_to_object, frame)
        img = dm.load_frame(index)
        if img is not None:
            image.draw_box(img, objs)
            img = image.convert_pil_to_cv(img)

            cv.imshow('Video', img)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break
    cv.destroyAllWindows()
