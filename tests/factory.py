import cv2 as cv
import numpy as np

from utils import image
from src.factory import Factory


def generate_triplets():
    data_names = ['MOT17-02', 'MOT17-04', 'MOT17-05',
                  'MOT17-09', 'MOT17-10', 'MOT17-11']
    for data_name in data_names:
        fac = Factory(data_name)
        dataset = fac.generator()
        fac.write_image(dataset)


def test_generator():
    fac = Factory('MOT17-05')
    dataset = fac.generator()
    dataset = iter(dataset)

    while True:
        imgs, _ = next(dataset)
        tensor = None
        for img in imgs:
            img = image.resize(img, (160, 160))
            tensor = img if tensor is None else np.concatenate(
                (tensor, img), axis=1)

        cv.imshow('Triplet', tensor)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()


def review_source():
    fac = Factory('MOT17-11')
    dataset = fac.gen_frames()

    for index, objs in enumerate(dataset):
        img = fac.load_frame(index)
        if img is not None:
            img = image.draw_objs(img, objs)
            cv.imshow('Video', img)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break
    cv.destroyAllWindows()
