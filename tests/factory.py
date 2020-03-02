import cv2 as cv
import numpy as np

from utils import image
from src.factory import Factory


def generate_data():
    batch_size = 32
    fac = Factory("MOT17-02", batch_size)
    pipeline = fac.input_pipeline()
    next_dm = Factory("MOT17-04", batch_size)
    next_pipeline = next_dm.input_pipeline()

    pipeline.concatenate(next_pipeline)

    dataset = pipeline.shuffle(128).batch(batch_size, drop_remainder=True)
    imgs, bboxes = next(iter(dataset))
    print(imgs.shape, bboxes.shape)


def gen_triplets():
    fac = Factory("MOT17-05", 32)
    frames = fac.gen_frames()
    triplets = fac.gen_triplets(frames)

    for triplet in triplets:
        imgs = None
        for obj in triplet:
            frame = fac.load_frame(obj[2])
            obj = fac.convert_array_to_object(obj)
            (xmin, ymin, xmax, ymax) = obj.bbox
            cropped_img = frame[ymin:ymax, xmin:xmax]
            resized_img = cv.resize(cropped_img, fac.img_shape)
            img = resized_img/255.0
            if imgs is None:
                imgs = img
            else:
                imgs = np.concatenate((imgs, img), axis=1)
        cv.imshow('Triplet', imgs)
        if cv.waitKey(500) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()


def review_source():
    fac = Factory("MOT17-05")
    dataset = fac.gen_frames()

    for index, frame in enumerate(dataset):
        objs = map(fac.convert_array_to_object, frame)
        img = fac.load_frame(index)
        if img is not None:
            img = image.convert_cv_to_pil(img)
            image.draw_objs(img, objs)
            img = image.convert_pil_to_cv(img)

            cv.imshow('Video', img)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break
    cv.destroyAllWindows()


def test_generator():
    fac = Factory()
    dataset = fac.generator()
    dataset = iter(dataset)

    while True:
        imgs, _ = next(dataset)
        tensor = None
        for img in imgs:
            if tensor is None:
                tensor = img
            else:
                tensor = np.concatenate((tensor, img), axis=1)

        cv.imshow('Video', tensor)
        if cv.waitKey(500) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()


def test_pipeline():
    fac = Factory()
    pipeline = fac.input_pipeline()
    pipeline = pipeline.shuffle(128)

    for _ in range(5):
        for data in pipeline.take(1):
            imgs, bboxes = data
            imgs = imgs.numpy()
            bboxes = bboxes.numpy()
        tensor = None
        for img in imgs:
            if tensor is None:
                tensor = img
            else:
                tensor = np.concatenate((tensor, img), axis=1)
        cv.imshow('Video', tensor)
        if cv.waitKey(500) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()
