import os
import cv2 as cv
import tensorflow as tf
import numpy as np

from utils import image
from src.identitytracking import IdentityTracking
from src.datamanufacture import DataManufacture
from src.humandetection import HumanDetection

VIDEO3 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/MOT17-05-SDP.mp4")


def train():
    idtr = IdentityTracking()
    dm = DataManufacture("MOT17-05", idtr.tensor_length, idtr.batch_size)
    pipeline = dm.input_pipeline()

    dataset = pipeline.shuffle(128).batch(idtr.batch_size, drop_remainder=True)
    idtr.train(dataset, 51, 5)


def predict():
    idtr = IdentityTracking()
    hd = HumanDetection()

    cap = cv.VideoCapture(VIDEO3)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    is_first_frame = True
    histories = []
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret != True:
            break

        img = image.convert_cv_to_pil(frame)
        objs = hd.predict(img)

        if is_first_frame:
            is_first_frame = False
            for _ in range(idtr.tensor_length):
                histories.append((objs[0], img))
        else:
            predictions = np.array([])
            for obj in objs:
                inputs = histories.copy()
                inputs.pop(0)
                inputs.append((obj, img))
                prediction = idtr.predict(inputs).numpy()
                predictions = np.concatenate((predictions, prediction), axis=0)
            print("==============================")
            print(predictions)
            obj = objs[np.argmax(predictions)]
            histories.pop(0)
            histories.append((obj, img))
            image.draw_box(img, [obj])

        img = image.convert_pil_to_cv(img)
        cv.imshow('Video', img)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
