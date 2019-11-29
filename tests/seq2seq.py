from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
import numpy as np
import cv2 as cv

from utils import detect, image
from utils.bbox import Object, BBox
from src.seq2seq import Seq2Seq
from src.humandetection import HumanDetection

BATCH_SIZE = 64
SAVED_PATH = "models/shallowRNN.h5"
VIDEO3 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/MOT17-05-SDP.mp4")
INIT_FRAMES = [[12/640, 147/480, 93/640, 343/480],
               [12/640, 147/480, 93/640, 343/480],
               [12/640, 147/480, 93/640, 343/480],
               [12/640, 147/480, 93/640, 343/480],
               [12/640, 147/480, 93/640, 343/480],
               [12/640, 147/480, 93/640, 343/480],
               [12/640, 147/480, 93/640, 343/480],
               [12/640, 147/480, 93/640, 343/480],
               [12/640, 147/480, 93/640, 343/480],
               [12/640, 147/480, 93/640, 343/480]]


def train_net():
    s2s = Seq2Seq()
    dataset = s2s.preprocess_data("MOT17-05", 100)
    print("Dataset dimension: ({},{}) - (feature, label)".format(
        dataset[0].shape, dataset[1].shape))
    s2s.train(dataset)


def run_net():
    hd = HumanDetection()
    s2s = Seq2Seq()

    cap = cv.VideoCapture(VIDEO3)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    histories = INIT_FRAMES.copy()

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            img = image.convert_cv_to_pil(frame)
            objs = hd.predict(img)

            inputs = []
            for obj in objs:
                tensor = histories.copy()
                coordinate = [obj.bbox.xmin/640, obj.bbox.ymin/480,
                              obj.bbox.xmax/640, obj.bbox.ymax/480]
                tensor.append(coordinate)
                inputs.append(tensor)
            predictions, argmax = s2s.predict(inputs)
            histories = inputs[argmax]
            histories.pop(0)
            tracking_obj = inputs[argmax][-1]
            obj = Object(id=1, frame=0, label=1, score=predictions[argmax],
                         bbox=BBox(xmin=int(tracking_obj[0]*640),
                                   ymin=int(tracking_obj[1]*480),
                                   xmax=int(tracking_obj[2]*640),
                                   ymax=int(tracking_obj[3]*480)))

            image.draw_box(img, [obj])
            # image.draw_box(img, objs)
            img = image.convert_pil_to_cv(img)

            cv.imshow('Video', img)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()
