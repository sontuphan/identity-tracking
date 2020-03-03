import os
import time
import cv2 as cv
import numpy as np

from utils import image
from src.humandetection import HumanDetection
from src.tracker import Tracker, Inference
from src.factory import Factory

VIDEO0 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/chaplin.mp4")
VIDEO5 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/MOT17-05-SDP.mp4")
VIDEO6 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/MOT17-06-SDP.mp4")
VIDEO7 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/MOT17-07-SDP.mp4")
VIDEO9 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/MOT17-09-FRCNN.mp4")


def train():
    tracker = Tracker()
    names = ['MOT17-05']
    # names = ['MOT17-05', 'MOT17-09', 'MOT17-10']
    # names = ['MOT17-02', 'MOT17-04', 'MOT17-05',
    #          'MOT17-09', 'MOT17-10', 'MOT17-11']

    pipeline = None
    for name in names:
        generator = Factory(
            name, tracker.batch_size, tracker.image_shape)
        next_pipeline = generator.input_pipeline()
        if pipeline is None:
            pipeline = next_pipeline
        else:
            pipeline = pipeline.concatenate(next_pipeline)

    dataset = pipeline.shuffle(256).batch(
        tracker.batch_size, drop_remainder=True)
    tracker.train(dataset, 5)


def convert():
    tracker = Tracker()
    generator = Factory(
        'MOT17-05', tracker.batch_size, tracker.image_shape)
    pipeline = generator.input_pipeline()
    tracker.convert(pipeline)


def predict(tpu=False):
    tracker = Tracker()
    if tpu:
        inference = Inference()
    hd = HumanDetection()

    cap = cv.VideoCapture(VIDEO0)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    prev_vector = None

    while(cap.isOpened()):
        print("======================================")
        timer = cv.getTickCount()
        ret, frame = cap.read()

        if ret != True:
            break

        imgstart = time.time()
        cv_img = cv.resize(frame, (300, 300))
        # cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY)
        # cv_img = cv.cvtColor(cv_img, cv.COLOR_GRAY2BGR)
        pil_img = image.convert_cv_to_pil(cv_img)
        imgend = time.time()
        print('Image estimated time {:.4f}'.format(imgend-imgstart))

        tpustart = time.time()
        objs = hd.predict(cv_img)
        tpuend = time.time()
        print('TPU estimated time {:.4f}'.format(tpuend-tpustart))

        if len(objs) == 0:
            continue

        if prev_vector is None:
            obj_id = 0
            if len(objs) <= obj_id:
                continue
            box, obj_img = tracker.formaliza_data(objs[obj_id], cv_img)
            if not tpu:
                prev_vector = tracker.predict([obj_img], [box])
            else:
                prev_vector = inference.predict([obj_img], [box])
        else:
            bboxes_batch = []
            obj_imgs_batch = []

            for obj in objs:
                box, obj_img = tracker.formaliza_data(obj, cv_img)
                bboxes_batch.append(box)
                obj_imgs_batch.append(obj_img)

            if not tpu:
                vectors = tracker.predict(obj_imgs_batch, bboxes_batch)
            else:
                vectors = inference.predict(obj_imgs_batch, bboxes_batch)
            argmax = 0
            distancemax = None
            vectormax = None
            for index, vector in enumerate(vectors):
                v = vector - prev_vector
                d = np.linalg.norm(v, 2)
                if index == 0:
                    distancemax = d
                    vectormax = vector
                    argmax = index
                    continue
                if d < distancemax:
                    distancemax = d
                    vectormax = vector
                    argmax = index
            print("Distance:", distancemax)
            if distancemax < 4:
                prev_vector = vectormax
                obj = objs[argmax]
                image.draw_objs(pil_img, [obj])

        # Test human detection
        # image.draw_objs(pil_img, objs)

        img = image.convert_pil_to_cv(pil_img)
        cv.imshow('Video', img)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

        # Calculate frames per second (FPS)
        print("Total estimated time: ",
              (cv.getTickCount()-timer)/cv.getTickFrequency())
        fps = cv.getTickFrequency() / (cv.getTickCount() - timer)
        print("FPS: {:.1f}".format(fps))
