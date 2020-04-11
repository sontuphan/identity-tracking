import os
import time
import cv2 as cv
import numpy as np
from random import randint

from utils import image
from src.humandetection import HumanDetection
from src.tracker import Tracker, Inference
from src.dataset import Dataset

VIDEO0 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/chaplin.mp4")
VIDEO5 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/MOT17-05-SDP.mp4")
VIDEO7 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/MOT17-07-SDP.mp4")
VIDEO9 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/MOT17-09-FRCNN.mp4")


def train():
    tracker = Tracker()
    dataset = Dataset(image_shape=(96, 96), batch_size=tracker.batch_size)

    pipeline = dataset.pipeline()
    tracker.train(pipeline, 20)


def convert():
    tracker = Tracker()
    dataset = Dataset(image_shape=(96, 96), batch_size=tracker.batch_size)
    pipeline = dataset.pipeline()
    tracker.convert(pipeline)


def predict():
    tracker = Tracker()
    hd = HumanDetection()

    cap = cv.VideoCapture(VIDEO5)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    prev_vector = None
    video_len = cap.get(cv.CAP_PROP_FRAME_COUNT)
    skipped_frame = randint(0, video_len)
    print("Video length:", video_len)
    print("Rand skipped frame:", skipped_frame)
    time.sleep(5)

    while(cap.isOpened()):
        timer = cv.getTickCount()
        ret, frame = cap.read()

        if ret != True:
            break
        if skipped_frame > 0:
            skipped_frame -= 1
            continue

        print("======================================")

        imgstart = time.time()
        img = cv.resize(frame, (300, 300))

        # Gray scale situtation
        # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        imgend = time.time()
        print('Image estimated time {:.4f}'.format(imgend-imgstart))

        tpustart = time.time()
        objs = hd.predict(img)
        tpuend = time.time()
        print('TPU estimated time {:.4f}'.format(tpuend-tpustart))

        if len(objs) == 0:
            continue

        if prev_vector is None:
            obj_id = 0
            if len(objs) <= obj_id:
                continue
            box, obj_img = tracker.formaliza_data(objs[obj_id], img)
            prev_vector = tracker.predict([obj_img], [box])
        else:
            bboxes_batch = []
            obj_imgs_batch = []

            for obj in objs:
                box, obj_img = tracker.formaliza_data(obj, img)
                bboxes_batch.append(box)
                obj_imgs_batch.append(obj_img)

            vectors = tracker.predict(obj_imgs_batch, bboxes_batch)
            argmax = 0
            distancemax = None
            vectormax = None
            distances = []
            for index, vector in enumerate(vectors):
                v = vector - prev_vector
                d = np.linalg.norm(v, 2)
                distances.append(d)
                if index == 0:
                    distancemax = d
                    vectormax = vector
                    argmax = index
                    continue
                if d < distancemax:
                    distancemax = d
                    vectormax = vector
                    argmax = index
            print("Distances:", distances)
            print("Min distance:", distancemax)
            if distancemax < 10:
                prev_vector = vectormax
                obj = objs[argmax]
                img = image.draw_objs(img, [obj])

        # Test human detection
        # img = image.draw_objs(img, objs)

        cv.imshow('Video', img)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

        # Calculate frames per second (FPS)
        print("Total estimated time: ",
              (cv.getTickCount()-timer)/cv.getTickFrequency())
        fps = cv.getTickFrequency() / (cv.getTickCount() - timer)
        print("FPS: {:.1f}".format(fps))


def infer():
    inference = Inference()
    hd = HumanDetection()

    cap = cv.VideoCapture(VIDEO5)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    video_len = cap.get(cv.CAP_PROP_FRAME_COUNT)
    skipped_frame = randint(0, video_len)
    print("Video length:", video_len)
    print("Rand skipped frame:", skipped_frame)
    time.sleep(5)

    history = None
    while(cap.isOpened()):
        timer = cv.getTickCount()
        ret, frame = cap.read()

        if ret != True:
            break
        if skipped_frame > 0:
            skipped_frame -= 1
            continue

        print("======================================")

        imgstart = time.time()
        img = cv.resize(frame, (300, 300))
        imgend = time.time()
        print('Image estimated time {:.4f}'.format(imgend-imgstart))

        tpustart = time.time()
        objs = hd.predict(img)
        tpuend = time.time()
        print('TPU estimated time {:.4f}'.format(tpuend-tpustart))

        if len(objs) == 0:
            continue

        if inference.prev_encoding is None:
            obj_id = 0
            if len(objs) <= obj_id:
                continue
            box, obj_img = inference.formaliza_data(objs[obj_id], img)
            inference.predict([obj_img], [box], True)
        else:
            bboxes_batch = []
            obj_imgs_batch = []

            for obj in objs:
                box, obj_img = inference.formaliza_data(obj, img)
                bboxes_batch.append(box)
                obj_imgs_batch.append(obj_img)

            confidences, argmax = inference.predict(
                obj_imgs_batch, bboxes_batch)
            print('Confidences:', confidences)
            if argmax is not None:
                obj = objs[argmax]
                history = image.crop(img, obj)
                img = image.draw_objs(img, [obj])

        # Test human detection
        # img = image.draw_objs(img, objs)

        cv.imshow('Video', img)
        if history is not None:
            cv.imshow('Tracking object', history)
            cv.moveWindow('Tracking object', 500, 500)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

        # Calculate frames per second (FPS)
        print("Total estimated time: ",
              (cv.getTickCount()-timer)/cv.getTickFrequency())
        fps = cv.getTickFrequency() / (cv.getTickCount() - timer)
        print("FPS: {:.1f}".format(fps))
