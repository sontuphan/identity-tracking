import os
import time
import cv2 as cv
import numpy as np
from random import randint

from utils import image
from src.humandetection import HumanDetection
from src.tracker import Tracker, Inference, formalize_data
from src.dataset import Dataset

VIDEO0 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../video/chaplin.mp4")
VIDEO5 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../video/MOT17-05-SDP.mp4")
VIDEO7 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../video/MOT17-07-SDP.mp4")
VIDEO9 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../video/MOT17-09-FRCNN.mp4")


def get_video():
    cap = cv.VideoCapture(VIDEO5)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    video_len = cap.get(cv.CAP_PROP_FRAME_COUNT)
    skipped_frame = randint(0, video_len)
    print("Video length:", video_len)
    print("Rand skipped frame:", skipped_frame)
    time.sleep(5)

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret != True:
            break
        if skipped_frame > 0:
            skipped_frame -= 1
            continue

        yield frame


def train():
    tracker = Tracker()
    dataset = Dataset(image_shape=tracker.image_shape,
                      batch_size=tracker.batch_size)
    pipeline = dataset.pipeline()
    tracker.train(pipeline, 10)


def convert():
    tracker = Tracker()
    dataset = Dataset(image_shape=tracker.image_shape,
                      batch_size=tracker.batch_size)
    pipeline = dataset.pipeline()
    tracker.convert(pipeline)


def predict():
    tracker = Tracker()
    hd = HumanDetection()
    video = get_video()

    prev_vector = None
    while(True):
        # Get raw image
        try:
            frame = next(video)
        except StopIteration:
            break
        # Convert to RGB
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # Gray scale situtation
        # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        start = time.time()
        print("======================================")

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
            obj_img, _ = formalize_data(objs[obj_id], img)
            obj_imgs = np.array([obj_img])
            prev_vector = tracker.extract_features(obj_imgs)
        else:
            obj_imgs_batch = []
            for obj in objs:
                obj_img, _ = formalize_data(obj, img)
                obj_imgs_batch.append(obj_img)
            obj_imgs_batch = np.array(obj_imgs_batch)

            vectors = tracker.extract_features(obj_imgs_batch)
            distancemin = None
            vectormin = None
            argmin = 0
            distances = []
            for index, vector in enumerate(vectors):
                d = np.linalg.norm(vector - prev_vector, 2)
                distances.append(d)
                if index == 0:
                    distancemin = d
                    vectormin = vector
                    argmin = index
                    continue
                if d < distancemin:
                    distancemin = d
                    vectormin = vector
                    argmin = index
            print("Distances:", distances)
            print("Min distance:", distancemin)
            if distancemin < 50:
                prev_vector = vectormin
                obj = objs[argmin]
                frame = image.draw_objs(frame, [obj])

        # Test human detection
        # frame = image.draw_objs(frame, objs)

        cv.imshow('Video', frame)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

        # Calculate frames per second (FPS)
        end = time.time()
        print('Total estimated time: {:.4f}'.format(end-start))
        fps = 1/(end-start)
        print("FPS: {:.1f}".format(fps))


def infer():
    inference = Inference()
    hd = HumanDetection()
    video = get_video()

    history = None
    while(True):
        # Get raw image
        try:
            frame = next(video)
        except StopIteration:
            break
        # Convert to RGB
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        start = time.time()
        print("======================================")

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
            obj_img, box = formalize_data(objs[obj_id], img)
            inference.set_anchor(obj_img, box)
        else:
            bboxes_batch = []
            obj_imgs_batch = []

            for obj in objs:
                obj_img, box = formalize_data(obj, img)
                bboxes_batch.append(box)
                obj_imgs_batch.append(obj_img)

            confidences, argmax = inference.predict(
                obj_imgs_batch, bboxes_batch)
            print('Confidences:', confidences)
            if argmax is not None:
                box = bboxes_batch[argmax]
                history = image.crop(frame, box)
                frame = image.draw_box(frame, box)

        # Test human detection
        # frame = image.draw_objs(frame, objs)

        cv.imshow('Video', frame)
        if history is not None:
            history = image.resize(history, (100, 100))
            cv.imshow('Tracking object', history)
            cv.moveWindow('Tracking object', 500, 500)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

        # Calculate frames per second (FPS)
        end = time.time()
        print('Total estimated time: {:.4f}'.format(end-start))
        fps = 1/(end-start)
        print("FPS: {:.1f}".format(fps))
