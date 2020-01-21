import os
import time
import cv2 as cv
import numpy as np

from utils import image
from src.humandetection import HumanDetection
from src.identitytracking import IdentityTracking
from src.datamanufacture import DataManufacture

VIDEO0 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/tokyo-walking.mp4")
VIDEO5 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/MOT17-05-SDP.mp4")
VIDEO7 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/MOT17-07-SDP.mp4")
VIDEO9 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/MOT17-09-FRCNN.mp4")


def train():
    idtr = IdentityTracking()
    # names = ['MOT17-05']
    names = ['MOT17-05', 'MOT17-09', 'MOT17-10']
    # names = ['MOT17-02', 'MOT17-04', 'MOT17-05',
    #          'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']

    pipeline = None
    for name in names:
        generator = DataManufacture(name, idtr.tensor_length,
                                    idtr.batch_size, idtr.image_shape)
        next_pipeline = generator.input_pipeline()
        if pipeline is None:
            pipeline = next_pipeline
        else:
            pipeline = pipeline.concatenate(next_pipeline)

    dataset = pipeline.shuffle(256).batch(
        idtr.batch_size, drop_remainder=True)
    idtr.train(dataset, 5)


def predict():
    idtr = IdentityTracking()
    hd = HumanDetection()

    cap = cv.VideoCapture(VIDEO0)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    is_first_frames = idtr.tensor_length
    historical_boxes = []
    historical_obj_imgs = []

    while(cap.isOpened()):
        timer = cv.getTickCount()
        ret, frame = cap.read()

        if ret != True:
            break

        imgstart = time.time()
        cv_img = cv.resize(frame, (300, 300))
        pil_img = image.convert_cv_to_pil(cv_img)
        imgend = time.time()
        print('Image estimated time {:.4f}'.format(imgend-imgstart))

        tpustart = time.time()
        objs = hd.predict(cv_img)
        tpuend = time.time()
        print('TPU estimated time {:.4f}'.format(tpuend-tpustart))

        if len(objs) == 0:
            continue

        if is_first_frames > 0:
            obj_id = 0
            if len(objs) > obj_id:
                is_first_frames -= 1
                box, obj_img = idtr.formaliza_data(objs[obj_id], cv_img)
                historical_boxes.append(box)
                historical_obj_imgs.append(obj_img)
            continue
        else:
            bboxes_batch = []
            obj_imgs_batch = []

            for obj in objs:
                box, obj_img = idtr.formaliza_data(obj, cv_img)
                boxes_tensor = historical_boxes.copy()
                boxes_tensor.pop(0)
                boxes_tensor.append(box)
                obj_imgs_tensor = historical_obj_imgs.copy()
                obj_imgs_tensor.pop(0)
                obj_imgs_tensor.append(obj_img)
                bboxes_batch.append(boxes_tensor)
                obj_imgs_batch.append(obj_imgs_tensor)

            predictions, argmax = idtr.predict(bboxes_batch, obj_imgs_batch)
            predictions = predictions.numpy()
            argmax = argmax.numpy()
            if predictions[argmax] >= 0.7:
                obj = objs[argmax]
                historical_boxes = bboxes_batch[argmax].copy()
                historical_obj_imgs = obj_imgs_batch[argmax].copy()
                image.draw_objs(pil_img, [obj])

            print("==================")
            print(predictions)
            print(predictions[argmax])

        # Test human detection
        # image.draw_objs(pil_img, objs)
        # Test historical frames
        his_img = None
        for _img in historical_obj_imgs:
            if his_img is None:
                his_img = _img
            else:
                his_img = np.concatenate((his_img, _img), axis=1)
        cv.imshow('History', his_img)
        cv.moveWindow('History', 90, 650)

        img = image.convert_pil_to_cv(pil_img)
        cv.imshow('Video', img)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

        # Calculate Frames per second (FPS)
        print("Total estimated Time: ",
              (cv.getTickCount()-timer)/cv.getTickFrequency())
        fps = cv.getTickFrequency() / (cv.getTickCount() - timer)
        print("FPS: {:.1f}".format(fps))
