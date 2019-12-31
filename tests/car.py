import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

from utils import image, car
from src.humandetection import HumanDetection
from src.identitytracking import IdentityTracking

HOST = "http://172.31.0.12"


def test_camera():
    picar = car.Car(HOST)
    picar.get_camera(24)


def test_snapshot():
    picar = car.Car(HOST)
    buffer = picar.get_snapshot()
    while True:
        plt.imshow(buffer.get())
        plt.show(block=False)
        plt.pause(0.01)


def test_action():
    picar = car.Car(HOST)
    picar.start()
    time.sleep(3)
    picar.left()
    time.sleep(3)
    picar.straight()
    time.sleep(3)
    picar.right()
    time.sleep(3)
    picar.stop()


def test_speed():
    picar = car.Car(HOST)
    picar.start()
    time.sleep(3)
    picar.speed(4)
    time.sleep(3)
    picar.stop()


def test_general():
    picar = car.Car(HOST)
    picar.speed(4)
    idtr = IdentityTracking()
    hd = HumanDetection()

    buffer = picar.get_snapshot()
    is_first_frames = idtr.tensor_length
    historical_boxes = []
    historical_obj_imgs = []

    while(True):
        frame = buffer.get()
        img = image.convert_cv_to_pil(frame)
        img = image.resize(img, (640, 480))
        objs = hd.predict(img)

        if len(objs) == 0:
            picar.stop()
            continue

        if is_first_frames > 0:
            obj_id = 0
            if len(objs) > obj_id:
                is_first_frames -= 1
                box, obj_img = idtr.formaliza_data(objs[obj_id], img)
                historical_boxes.append(box)
                historical_obj_imgs.append(obj_img)
            continue
        else:
            bboxes_batch = []
            obj_imgs_batch = []
            for obj in objs:
                box, obj_img = idtr.formaliza_data(obj, img)
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
                image.draw_objs(img, [obj])

                # Drive car
                xmed = (obj.bbox.xmin + obj.bbox.xmax)/2
                # ymin = obj.bbox.ymin
                area = (obj.bbox.xmax-obj.bbox.xmin) * \
                    (obj.bbox.ymax-obj.bbox.ymin)
                # if ymin < 20:
                if area > 170000:
                    picar.back()
                    if xmed < 310:
                        picar.right()
                    elif xmed > 330:
                        picar.left()
                # elif ymin > 60:
                elif area < 150000:
                    picar.start()
                    if xmed < 310:
                        picar.left()
                    elif xmed > 330:
                        picar.right()
                else:
                    picar.stop()
                print("==================")
                print(predictions)
                print(predictions[argmax])

        # Test human detection
        # image.draw_objs(img, objs)
        # Test historical frames
        his_img = None
        for _img in historical_obj_imgs:
            if his_img is None:
                his_img = _img
            else:
                his_img = np.concatenate((his_img, _img), axis=1)
        cv.imshow('History', his_img)

        img = image.convert_pil_to_cv(img)
        cv.imshow('Video', img)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
