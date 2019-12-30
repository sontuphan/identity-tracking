import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

from utils import image, car
from src.humandetection import HumanDetection
from src.identitytracking import IdentityTracking
from src.datamanufacture import DataManufacture

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
    # picar.speed(2)
    idtr = IdentityTracking()
    hd = HumanDetection()

    buffer = picar.get_snapshot()
    is_first_frames = idtr.tensor_length
    histories = []

    while(True):
        frame = buffer.get()
        img = image.convert_cv_to_pil(frame)
        img = image.resize(img, (640, 480))
        objs = hd.predict(img)

        if len(objs) == 0:
            picar.stop()

        if is_first_frames > 0:
            obj_id = 0
            _img = image.convert_pil_to_cv(img)
            cv.imshow('Video', _img)
            cv.waitKey(10)
            if len(objs) > obj_id:
                is_first_frames -= 1
                histories.append((objs[obj_id], img))
            continue
        else:
            inputs = []
            for obj in objs:
                tensor = histories.copy()
                tensor.pop(0)
                tensor.append((obj, img))
                inputs.append(tensor)
            if len(inputs) > 0:
                predictions, argmax = idtr.predict(inputs)
                predictions = predictions.numpy()
                argmax = argmax.numpy()
                if predictions[argmax] >= 0.7:
                    obj = objs[argmax]
                    histories.pop(0)
                    histories.append((obj, img.copy()))
                    image.draw_box(img, [obj])

                    # Drive car
                    xmin = obj.bbox.xmin
                    ymin = obj.bbox.ymin
                    if ymin < 20:
                        picar.back()
                        if xmin < 220:
                            picar.right()
                        elif xmin > 260:
                            picar.left()
                    elif ymin > 60:
                        picar.start()
                        if xmin < 220:
                            picar.left()
                        elif xmin > 260:
                            picar.right()
                    else:
                        picar.stop()

                    
                # else:
                #     obj = DataManufacture().convert_array_to_object([0, 0, 0, 0., 0, 0, 0, 0])
                #     histories.pop(0)
                #     histories.append((obj, img.copy()))

                print("==================")
                print(predictions)
                print(predictions[argmax])

        # Test human detection
        # image.draw_box(img, objs)
        # Test historical frames
        his_img = None
        for history in histories:
            (_obj, _img) = history
            if his_img is None:
                cropped_img = image.crop(_img, _obj)
                resized_img = image.resize(cropped_img, (96, 96))
                his_img = image.convert_pil_to_cv(resized_img)
            else:
                cropped_img = image.crop(_img, _obj)
                resized_img = image.resize(cropped_img, (96, 96))
                his_img = np.concatenate(
                    (his_img, image.convert_pil_to_cv(resized_img)), axis=1)
        cv.imshow('History', his_img)

        img = image.convert_pil_to_cv(img)
        cv.imshow('Video', img)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
