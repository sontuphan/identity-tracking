import os
import cv2 as cv
import numpy as np

from utils import image
from src.humandetection import HumanDetection
from src.identitytracking import IdentityTracking
from src.datamanufacture import DataManufacture

VIDEO0 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/3rd-camera.mp4")
VIDEO5 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/MOT17-05-SDP.mp4")
VIDEO6 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/MOT17-06-SDP.mp4")
VIDEO9 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/MOT17-09-FRCNN.mp4")


def train():
    idtr = IdentityTracking()
    names = ['MOT17-05']
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


def validate():
    idtr = IdentityTracking()
    dm = DataManufacture('MOT17-05', idtr.tensor_length,
                         idtr.batch_size, idtr.image_shape)

    ground_objs = dm.gen_data_by_frame()
    is_first_frames = idtr.tensor_length
    histories = []
    for frame, objs in enumerate(ground_objs):
        img = dm.load_frame(frame)

        objs = map(dm.convert_array_to_object, objs)
        objs = list(objs)

        if is_first_frames > 0:
            obj_id = 0
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
                if predictions[argmax] >= 0.5:
                    obj = objs[argmax]
                    histories.pop(0)
                    histories.append((obj, img.copy()))
                    image.draw_box(img, [obj])

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


def predict():
    idtr = IdentityTracking()
    hd = HumanDetection()

    cap = cv.VideoCapture(VIDEO5)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    is_first_frames = idtr.tensor_length
    histories = []
    while(cap.isOpened()):

        ret, frame = cap.read()

        if ret != True:
            break

        img = image.convert_cv_to_pil(frame)
        img = image.resize(img, (640, 480))
        objs = hd.predict(img)

        if is_first_frames > 0:
            obj_id = 0
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
