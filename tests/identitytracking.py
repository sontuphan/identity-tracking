import os
import cv2 as cv

from utils import image
from src.identitytracking import IdentityTracking
from src.datamanufacture import DataManufacture
from src.humandetection import HumanDetection

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
    # names = ['MOT17-05']
    names = ['MOT17-02', 'MOT17-04', 'MOT17-05',
             'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']

    pipeline = None
    for name in names:
        generator = DataManufacture(name, idtr.tensor_length, idtr.batch_size)
        next_pipeline = generator.input_pipeline()
        if pipeline is None:
            pipeline = next_pipeline
        else:
            pipeline = pipeline.concatenate(next_pipeline)

    dataset = pipeline.shuffle(128).batch(
        idtr.batch_size, drop_remainder=True)
    idtr.train(dataset, 5)


def predict():
    idtr = IdentityTracking()
    hd = HumanDetection()

    cap = cv.VideoCapture(VIDEO0)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    is_first_frames = True
    histories = []
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret != True:
            break

        img = image.convert_cv_to_pil(frame)
        img = image.resize(img, (640, 480))
        objs = hd.predict(img)

        if is_first_frames:
            for _ in range(idtr.tensor_length):
                is_first_frames = False
                histories.append((objs[0], img))
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
                if predictions[argmax] >= 0.2:
                    obj = objs[argmax]
                    histories.pop(0)
                    histories.append((obj, img))
                    image.draw_box(img, [obj])

                print("==================")
                print(predictions)

        # Test human detection
        # image.draw_box(img, objs)

        img = image.convert_pil_to_cv(img)
        cv.imshow('Video', img)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
