import os
import cv2 as cv

from utils import camera, image
from src import humandetection

VIDEO1 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/chaplin.mp4")
VIDEO2 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/gta.mp4")
VIDEO3 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/MOT17-05-SDP.mp4")
VIDEO4 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/MOT17-09-FRCNN.mp4")


def test_with_camera():
    hd = humandetection.HumanDetection(0.6)
    cam = camera.Camera()
    stream = cam.get_stream()
    print("You can press Q button to terminate the process!")

    while True:
        img = stream.get()

        img = image.convert_cv_to_pil(img)
        objs = hd.predict(img)
        image.draw_objs(img, objs)
        img = image.convert_pil_to_cv(img)

        cv.imshow("Debug", img)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    cam.terminate()


def test_with_video(id):
    hd = humandetection.HumanDetection(0.55)
    if id == 1:
        video = VIDEO1
    if id == 2:
        video = VIDEO2
    if id == 3:
        video = VIDEO3
    if id == 4:
        video = VIDEO4
    cap = cv.VideoCapture(video)

    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv_img = cv.resize(frame, (300, 300))
            pil_img = image.convert_cv_to_pil(cv_img)
            objs = hd.predict(cv_img)
            image.draw_objs(pil_img, objs)
            img = image.convert_pil_to_cv(pil_img)

            cv.imshow('Video', img)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()
