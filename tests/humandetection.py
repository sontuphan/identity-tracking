import os
import cv2 as cv

from utils import image
from src import humandetection

VIDEO1 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/chaplin.mp4")
VIDEO2 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/gta.mp4")
VIDEO3 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/MOT17-05-SDP.mp4")
VIDEO4 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/MOT17-09-FRCNN.mp4")


def test_with_video(video_id):
    hd = humandetection.HumanDetection(0.7)
    if video_id == 1:
        video = VIDEO1
    if video_id == 2:
        video = VIDEO2
    if video_id == 3:
        video = VIDEO3
    if video_id == 4:
        video = VIDEO4
    cap = cv.VideoCapture(video)

    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            img = image.resize(frame, hd.input_shape)
            objs = hd.predict(img)
            img = image.draw_objs(img, objs)

            cv.imshow('Video', img)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()
