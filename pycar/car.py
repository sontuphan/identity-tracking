import cv2 as cv
import threading
import time
import requests
from queue import Queue


class Car:
    def __init__(self, host):
        self.host = host
        self.stream_port = 8080
        self.cmd_port = 8000
        self.LEVEL_SPEED = [0, 40, 50, 60, 70, 80, 90, 100]
        self.speed(1)

    def get_stream_url(self):
        return self.host + ":" + str(self.stream_port) + '/?action=stream'

    def get_action_url(self, action):
        return self.host + ":" + str(self.cmd_port) + '/run/?action=' + action

    def get_speed_url(self, speed):
        return self.host + ":" + str(self.cmd_port) + '/run/?speed=' + str(speed)

    def play(self, q, sec):
        while True:
            time.sleep(sec)
            frame = q.get()
            cv.imshow("car-camera-tups", frame)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break
        cv.destroyWindow("car-camera-tups")

    def buffer(self, q, stream):
        while stream.isOpened():
            ret, frame = stream.read()
            if ret is not True:
                break
            if q.full():
                q.get()
            q.put(frame)

    def get_camera(self, rate):
        url = self.get_stream_url()
        q = Queue(2)  # Buffer only 2 frames
        stream = cv.VideoCapture(url)
        buffer_thread = threading.Thread(target=self.buffer, args=(q, stream,))
        play_thread = threading.Thread(target=self.play, args=(q, 1/rate,))
        buffer_thread.start()
        play_thread.start()

    def get_snapshot(self):
        url = self.get_stream_url()
        q = Queue(2)
        stream = cv.VideoCapture(url)
        buffer_thread = threading.Thread(target=self.buffer, args=(q, stream,))
        buffer_thread.start()
        return q

    def speed(self, level):
        speed_url = self.get_speed_url(self.LEVEL_SPEED[level])
        requests.get(speed_url)

    def start(self):
        self.run_action("forward")

    def back(self):
        self.run_action("backward")

    def stop(self):
        self.run_action("stop")

    def left(self):
        self.run_action("camright")

    def right(self):
        self.run_action("camleft")

    def straight(self):
        self.run_action("camstraight")

    def run_action(self, action):
        # bwready | forward | backward | stop
        # fwready | fwleft | fwright |  fwstraight
        url = self.get_action_url(action)
        requests.get(url)
