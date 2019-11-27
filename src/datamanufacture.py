from __future__ import absolute_import, division, print_function, unicode_literals

import os
import cv2 as cv
import tensorflow as tf
import numpy as np

from utils import image
from utils.bbox import BBox, Object


class DataManufacture():
    def __init__(self, data_name="MOT17-05"):
        self.data_dir = "data/MOT17Det/train/"
        self.data_name = data_name

    def map_frame(self, objs):
        frames = {}
        for obj in objs:
            if frames.get(obj.frame) is None:
                frames[obj.frame] = []
            frames[obj.frame].append(obj)
        return frames

    def load_data(self, only_id=None):
        data_dir = self.data_dir + self.data_name + "/gt/gt.txt"
        data_dir = os.path.abspath(data_dir)
        dataset = np.loadtxt(
            data_dir,
            delimiter=",",
            dtype='int,int,int,int,int,int,int,float,float'
        )
        objs = filter(lambda line: line[6] == 1, dataset)
        if only_id is not None:
            objs = map(lambda line: Object(
                id=line[1],
                label=1 if line[1] == only_id else 0,
                frame=line[0],
                score=line[6],
                bbox=BBox(xmin=line[2], ymin=line[3],
                          xmax=line[2]+line[4], ymax=line[3]+line[5])
            ), objs)
        else:
            objs = map(lambda line: Object(
                id=line[1],
                label=line[1],
                frame=line[0],
                score=line[6],
                bbox=BBox(xmin=line[2], ymin=line[3],
                          xmax=line[2]+line[4], ymax=line[3]+line[5])
            ), objs)
        objs = self.map_frame(objs)
        return objs

    def load_image(self, img_id):
        name = str(img_id)
        while(len(name) < 6):
            name = "0" + name
        name = name + ".jpg"
        data_dir = self.data_dir + self.data_name + "/img1/"+name
        data_dir = os.path.abspath(data_dir)
        frame = cv.imread(data_dir)
        return frame

    def review_data(self, only_id=None):
        dataset = self.load_data(only_id)

        for frame in dataset:
            objs = dataset.get(frame)
            img = self.load_image(frame)
            if img is not None:
                img = image.convert_cv_to_pil(img)
                image.draw_box(img, objs)
                img = image.convert_pil_to_cv(img)

                cv.imshow('Video', img)
                if cv.waitKey(10) & 0xFF == ord('q'):
                    break

        cv.destroyAllWindows()
