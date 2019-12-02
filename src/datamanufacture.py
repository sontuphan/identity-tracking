from __future__ import absolute_import, division, print_function, unicode_literals

import os
import cv2 as cv
import tensorflow as tf
import numpy as np
import json

from utils import image
from utils.bbox import BBox, Object


class DataManufacture():
    def __init__(self, data_name="MOT17-05"):
        self.data_dir = "data/MOT17Det/train/"
        self.data_name = data_name
        self.src_dir = "data/train/{}/".format(self.data_name)

        if not os.path.exists(self.src_dir):
            os.mkdir(self.src_dir)

    def generate_data(self):
        frames = self.process_data()
        self.save_data(frames)
        self.process_image(frames)

    def process_image(self, frames):
        for frame in frames:
            container = self.src_dir+frame
            if not os.path.exists(container):
                os.mkdir(container)
            objs = map(self.convert_array_to_object, frames.get(frame))
            img = self.load_image(frame)
            img = image.convert_cv_to_pil(img)
            for obj in objs:
                cropped_img = image.crop(img, obj)
                resized_img = image.resize(cropped_img, (150, 150))
                resized_img.save(container+"/"+str(obj[0])+".jpg")

    def load_image(self, img_id):
        name = str(img_id)
        while(len(name) < 6):
            name = "0" + name
        name = name + ".jpg"
        data_dir = self.data_dir + self.data_name + "/img1/"+name
        data_dir = os.path.abspath(data_dir)
        frame = cv.imread(data_dir)
        return frame

    def save_data(self, data):
        data = json.dumps(data)
        f = open("{}data.json".format(self.src_dir), "w")
        f.write(data)
        f.close()

    def load_data(self):
        f = open("{}data.json".format(self.src_dir), "r")
        data = f.read()
        f.close()
        return json.loads(data)

    def process_data(self, only_id=None):
        data_dir = self.data_dir + self.data_name + "/gt/gt.txt"
        data_dir = os.path.abspath(data_dir)
        dataset = np.loadtxt(
            data_dir,
            delimiter=",",
            dtype='int,int,int,int,int,int,int,float,float'
        )
        objs = filter(lambda line: line[6] == 1 and line[8] >= 0.2, dataset)
        # id/label/frame/score/xmin/ymin/xmax/ymax
        if only_id is not None:
            objs = map(lambda line: [
                int(line[1]),  # id
                1 if line[1] == only_id else 0,  # label
                int(line[0]),  # frame
                float(line[8]),  # score
                int(line[2]), int(line[3]),  # xmin, ymin
                int(line[2])+int(line[4]), int(line[3])+int(line[5])]  # xmax, ymax
                , objs)
        else:
            objs = map(lambda line: [
                int(line[1]),  # id
                int(line[1]),  # label
                int(line[0]),  # frame
                float(line[8]),  # score
                int(line[2]), int(line[3]),  # xmin, ymin
                int(line[2])+int(line[4]), int(line[3])+int(line[5])]  # xmax, ymax
                , objs)

        frames = {}
        for obj in objs:
            frame = str(obj[2])
            if frames.get(frame) is None:
                frames[frame] = []
            frames[frame].append(obj)

        return frames

    def convert_array_to_object(self, array):
        return Object(
            id=array[0],
            label=array[1],
            frame=array[2],
            score=array[3],
            bbox=BBox(xmin=array[4], ymin=array[5],
                      xmax=array[6], ymax=array[7])
        )

    def review_data(self, only_id=None):
        dataset = self.process_data(only_id)

        for frame in dataset:
            objs = map(self.convert_array_to_object, dataset.get(frame))
            img = self.load_image(frame)
            if img is not None:
                img = image.convert_cv_to_pil(img)
                image.draw_box(img, objs)
                img = image.convert_pil_to_cv(img)

                cv.imshow('Video', img)
                if cv.waitKey(10) & 0xFF == ord('q'):
                    break

        cv.destroyAllWindows()
