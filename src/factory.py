import os
import random
import configparser
import cv2 as cv
import numpy as np
import tensorflow as tf

from utils.bbox import BBox, Object

FRAME_SHAPE = (300, 300)


class Factory():
    def __init__(self, data_name='MOT17-05', batch_size=64,
                 img_shape=(96, 96)):
        self.data_dir = 'data/train/'
        self.data_name = data_name
        self.batch_size = batch_size
        self.img_shape = img_shape

        config = configparser.ConfigParser()
        config.read(self.data_dir + self.data_name + '/seqinfo.ini')
        self.metadata = (int(config['Sequence']['imWidth']),
                         int(config['Sequence']['imHeight']))

    def input_pipeline(self):
        pipeline = tf.data.Dataset.from_generator(
            self.generator, args=[],
            output_types=(tf.float32, tf.float32),
            output_shapes=(((3,)+self.img_shape+(3,)), (3, 4)), )
        return pipeline

    def generator(self):
        frames = self.gen_frames()
        triplets = self.gen_triplets(frames)

        for triplet in triplets:
            imgs, bboxes = self.normalize_data(triplet)
            yield imgs, bboxes

    def normalize_data(self, objs):
        img_tensor = []
        bbox_tensor = []
        for obj in objs:
            frame = self.load_frame(obj[2])
            obj = self.convert_array_to_object(obj)
            (xmin, ymin, xmax, ymax) = obj.bbox

            cropped_img = frame[ymin:ymax, xmin:xmax]
            resized_img = cv.resize(cropped_img, self.img_shape)
            img = resized_img/255.0
            img_tensor.append(img)

            bbox = [xmin/FRAME_SHAPE[0],
                    ymin/FRAME_SHAPE[1],
                    xmax/FRAME_SHAPE[0],
                    ymax/FRAME_SHAPE[1]]
            bbox_tensor.append(bbox)

        return img_tensor, bbox_tensor

    def load_frame(self, img_id):
        name = str(img_id)
        while(len(name) < 6):
            name = "0" + name
        name = name + ".jpg"
        data_dir = self.data_dir + self.data_name + "/img1/"+name
        data_dir = os.path.abspath(data_dir)
        frame = cv.imread(data_dir)
        if frame is not None:
            return cv.resize(frame, FRAME_SHAPE)
        else:
            return None

    def convert_array_to_object(self, array):
        return Object(
            id=array[0],
            label=array[1],
            frame=array[2],
            score=array[3],
            bbox=BBox(xmin=array[4], ymin=array[5],
                      xmax=array[6], ymax=array[7])
        )

    def gen_triplets(self, frames):
        data = []
        for index, objs in enumerate(frames):
            if len(frames) <= index+1:
                break
            for obj in objs:
                pos = self.get_obj_by_id(obj[0], frames[index + 1], False)
                if pos is None:
                    continue
                neg = self.get_obj_by_id(obj[0], frames[index + 1], True)
                data.append([obj, pos, neg])
        return data

    def get_obj_by_id(self, obj_id, objs, negative):
        if negative and len(objs) >= 2:
            obj = random.choice(objs)
            while obj[0] == obj_id:
                obj = random.choice(objs)
            return obj
        for obj in objs:
            if obj[0] == obj_id:
                return obj
        return None

    def gen_frames(self):
        data_dir = self.data_dir + self.data_name + "/gt/gt.txt"
        data_dir = os.path.abspath(data_dir)
        dataset = np.loadtxt(
            data_dir,
            delimiter=",",
            dtype='int,int,int,int,int,int,int,float,float'
        )
        objs = filter(lambda line: line[6] == 1 and line[8] >= 0.2, dataset)

        width_scale = self.metadata[0]/FRAME_SHAPE[0]
        height_scale = self.metadata[1]/FRAME_SHAPE[1]
        objs = map(lambda line: [
            int(line[1]),  # id
            int(line[1]),  # label
            int(line[0]),  # frame
            float(line[8]),  # score
            int((line[2] if line[2] > 0 else 0)/width_scale),  # xmin
            int((line[3] if line[3] > 0 else 0)/height_scale),  # ymin
            int((line[2]+line[4])/width_scale if (line[2]+line[4]) / \
                width_scale < FRAME_SHAPE[0] else FRAME_SHAPE[0]-1),  # xmax
            int((line[3]+line[5])/height_scale if (line[3]+line[5]) / \
                height_scale < FRAME_SHAPE[1] else FRAME_SHAPE[1]-1)  # ymax
        ], objs)

        objs = list(objs)
        stop = len(objs)
        frames = []
        counter = 0
        while stop > 0:
            counter += 1
            frame = []
            for obj in objs:
                if counter == obj[2]:
                    frame.append(obj)
                    stop -= 1
            frames.append(frame)
        return frames
