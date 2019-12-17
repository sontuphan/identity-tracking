import os
import time
import random
import configparser
import cv2 as cv
import numpy as np
import tensorflow as tf

from utils import image
from utils.bbox import BBox, Object

FRAME_SHAPE = (640, 480)


class DataManufacture():
    def __init__(self, data_name='MOT17-05', hist_len=32, batch_size=64,
                 img_shape=(96, 96)):
        self.data_dir = 'data/MOT17Det/train/'
        self.data_name = data_name
        self.hist_len = hist_len
        self.batch_size = batch_size
        self.IMAGE_SHAPE = img_shape

        config = configparser.ConfigParser()
        config.read(self.data_dir + self.data_name + '/seqinfo.ini')
        self.metadata = (int(config['Sequence']['imWidth']),
                         int(config['Sequence']['imHeight']))

    def input_pipeline(self):
        (img_width, img_height) = self.IMAGE_SHAPE
        pipeline = tf.data.Dataset.from_generator(
            self.generator, args=[False],
            output_types=(tf.float32, tf.float32, tf.bool),
            output_shapes=((self.hist_len, 4), (self.hist_len, img_width, img_height, 3), ()), )
        return pipeline

    def generator(self, verbose=False):
        frames = self.gen_data_by_frame()
        hist_data = self.gen_data_by_hist(frames, self.hist_len)
        label_data, labels = self.gen_data_by_label(frames, hist_data)

        avg_time_iter = 0
        for index, _ in enumerate(labels):
            if verbose:
                start_iter = time.time()

            objs = label_data[index]
            coordinates = list(
                map(lambda obj: [obj[-4]/FRAME_SHAPE[0],
                                 obj[-3]/FRAME_SHAPE[1],
                                 obj[-2]/FRAME_SHAPE[0],
                                 obj[-1]/FRAME_SHAPE[1]], objs))
            imgs = self.get_data_by_img(objs)
            label = labels[index]

            if verbose:
                end_iter = time.time()
                avg_time_iter += end_iter-start_iter
                if (index+1) % 100 == 0:
                    print('Estimated time for a iteration (over {} iterations): {:.4f} sec'.format(
                        index+1, avg_time_iter/(index+1)))

            yield coordinates, imgs, label

    def get_data_by_img(self, objs):
        img_tensor = []
        for obj in objs:
            img = self.process_image(obj)
            img_tensor.append(img)
        return img_tensor

    def process_image(self, obj):
        img = self.load_frame(obj[2])
        obj = self.convert_array_to_object(obj)
        cropped_img = image.crop(img, obj)
        resized_img = image.resize(cropped_img, self.IMAGE_SHAPE)
        img_arr = image.convert_pil_to_cv(resized_img)
        return img_arr/255.0

    def load_frame(self, img_id):
        name = str(img_id)
        while(len(name) < 6):
            name = "0" + name
        name = name + ".jpg"
        data_dir = self.data_dir + self.data_name + "/img1/"+name
        data_dir = os.path.abspath(data_dir)
        frame = cv.imread(data_dir)
        if frame is not None:
            img = image.convert_cv_to_pil(frame)
            img = image.resize(img, FRAME_SHAPE)
            return img
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

    def gen_data_by_label(self, frames, hist_data):
        label_data = []
        labels = []
        for tensor in hist_data:
            last_element = tensor[-1]
            index = last_element[2] - 1
            objs = frames[index]
            # Import positive label
            feature1 = tensor.copy()
            label1 = True
            label_data.append(feature1)
            labels.append(label1)
            # Import only one negative label
            feature2 = tensor.copy()
            label2 = False
            while feature2[-1][0] == last_element[0]:
                feature2[-1] = random.choice(objs)
            label_data.append(feature2)
            labels.append(label2)

        return label_data, labels

    def gen_data_by_hist(self, frames, hist_len):
        hist_data = []
        for index, objs in enumerate(frames):
            if len(frames) < index + hist_len:
                break
            for obj in objs:
                tensor = []
                for i in range(hist_len):
                    element = self.get_obj_by_id(obj[0], frames[index + i])
                    tensor.append(element)
            hist_data.append(tensor)
        return hist_data

    def get_obj_by_id(self, id, objs):
        for obj in objs:
            if obj[0] == id:
                return obj
        return [id, id, obj[2], 0.0, 0, 0, 0, 0]

    def gen_data_by_frame(self):
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

    def review_source(self):
        dataset = self.gen_data_by_frame()

        for index, frame in enumerate(dataset):
            objs = map(self.convert_array_to_object, frame)
            img = self.load_frame(index)
            if img is not None:
                image.draw_box(img, objs)
                img = image.convert_pil_to_cv(img)

                cv.imshow('Video', img)
                if cv.waitKey(10) & 0xFF == ord('q'):
                    break
        cv.destroyAllWindows()
