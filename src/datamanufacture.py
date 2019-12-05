import os
import time
import cv2 as cv
import numpy as np
import tensorflow as tf

from utils import image
from utils.bbox import BBox, Object

IMAGE_SHAPE = (224, 224)


class DataManufacture():
    def __init__(self, data_name='MOT17-05', hist_len=32):
        self.data_dir = 'data/MOT17Det/train/'
        self.data_name = data_name
        self.hist_len = hist_len

    def input_pipeline(self):
        (img_width, img_height) = IMAGE_SHAPE
        pipeline = tf.data.Dataset.from_generator(
            self.generator, args=[True],
            output_types=(tf.float32, tf.uint8, tf.bool),
            output_shapes=((self.hist_len, 4), (self.hist_len, img_width, img_height, 3), ()), )
        return pipeline

    def generator(self, verbose=False):
        frames = self.gen_data_by_frame()
        hist_data = self.gen_data_by_hist(frames, self.hist_len)
        label_data, labels = self.gen_data_by_label(frames, hist_data)

        for index, _ in enumerate(labels):
            start = time.time()
            objs = label_data[index]
            cordinates = list(
                map(lambda obj: [obj[-4]/640, obj[-3]/480, obj[-2]/640, obj[-1]/480], objs))
            imgs = self.get_data_by_img(objs)
            label = labels[index]
            end = time.time()
            if verbose:
                print('Estimated time for one iteration: {} sec'.format(end-start))
            yield cordinates, imgs, label

    def get_data_by_img(self, objs):
        img_tensor = []
        for obj in objs:
            img = self.process_image(obj)
            img_tensor.append(img.tolist())
        return img_tensor

    def process_image(self, obj):
        obj = self.convert_array_to_object(obj)
        img = self.load_frame(obj[2])
        img = image.convert_cv_to_pil(img)
        cropped_img = image.crop(img, obj)
        resized_img = image.resize(cropped_img, IMAGE_SHAPE)
        img_arr = image.convert_pil_to_cv(resized_img)
        return img_arr

    def load_frame(self, img_id):
        name = str(img_id)
        while(len(name) < 6):
            name = "0" + name
        name = name + ".jpg"
        data_dir = self.data_dir + self.data_name + "/img1/"+name
        data_dir = os.path.abspath(data_dir)
        frame = cv.imread(data_dir)
        return frame

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
            for obj in objs:
                feature = tensor.copy()
                feature[-1] = obj
                label = True if obj[0] == last_element[0] else False
                label_data.append(feature)
                labels.append(label)

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
        objs = map(lambda line: [
            int(line[1]),  # id
            int(line[1]),  # label
            int(line[0]),  # frame
            float(line[8]),  # score
            int(line[2]), int(line[3]),  # xmin, ymin
            int(line[2])+int(line[4]), int(line[3])+int(line[5])]  # xmax, ymax
            , objs)

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
                img = image.convert_cv_to_pil(img)
                image.draw_box(img, objs)
                img = image.convert_pil_to_cv(img)

                cv.imshow('Video', img)
                if cv.waitKey(10) & 0xFF == ord('q'):
                    break

        cv.destroyAllWindows()
