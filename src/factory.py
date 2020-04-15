import os
import random
import configparser
import cv2 as cv
import numpy as np
import csv

from utils import image

# object =  [id, label, frame, score, xmin, ymin, xmax, ymax]


class Factory():
    def __init__(self, data_name):
        self.data_dir = 'data/raw/'
        self.data_name = data_name
        self.out_dir = 'data/train/' + self.data_name + '/'

        config = configparser.ConfigParser()
        config.read(self.data_dir + self.data_name + '/seqinfo.ini')
        self.metadata = (int(config['Sequence']['imWidth']),
                         int(config['Sequence']['imHeight']))

    def write_image(self, dataset):
        iterator = iter(dataset)
        counter = 0
        try:
            while True:
                counter += 1
                imgs, bboxes = next(iterator)
                out_dir = self.out_dir+str(counter)
                os.makedirs(out_dir, exist_ok=True)
                csv_file = open(out_dir+'/box.csv', 'w+')
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(['xmin', 'ymin', 'xmax', 'ymax',
                                     'frame_width', 'frame_height'])

                for index, img in enumerate(imgs):
                    box = bboxes[index][:4]
                    box = [box[0], box[1], box[2], box[3],
                           self.metadata[0], self.metadata[1]]
                    csv_writer.writerow(box)
                    name = None
                    if index == 0:
                        name = 'anchor.jpg'
                    if index == 1:
                        name = 'positive.jpg'
                    if index == 2:
                        name = 'negative.jpg'
                    out_img = os.path.abspath(out_dir+'/'+name)
                    cv.imwrite(out_img, img)
        except StopIteration:
            pass

    def generator(self):
        frames = self.gen_frames()
        triplets = self.gen_triplets(frames)

        for triplet in triplets:
            imgs, bboxes = self.get_obj_img(triplet)
            yield imgs, bboxes

    def get_obj_img(self, objs):
        img_tensor = []
        bbox_tensor = []
        for obj in objs:
            fram_id = obj[2]
            frame = self.load_frame(fram_id)
            box = [obj[4], obj[5], obj[6], obj[7]]
            img = image.crop(frame, box)
            img_tensor.append(img)
            bbox_tensor.append(box)

        return img_tensor, bbox_tensor

    def load_frame(self, img_id):
        name = str(img_id)
        while(len(name) < 6):
            name = "0" + name
        name = name + ".jpg"
        data_dir = self.data_dir + self.data_name + "/img1/"+name
        data_dir = os.path.abspath(data_dir)
        frame = cv.imread(data_dir)
        return frame

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
        objs = filter(lambda line: line[6] == 1 and line[8] >= 0.7, dataset)

        width = self.metadata[0]
        height = self.metadata[1]
        objs = map(lambda line: [
            int(line[1]),  # id
            int(line[1]),  # label
            int(line[0]),  # frame
            float(line[8]),  # score
            int(line[2] if line[2] > 0 else 0),  # xmin
            int(line[3] if line[3] > 0 else 0),  # ymin
            int((line[2]+line[4]) if (line[2]+line[4])
                < width else width-1),  # xmax
            int((line[3]+line[5]) if (line[3]+line[5])
                < height else height-1)  # ymax
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
