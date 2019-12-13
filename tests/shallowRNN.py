from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
import numpy as np
import cv2 as cv

from utils import detect, image
from utils.bbox import BBox, bbox_to_centroid, centroid_distance
from src.shallowRNN import ShallowRNN
# from src.humandetection import HumanDetection

BATCH_SIZE = 64
SAVED_PATH = "models/shallowRNN.h5"
VIDEO3 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/MOT17-05-SDP.mp4")
INIT_FRAMES = [BBox(xmin=12, ymin=147, xmax=93, ymax=343),
               BBox(xmin=12, ymin=147, xmax=93, ymax=343),
               BBox(xmin=12, ymin=147, xmax=93, ymax=343),
               BBox(xmin=12, ymin=147, xmax=93, ymax=343),
               BBox(xmin=12, ymin=147, xmax=93, ymax=343),
               BBox(xmin=12, ymin=147, xmax=93, ymax=343),
               BBox(xmin=12, ymin=147, xmax=93, ymax=343),
               BBox(xmin=12, ymin=147, xmax=93, ymax=343),
               BBox(xmin=12, ymin=147, xmax=93, ymax=343),
               BBox(xmin=12, ymin=147, xmax=93, ymax=343)
               ]


def train_net():
    sRNN = ShallowRNN()
    dataset = sRNN.preprocess_data("MOT17-05")
    train_dataset_batch = dataset.batch(BATCH_SIZE, drop_remainder=True)
    print("Train dataset dimension:", train_dataset_batch)

    model = sRNN.create_model()
    model.fit(train_dataset_batch, epochs=10)
    model.save(SAVED_PATH)


# def run_net():
#     hd = HumanDetection()
#     model = tf.keras.models.load_model(SAVED_PATH)

#     def convert_to_point(predictions):
#         return (int(round(predictions[0][0]*640)),
#                 int(round(predictions[0][1]*480)))

#     cap = cv.VideoCapture(VIDEO3)
#     if (cap.isOpened() == False):
#         print("Error opening video stream or file")

#     init_tensor = []
#     for bbox in INIT_FRAMES:
#         init_tensor.append([(bbox.xmin+bbox.xmax)/(2*640),
#                             (bbox.ymin+bbox.ymax)/(2*480)])

#     while(cap.isOpened()):
#         ret, frame = cap.read()
#         if ret == True:
#             img = image.convert_cv_to_pil(frame)
#             objs = hd.predict(img)

#             predictions = model.predict(np.array([init_tensor]))
#             predicted_centroid = convert_to_point(predictions)

#             real_centroid = (0, 0)
#             for obj in objs:
#                 centroid = bbox_to_centroid(obj.bbox)
#                 if centroid_distance(real_centroid, predicted_centroid) >= centroid_distance(centroid, predicted_centroid):
#                     real_centroid = centroid

#             init_tensor.pop(0)
#             init_tensor.append([real_centroid[0]/640, real_centroid[1]/480])

#             image.draw_point(img, real_centroid, "blue")
#             image.draw_point(img, predicted_centroid, "red")
#             image.draw_box(img, objs)
#             img = image.convert_pil_to_cv(img)

#             cv.imshow('Video', img)
#             if cv.waitKey(10) & 0xFF == ord('q'):
#                 break
#         else:
#             break

#     cap.release()
#     cv.destroyAllWindows()
