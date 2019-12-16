import os
import cv2 as cv
import numpy as np

from utils import image
from src.identitytracking import IdentityTracking, FeaturesExtractor, DimensionExtractor
from src.datamanufacture import DataManufacture

VIDEO0 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/gta.mp4")
VIDEO5 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/MOT17-05-SDP.mp4")
VIDEO6 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/MOT17-06-SDP.mp4")
VIDEO9 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/video/MOT17-09-FRCNN.mp4")


def summarize():
    pass


def train():
    idtr = IdentityTracking()
    # names = ['MOT17-05']
    names = ['MOT17-02', 'MOT17-04', 'MOT17-05',
             'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']

    pipeline = None
    for name in names:
        generator = DataManufacture(name, idtr.tensor_length, idtr.batch_size)
        next_pipeline = generator.input_pipeline()
        if pipeline is None:
            pipeline = next_pipeline
        else:
            pipeline = pipeline.concatenate(next_pipeline)

    dataset = pipeline.shuffle(128).batch(
        idtr.batch_size, drop_remainder=True)
    idtr.train(dataset, 10)