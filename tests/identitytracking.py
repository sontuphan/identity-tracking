import cv2 as cv
import tensorflow as tf
from src.identitytracking import IdentityTracking


def load_data():
    idtr = IdentityTracking()
    dataset = idtr.load_data()
    for (label, cnn, rnn) in dataset:
        x = tf.train.Example.FromString(rnn)
        print(x)
        break
        # cv.imshow('Video', cnn[0])
        # if cv.waitKey(10) & 0xFF == ord('q'):
        #     break


def train():
    idtr = IdentityTracking()
    # idtr.train()


def predict():
    idtr = IdentityTracking()
    idtr.predict()
