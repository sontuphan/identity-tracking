import cv2 as cv
import os
import matplotlib.pyplot as plt
from PIL import Image

from utils import image

IMAGE = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../data/MOT17Det/train/MOT17-05/img1/000001.jpg")


def review_image():
    img = Image.open(IMAGE)
    plt.imshow(img)
    plt.show()


def resize_image():
    img = Image.open(IMAGE)
    resized_img = image.resize(img, (140, 140))
    plt.imshow(resized_img)
    plt.show()


def crop_image():
    img = Image.open(IMAGE)
    cropped_img = img.crop((12, 147, 93, 343))
    plt.imshow(cropped_img)
    plt.show()
