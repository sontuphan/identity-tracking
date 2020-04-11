import cv2 as cv
import hashlib


def resize(img, size):
    return cv.resize(img, size)


def crop(img, box):
    [xmin, ymin, xmax, ymax] = box
    return img[ymin:ymax, xmin:xmax]


def colorize(number):
    seed = hashlib.sha1(str(number).encode('utf-8')).hexdigest()
    value = seed[-6:]
    color = tuple(int(value[i:i+2], 16) for i in range(0, 6, 2))
    return color


def draw_objs(img, objs):
    for obj in objs:
        color = colorize(obj[0])
        img = cv.rectangle(img, (obj[4], obj[5]),
                           (obj[6], obj[7]), color, 1)
        img = cv.putText(img, 'id: %d' % (obj[0]), (obj[4]+10, obj[5]+10),
                         cv.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv.LINE_AA)
        img = cv.putText(img, 'label: %s' % (obj[1]), (obj[4]+10, obj[5]+20),
                         cv.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv.LINE_AA)
        img = cv.putText(img, 'score: %.2f' % (obj[3]), (obj[4]+10, obj[5]+30),
                         cv.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv.LINE_AA)
    return img


def draw_boxes(img, boxes):
    for index, box in enumerate(boxes):
        color = colorize(index)
        img = cv.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 1)
    return img
