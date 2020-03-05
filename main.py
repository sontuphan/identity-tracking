import sys

import tensorflow as tf
from tests import utils, factory, tracker
from tests import extractor, humandetection
from pycar import start as car

tf.get_logger().setLevel('ERROR')

if __name__ == "__main__":
    if sys.argv[1] == "--test":

        if sys.argv[2] == "utils":
            if sys.argv[3] == "review_image":
                utils.review_image()
            if sys.argv[3] == "resize_image":
                utils.resize_image()
            if sys.argv[3] == "crop_image":
                utils.crop_image()

        elif sys.argv[2] == "humandetection":
            if sys.argv[3] == "test_with_camera":
                humandetection.test_with_camera()
            if sys.argv[3] == "test_with_video_1":
                humandetection.test_with_video(1)
            if sys.argv[3] == "test_with_video_2":
                humandetection.test_with_video(2)
            if sys.argv[3] == "test_with_video_3":
                humandetection.test_with_video(3)
            if sys.argv[3] == "test_with_video_4":
                humandetection.test_with_video(4)

        elif sys.argv[2] == "factory":
            if sys.argv[3] == "generate_data":
                factory.generate_data()
            if sys.argv[3] == "review_source":
                factory.review_source()
            if sys.argv[3] == "gen_triplets":
                factory.gen_triplets()
            if sys.argv[3] == "test_generator":
                factory.test_generator()
            if sys.argv[3] == "test_pipeline":
                factory.test_pipeline()

        elif sys.argv[2] == "tracker":
            if sys.argv[3] == "train":
                tracker.train()
            if sys.argv[3] == "convert":
                tracker.convert()
            if sys.argv[3] == "predict":
                tracker.predict(False)
            if sys.argv[3] == "inference":
                tracker.predict(True)

        elif sys.argv[2] == "extractor":
            if sys.argv[3] == "test_96":
                extractor.test_96()
            if sys.argv[3] == "test_224":
                extractor.test_224()
            if sys.argv[3] == "test_inception":
                extractor.test_inception()
            if sys.argv[3] == "test_mobilenet":
                extractor.test_mobilenet()

    elif sys.argv[1] == '--pycar':
        if sys.argv[2] == "test_camera":
            car.test_camera()
        if sys.argv[2] == "test_snapshot":
            car.test_snapshot()
        if sys.argv[2] == "test_action":
            car.test_action()
        if sys.argv[2] == "test_speed":
            car.test_speed()
        if sys.argv[2] == "start":
            car.start()

    else:
        print("Error: Invalid option!")
