import sys

import tensorflow as tf
from tests import utils, datamanufacture, identitytracking
from tests import humandetection
from tests import extractor, car

tf.get_logger().setLevel('ERROR')

if __name__ == "__main__":
    if len(sys.argv) == 4:
        cmd, test, func = sys.argv[1], sys.argv[2], sys.argv[3]
        if cmd == "--test":

            if test == "utils":
                if func == "review_image":
                    utils.review_image()
                if func == "resize_image":
                    utils.resize_image()
                if func == "crop_image":
                    utils.crop_image()

            elif test == "humandetection":
                if func == "test_with_camera":
                    humandetection.test_with_camera()
                if func == "test_with_video_1":
                    humandetection.test_with_video(1)
                if func == "test_with_video_2":
                    humandetection.test_with_video(2)
                if func == "test_with_video_3":
                    humandetection.test_with_video(3)
                if func == "test_with_video_4":
                    humandetection.test_with_video(4)

            elif test == "datamanufacture":
                if func == "generate_small_data":
                    datamanufacture.generate_small_data()
                if func == "generate_data":
                    datamanufacture.generate_data()
                if func == "review_source":
                    datamanufacture.review_source()
                if func == "review_hist_data":
                    datamanufacture.review_hist_data()

            elif test == "identitytracking":
                if func == "train":
                    identitytracking.train()
                if func == "predict":
                    identitytracking.predict()

            elif test == "visualization":
                if func == "test_generator":
                    extractor.test_generator()
                if func == "test_pipeline":
                    extractor.test_pipeline()
                if func == "test_96":
                    extractor.test_96()
                if func == "test_224":
                    extractor.test_224()
                if func == "test_inception":
                    extractor.test_inception()

            elif test == "car":
                if func == "test_camera":
                    car.test_camera()
                if func == "test_snapshot":
                    car.test_snapshot()
                if func == "test_action":
                    car.test_action()
                if func == "test_speed":
                    car.test_speed()
                if func == "test_general":
                    car.test_general()

            else:
                print("Error: Test file does not exist.")
        else:
            print("Error: Invalid option!")
    else:
        print("Error: No params detected!")
