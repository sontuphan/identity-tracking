import sys

import tensorflow as tf
from tests import utils, humandetection, datamanufacture, classification
from tests import shallowRNN, prototype, seq2seq, identitytracking

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
                if func == "with_camera":
                    humandetection.test_with_camera()
                if func == "with_video1":
                    humandetection.test_with_video(1)
                if func == "with_video2":
                    humandetection.test_with_video(2)
                if func == "with_video3":
                    humandetection.test_with_video(3)
                if func == "with_video4":
                    humandetection.test_with_video(4)

            elif test == "datamanufacture":
                if func == "generate_small_data":
                    datamanufacture.generate_small_data()
                if func == "generate_data":
                    datamanufacture.generate_data()
                if func == "review_source":
                    datamanufacture.review_source()

            elif test == "shallowrnn":
                if func == "train_net":
                    shallowRNN.train_net()
                if func == "run_net":
                    shallowRNN.run_net()

            elif test == "prototype":
                if func == "draw_prototype":
                    prototype.draw_prototype()

            elif test == "seq2seq":
                if func == "train_net":
                    seq2seq.train_net()
                if func == "run_net":
                    seq2seq.run_net()

            elif test == "identitytracking":
                if func == "train":
                    identitytracking.train()
                if func == "predict":
                    identitytracking.predict()

            elif test == "classification":
                if func == "train":
                    classification.train()
                if func == "predict":
                    classification.predict()

            else:
                print("Error: Test file does not exist.")
        else:
            print("Error: Invalid option!")
    else:
        print("Error: No params detected!")
