import sys

import tensorflow as tf
from tests import utils, humandetection, datamanufacture, shallowRNN, prototype, seq2seq

tf.get_logger().setLevel('ERROR')

if __name__ == "__main__":
    if len(sys.argv) == 4:
        cmd, test, function = sys.argv[1], sys.argv[2], sys.argv[3]
        if cmd == "--test":
            if test == "utils":
                if function == "review_image":
                    utils.review_image()
                if function == "resize_image":
                    utils.resize_image()
                if function == "crop_image":
                    utils.crop_image()
            elif test == "humandetection":
                if function == "with_camera":
                    humandetection.test_with_camera()
                if function == "with_video1":
                    humandetection.test_with_video(1)
                if function == "with_video2":
                    humandetection.test_with_video(2)
                if function == "with_video3":
                    humandetection.test_with_video(3)
                if function == "with_video4":
                    humandetection.test_with_video(4)
            elif test == "datamanufacture":
                if function == "review_data":
                    datamanufacture.review_data()
            elif test == "shallowrnn":
                if function == "train_net":
                    shallowRNN.train_net()
                if function == "run_net":
                    shallowRNN.run_net()
            elif test == "prototype":
                if function == "draw_prototype":
                    prototype.draw_prototype()
            elif test == "seq2seq":
                if function == "train_net":
                    seq2seq.train_net()
                if function == "run_net":
                    seq2seq.run_net()

            else:
                print("Error: Test file does not exist.")
        else:
            print("Error: Invalid option!")
    else:
        print("Error: No params detected!")
