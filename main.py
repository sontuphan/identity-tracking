import sys

import tensorflow as tf
from tests import utils, datamanufacture, identitytracking

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

            elif test == "datamanufacture":
                if func == "generate_small_data":
                    datamanufacture.generate_small_data()
                if func == "generate_data":
                    datamanufacture.generate_data()
                if func == "review_source":
                    datamanufacture.review_source()

            elif test == "identitytracking":
                if func == "train":
                    identitytracking.train()
                if func == "predict":
                    identitytracking.predict()

            else:
                print("Error: Test file does not exist.")
        else:
            print("Error: Invalid option!")
    else:
        print("Error: No params detected!")
