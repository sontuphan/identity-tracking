import sys

from tests import humandetection, datamanufacture, shallowRNN, prototype

if __name__ == "__main__":
    if len(sys.argv) == 4:
        cmd, test, function = sys.argv[1], sys.argv[2], sys.argv[3]
        if cmd == "--test":
            if test == "humandetection":
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

            else:
                print("Error: Test file does not exist.")
        else:
            print("Error: Invalid option!")
    else:
        print("Error: No params detected!")
