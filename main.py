import sys

from tests import factory, dataset, humandetection, tracker


if __name__ == "__main__":
    if sys.argv[1] == "--test":

        if sys.argv[2] == "humandetection":
            if sys.argv[3] == "test_with_video_1":
                humandetection.test_with_video(1)
            if sys.argv[3] == "test_with_video_2":
                humandetection.test_with_video(2)
            if sys.argv[3] == "test_with_video_3":
                humandetection.test_with_video(3)
            if sys.argv[3] == "test_with_video_4":
                humandetection.test_with_video(4)

        elif sys.argv[2] == "factory":
            if sys.argv[3] == "generate_triplets":
                factory.generate_triplets()
            if sys.argv[3] == "test_generator":
                factory.test_generator()
            if sys.argv[3] == "review_source":
                factory.review_source()

        elif sys.argv[2] == "tracker":
            if sys.argv[3] == "train":
                tracker.train()
            if sys.argv[3] == "convert":
                tracker.convert()
            if sys.argv[3] == "predict":
                tracker.predict()
            if sys.argv[3] == "infer":
                tracker.infer()

        elif sys.argv[2] == "dataset":
            if sys.argv[3] == "show_info":
                dataset.show_info()
            if sys.argv[3] == "benchmark":
                dataset.benchmark()
            if sys.argv[3] == "show_triplets":
                dataset.show_triplets()
            if sys.argv[3] == "show_extractor":
                dataset.show_extractor()
    else:
        print("Error: Invalid option!")
