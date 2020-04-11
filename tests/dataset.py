import matplotlib.pyplot as plt
import time

from src.dataset import Dataset


def show_info():
    ds = Dataset()
    ds.print_dataset_info()


def benchmark():
    ds = Dataset()
    start = time.time()
    pipeline = ds.pipeline()
    batch = next(iter(pipeline))
    print('*** Batch of images:', batch[0].shape)
    print('*** Batch of bouding boxes:', batch[1].shape)
    end = time.time()
    print('*** Estimated time %fs' % (end-start))


def show_triplets():
    ds = Dataset()
    pipeline = ds.pipeline()
    batch = next(iter(pipeline))
    batch_imgs = batch[0]
    plt.figure(figsize=(10, 10))
    for i in range(5):
        imgs = batch_imgs[i]
        for j, img in enumerate(imgs):
            plt.subplot(5, 3, 3*i+j+1)
            plt.imshow(img.numpy())
            plt.axis('off')
    plt.show()
