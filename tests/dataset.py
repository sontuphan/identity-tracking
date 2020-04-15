import matplotlib.pyplot as plt
import time
import tensorflow as tf

from src.dataset import Dataset
from src.tracker import Extractor


def show_info():
    ds = Dataset()
    ds.print_dataset_info()


def benchmark():
    ds = Dataset()

    start = time.time()

    steps_per_epoch = 0
    pipeline = ds.pipeline()
    iterator = iter(pipeline)
    try:
        while True:
            substart = time.time()
            imgs, bboxes = next(iterator)
            print('*** Batch number', steps_per_epoch)
            print('*** Batch of images:', imgs.shape)
            print('*** Batch of bouding boxes:', bboxes.shape)
            steps_per_epoch += 1
            subend = time.time()
            print('*** Step estimated time %fs' % (subend-substart))
    except StopIteration:
        pass

    end = time.time()
    print('*** Total estimated time %fs' % (end-start))


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


def show_extractor():
    extractor = Extractor()
    loss_metric = tf.keras.metrics.Mean(name='train_loss')
    ds = Dataset(batch_size=256, image_shape=(96, 96))
    pipeline = ds.pipeline()
    iterator = iter(pipeline)

    start = time.time()
    step = 0
    try:
        while True:
            print('Step %d ==================================' % step)
            imgs, bboxes = next(iterator)
            ais, pis, nis = tf.split(imgs, [1, 1, 1], axis=1)
            ais = tf.reshape(
                ais, [ds.batch_size, ds.image_shape[0], ds.image_shape[1], 3])
            pis = tf.reshape(
                pis, [ds.batch_size, ds.image_shape[0], ds.image_shape[1], 3])
            nis = tf.reshape(
                nis, [ds.batch_size, ds.image_shape[0], ds.image_shape[1], 3])

            afs = extractor(ais)
            pfs = extractor(pis)
            nfs = extractor(nis)

            print('afs', afs.numpy())
            print('pfs', pfs.numpy())
            print('nfs', nfs.numpy())

            lloss = tf.sqrt(tf.reduce_sum(tf.square(afs - pfs), 1))
            rloss = tf.sqrt(tf.reduce_sum(tf.square(afs - nfs), 1))
            loss = tf.reduce_mean(tf.maximum(lloss - rloss + 13, 0))

            print('lloss', lloss.numpy())
            print('rloss', rloss.numpy())
            print('loss', loss.numpy())

            loss_metric(loss)
            step += 1
    except StopIteration:
        pass

    print('*** Loss Metric {:.4f}'.format(loss_metric.result()))
    end = time.time()
    print('*** Step estimated time {:.4f}s'.format(end-start))
