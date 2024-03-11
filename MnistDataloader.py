import numpy as np  # linear algebra
import struct
from array import array
import os
import random
import matplotlib.pyplot as plt

class MnistDataloader(object):
    def __init__(self):
        names = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
        files = ["\\".join([os.getcwd(), "archive", name, name]) for name in names]
        self.train_images_file, self.train_labels_file, self.test_images_file, self.test_labels_file = files

    def read_images_labels(self, images_file, labels_file):
        with open(labels_file, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_file, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        image_func = lambda i: np.array(image_data[i * rows * cols:(i + 1) * rows * cols]).reshape(28, 28)
        images = [image_func(i) for i in range(size)]
        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.train_images_file, self.train_labels_file)
        x_test, y_test = self.read_images_labels(self.test_images_file, self.test_labels_file)
        return (x_train, y_train), (x_test, y_test)

    def show_images(self):
        (x_train, y_train), (x_test, y_test) = self.load_data()
        def index_image_title(i):
            if i < 10:
                r = random.randint(1, 60000)
                return i + 1, x_train[r], "train [{}] = {}".format(r, y_train[r])
            else:
                r = random.randint(1, 10000)
                return i + 1, x_test[r], "test [{}] = {}".format(r, y_test[r])

        size, cols = 15, 5
        rows = int(size / cols) + 1
        plt.figure(figsize=(30, 20))
        for index, image, title in [index_image_title(i) for i in range(15)]:
            plt.subplot(rows, cols, index)
            plt.title(title, fontsize=15)
            plt.imshow(image, cmap="gray", vmin=0, vmax=255)
        plt.show()
