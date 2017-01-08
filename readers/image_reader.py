from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from scipy.misc import imread
from readers import label_util
import tensorflow as tf

IMAGE_WIDTH = 160
IMAGE_HEIGHT = 80
CAPTCHA_LENGTH = 5

# Global constants describing the captcha data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 100
training_folder = "../imgs/"
testing_folder = "../test_imgs/"


def load_training_dataset():
    return load_dataset(training_folder)

def load_testing_dataset():
    no_files = len(os.listdir(testing_folder))
    return load_dataset(testing_folder, 0, no_files)

def normalize_data(X):
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    X = (X - x_mean) / (x_std + 0.00001)
    return X
    
def training_dataset_length():
    return len(os.listdir(training_folder))


def load_dataset(folder, fromPos, toPos):
    file_list = os.listdir(folder)

    X = np.zeros([toPos - fromPos, IMAGE_HEIGHT * IMAGE_WIDTH])
    Y = np.zeros([toPos - fromPos, 5 * NUM_CLASSES])

    for i, filename in enumerate(file_list[fromPos:toPos]):
        path = folder + filename
        img = imread(path, flatten=True)

        captcha_text = filename[0:CAPTCHA_LENGTH]

        X[i, :] = img.flatten()
        Y[i, :] = label_util.words_to_vec(captcha_text)

    X = normalize_data(X)
    return X, Y


def tf_load_dataset(folder):
    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(folder + "/*.jpg"))

    image_reader = tf.WholeFileReader()
    _, image_file = image_reader.read(filename_queue)
    image = tf.image.decode_jpeg(image_file)

    Y = np.zeros([len(filename_queue), 5 * NUM_CLASSES])
