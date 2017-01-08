from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from scipy.misc import imread
from readers import label_util

IMAGE_WIDTH = 216
IMAGE_HEIGHT = 128
CAPTCHA_LENGTH = 5

# Global constants describing the captcha data set.
NUM_CLASSES = 36
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 100
training_folder = "../data_gen/imgs/"
testing_folder = "../data_gen/test_imgs/"


def load_training_dataset():
    return load_dataset(training_folder)

def load_testing_dataset():
    return load_dataset(testing_folder)

def normalize_data(X):
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    X = (X - x_mean) / (x_std + 0.00001)
    return X

def load_dataset(folder):
    file_list = os.listdir(folder)
    no_files = len(file_list)

    X = np.zeros([no_files, IMAGE_HEIGHT * IMAGE_WIDTH])
    Y = np.zeros([no_files, 5 * NUM_CLASSES])

    for i, filename in enumerate(file_list):
        path = folder + filename
        img = imread(path, flatten=True)

        captcha_text = filename[0:CAPTCHA_LENGTH]

        X[i, :] = img.flatten()
        Y[i, :] = label_util.words_to_vec(captcha_text)

    X = normalize_data(X)
    return X, Y
