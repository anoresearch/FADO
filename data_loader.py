import numpy as np
import random
from tensorflow.keras.datasets import cifar10

def load_and_preprocess_data(class_a, class_b, downsample_factor=25):
    """
    Load CIFAR-10 data, filter for two classes, and downsample one class to create imbalance.
    :param class_a: First class to include.
    :param class_b: Second class to include.
    :param downsample_factor: Factor to reduce the size of the second class.
    :return: Preprocessed train and test datasets (x_train, y_train, x_test, y_test).
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Filter for the specified classes
    train_filter = (y_train == class_a) | (y_train == class_b)
    test_filter = (y_test == class_a) | (y_test == class_b)
    x_train, y_train = x_train[train_filter.flatten()], y_train[train_filter.flatten()]
    x_test, y_test = x_test[test_filter.flatten()], y_test[test_filter.flatten()]

    # Convert labels to binary (class_a: 0, class_b: 1)
    y_train = (y_train == class_b).astype(int)
    y_test = (y_test == class_b).astype(int)

    # Downsample class_b to create imbalance
    class_b_indices = np.where(y_train == 1)[0]
    reduced_class_b_indices = random.sample(list(class_b_indices), len(class_b_indices) // downsample_factor)
    keep_indices = np.where(y_train == 0)[0].tolist() + reduced_class_b_indices
    x_train, y_train = x_train[keep_indices], y_train[keep_indices]

    # Normalize the image data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    return x_train, y_train, x_test, y_test
