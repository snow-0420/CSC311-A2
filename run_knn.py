from l2_distance import l2_distance
from utils import *

import matplotlib.pyplot as plt
import numpy as np


def knn(k, train_data, train_labels, valid_data):
    """ Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples,
          M is the number of features per example.

    :param k: The number of neighbours to use for classification
    of a validation example.
    :param train_data: N_TRAIN x M array of training data.
    :param train_labels: N_TRAIN x 1 vector of training labels
    corresponding to the examples in train_data (must be binary).
    :param valid_data: N_VALID x M array of data to
    predict classes for validation data.
    :return: N_VALID x 1 vector of predicted labels for
    the validation data.
    """
    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:, :k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # Note this only works for binary labels:
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1, 1)

    return valid_labels


def run_knn():
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    #####################################################################
    # TODO:                                                             #
    # Implement a function that runs kNN for different values of k,     #
    # plots the classification rate on the validation set, and etc.     #
    #####################################################################
    fig, ax = plt.subplots()
    accuracies = []
    for k in np.arange(1, 10, 2):
        num_valid = len(valid_inputs)
        pred = knn(k, train_inputs, train_targets, valid_inputs)

        num_correct = 0
        for i in range(num_valid):
            if pred[i] == valid_targets[i]:
                num_correct += 1
        accuracies.append(num_correct / num_valid * 100)

    ax.plot([1, 3, 5, 7, 9], accuracies, 'og-')
    ax.set_xlabel("k")
    ax.set_ylabel("accuracy %")
    ax.set_xticks([1, 3, 5, 7, 9])
    ax.legend()
    plt.show()

    # for q3b only
    """fig, ax = plt.subplots()
    accuracies_valid, accuracies_test = [], []
    for k in range(1, 10, 2):
        num_valid, num_test = len(valid_inputs), len(test_inputs)
        pred_valid = knn(k, train_inputs, train_targets, valid_inputs)
        pred_test = knn(k, train_inputs, train_targets, test_inputs)

        num_correct_valid, num_correct_test = 0, 0
        for i in range(num_valid):
            if pred_valid[i] == valid_targets[i]:
                num_correct_valid += 1
        for i in range(num_test):
            if pred_test[i] == test_targets[i]:
                num_correct_test += 1

        accuracies_valid.append(num_correct_valid / num_valid * 100)
        accuracies_test.append(num_correct_test / num_test * 100)

    ax.plot([1, 3, 5, 7, 9], accuracies_valid, 'og-', label="Validation")
    ax.plot([1, 3, 5, 7, 9], accuracies_test, 'ob-', label="Test")
    ax.set_xlabel("k")
    ax.set_ylabel("accuracy %")
    ax.set_xticks([1, 3, 5, 7, 9])
    ax.legend()
    plt.show()"""
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    run_knn()
