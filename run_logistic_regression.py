from check_grad import check_grad
from utils import *
from logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    train_inputs, train_targets = load_train()
    # train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    hyperparameters = {
        "learning_rate": 10 ** -1,
        "weight_regularization": 0.,
        "num_iterations": 1000
    }
    weights = np.reshape([hyperparameters["weight_regularization"]] * (M + 1), (M+1, 1))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################
    # for plotting
    """fig, ax = plt.subplots()
    f_train, f_valid_lst = [], []"""

    for t in range(hyperparameters["num_iterations"]):
        # performance on training set
        f, df, y = logistic(weights, train_inputs, train_targets, hyperparameters)

        # performance on valid set
        f_valid, _, y_valid = logistic(weights, valid_inputs, valid_targets, hyperparameters)

        # gradient descent using training performance
        weights = weights - hyperparameters["learning_rate"] * df

        # for plotting
        """f_train.append(f)
        f_valid_lst.append(f_valid)"""

    # performance on test set
    test_inputs, test_targets = load_test()
    f_test, _, y_test = logistic(weights, test_inputs, test_targets, hyperparameters)

    _, acc_train = evaluate(train_targets, y)
    _, acc_valid = evaluate(valid_targets, y_valid)
    _, acc_test = evaluate(test_targets, y_test)
    print("Train:\ncross entropy: {0}\naccuracy: {1}\n".format(f, acc_train))
    print("Valid:\ncross entropy: {0}\naccuracy: {1}\n".format(f_valid, acc_valid))
    print("Test:\ncross entropy: {0}\naccuracy: {1}\n".format(f_test, acc_test))

    # plotting
    """ax.plot(range(hyperparameters["num_iterations"]), f_train, 'b-', label="train")
    ax.plot(range(hyperparameters["num_iterations"]), f_valid_lst, 'g-', label="valid")
    ax.set_title("mnist_train")
    #ax.set_title("mnist_train_small")
    ax.set_xlabel("number of iterations")
    ax.set_ylabel("average cross entropy")
    ax.legend()
    plt.show()
    """
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()
