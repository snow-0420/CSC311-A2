# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist



#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    ## TODO
    # compute a^i for every i
    size_train = x_train.shape[0]
    values = np.zeros(size_train)
    denominator = 0
    for i in range(size_train):
        numerator = np.exp(-(l2(test_datum.reshape(1, -1), x_train[i, :].reshape(1, -1)) / (2 * (tau ** 2))))
        values[i] = numerator
        denominator += numerator
    values = values / denominator
    A = np.diag(values[0])

    # compute the optimal weights using direct solution in part (a)
    temp = np.matmul(x_train.transpose(), A)
    temp = np.matmul(temp, x_train)
    temp = temp + np.diag([lam] * d)
    temp = np.linalg.inv(temp)
    temp = np.matmul(temp, x_train.transpose())
    temp = np.matmul(temp, A)
    w_star = np.matmul(temp, y_train)
    # w_star = np.einsum('aj,ji,ik,k->a', np.linalg.inv(np.einsum('ai,ij,jx->ax', x_train.transpose(), A, x_train) + np.diag([lam] * d)), x_train.transpose(), A, y_train)

    # compute prediction
    pred = np.dot(test_datum, w_star)

    return pred
    ## TODO




def run_validation(x,y,taus,val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    ## TODO
    # split the data set into training and valid set
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=val_frac)
    # initialize return vector
    loss_train_lst, loss_valid_lst = [], []

    for tau in taus:
        loss_train, loss_valid = 0, 0
        for i in range(x_train.shape[0]):
            pred_train = LRLS(x_train[i], x_train, y_train, tau)
            loss_train += (y_train[i] - pred_train) ** 2
        loss_train = loss_train / x_train.shape[0]

        for i in range(x_valid.shape[0]):
            pred_valid = LRLS(x_valid[i], x_train, y_train, tau)
            loss_valid += (y_valid[i] - pred_valid) ** 2
        loss_valid = loss_valid / x_valid.shape[0]

        loss_train_lst.append(loss_train)
        loss_valid_lst.append(loss_valid)

    return loss_train_lst, loss_valid_lst
    ## TODO


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    train_losses, test_losses = run_validation(x,y,taus,val_frac=0.3)
    plt.semilogx(taus, train_losses, label="train loss")
    plt.semilogx(taus, test_losses, label="test loss")
    plt.xlabel("tau values")
    plt.ylabel("total loss")
    plt.legend()
    plt.show()
