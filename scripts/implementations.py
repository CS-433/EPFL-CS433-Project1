# -*- coding: utf-8 -*-
"""
Implementation of some Machine Learning methods.
The loss is always calculated with MSE.
"""
import numpy as np
from proj1_helpers import *

def compute_loss(y, tx, w):
    """Calculate the loss with MSE."""
    N = tx.shape[0]
    e = y - tx@w
    loss = 1/(2*N)*e.T@e
 
    return loss


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    loss = compute_loss(y, tx, w)
    
    N = tx.shape[0]    
    e = y - tx@w
    dL = -1/N*tx.T@e
    
    return dL, loss


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent."""
    # Initialize parameters
    w = initial_w
    loss = 0

    for n_iter in range(max_iters):
        # Compute gradient and loss
        dL, loss = compute_gradient(y, tx, w)
        
        # Update w by gradient
        w = w - gamma * dL

    return w, loss


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    loss = compute_loss(y, tx, w)
    e = y - tx@w
    dL = -tx.T@e
    
    return dL, loss


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Generate a minibatch iterator for a dataset."""
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent."""
    # Initialize parameters
    w = initial_w
    loss = 0
    
    for n_iter in range(max_iters):
        dL = 0
        
        # Select a mini-batch B -> batch_size is always 1 as it's said in the project description
        # When batch_size=1 -> SGD
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
            dL, loss = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
                    
        # Compute w
        w = w - gamma * dL
        
    return w, loss


def least_squares(y, tx):
    """Least squares regression using normal equations."""
    w = np.linalg.solve(tx.T@tx, tx.T@y)
    loss = compute_loss(y, tx, w)
    
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    N, M = tx.shape
    w = np.zeros(M)
    lambda_pr = 2 * N * lambda_
    
    A = tx.T @ tx + N*lambda_ * np.eye(M)
    b = tx.T @ y
    
    w = np.linalg.solve(A, b)
    loss = compute_loss(y, tx, w)
    
    return w, loss


def sigmoid(t):
    """Apply sigmoid function on parameter t."""
    return 1 / (1 + np.exp(-t))


def calculate_loss(y, tx, w):
    """Compute the loss: negative log likelihood."""
    sigma = sigmoid(tx@w)
    loss = y.T@(np.log(sigma)) + (1 - y).T@(np.log(1 - sigma))

    return np.squeeze(- loss)


def calculate_gradient(y, tx, w):
    """Compute the gradient of loss."""
    sigma = sigmoid(tx@w)
    return tx.T@(sigma-y)


def calculate_hessian(y, tx, w):
    """Return the Hessian of the loss function."""
    pred = sigmoid(tx.dot(w))
    pred = np.diag(pred.T[0])
    r = np.multiply(pred, (1-pred))

    return tx.T.dot(r).dot(tx)


def logistic_regression(y, tx, w):
    """Return the loss, gradient, and Hessian."""
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    hessian = calculate_hessian(y, tx, w)

    return loss, grad, hessian


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    w = w - gamma*grad
    
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent."""
    # Initialize parameters
    w = initial_w
    loss = 0
    
    for i in range(max_iters):
        w, loss = learning_by_gradient_descent(y, tx, w, gamma)

    return w, loss


def penalized_logistic_regression(y, tx, w, lambda_):
    """Return the loss, gradient."""
    loss = calculate_loss(y, tx, w) + lambda_ * w.T@w
    gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    hessian = calculate_gradient(y, tx, w) + 2 * lambda_

    return loss, gradient, hessian


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, grad, hessian = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * grad

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent."""
    # Initialize parameters
    w = initial_w
    loss = 0
    
    for i in range(max_iters):
        w, loss = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        
    return w, loss

def compute_accuracy(y, predictions):
    """Computes the accuracy of the model"""
    n_equals = 0
    for i in range(y.shape[0]):
        if predictions[i] == y[i]:
            n_equals += 1

    return n_equals / y.shape[0]


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, method, **args):
    """
    Completes k-fold cross-validation using the regression method
    passed as argument.
    """
    # Get k'th subgroup in test, others in train
    tr_indices_set = np.delete(k_indices, k, 0).flatten()
    x_tr = x[tr_indices_set]
    y_tr = y[tr_indices_set] 
    
    te_indices = k_indices[k]
    x_te = x[te_indices]
    y_te = y[te_indices]

    # Apply the regression method
    w, loss = method(y=y_tr, tx=x_tr, **args)

    # Predict outputs with the w 
    predictions = predict_labels(w, x_te)

    # Calculate accurancy
    accurancy = compute_accuracy(y_te, predictions)

    return accurancy



def replace_na_values(data):
    """Replace NA values (-999.0) with the mean value of their column."""
    for i in range(data.shape[1]):
            # If NA values in column
            if na(data[:, i]):
                msk = (data[:, i] != -999.)
                # Replace NA values with mean value
                median = np.median(data[msk, i])
                if math. isnan(median):
                    median = 0
                data[~msk, i] = median
    return data


def get_masks(x):
    """Returns 3 masks depending on the number of jets of the event."""
    return {
        0: x[:, 22] == 0,
        1: x[:, 22] == 1,
        2: np.logical_or(x[:, 22] == 2, x[:, 22] == 3)
    }
