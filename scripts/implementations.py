# -*- coding: utf-8 -*-
"""
Implementation of some Machine Learning methods.
The loss is always calculated with MSE.
"""
import numpy as np


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
    N = tx.shape[0]
    lambda_pr = 2 * N * lambda_
    
    A = tx.T@tx + lambda_pr * np.identity(N)
    b = tx.T@y
    
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


def build_poly(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree."""
    # This function should return the matrix formed
    # by applying the polynomial basis to the input data
    n = x.shape[0]
    phi = np.zeros((n,degree+1))
    for i in range(n):
        for j in range(degree+1):
            phi[i][j] = x[i]**j
    
    return phi

def cross_validation(y, x, k_indices, k, degree, method, lambda_=0):
    """Return the loss of ridge regression."""
    # Get k'th subgroup in test, others in train
    tr_indices_set = np.delete(k_indices, k, 0)
    x_tr = x[tr_indices_set].flatten()
    y_tr = y[tr_indices_set].flatten()
    
    te_indices = k_indices[k]
    x_te = x[te_indices]
    y_te = y[te_indices]
    
    # Form data with polynomial degree
    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)
    
    # Regression on the training set
    if method=='ridge_regression':
        w, loss_tr = ridge_regression(y_tr, tx_tr, lambda_)
    elif method=='least_squares_GD':
        w, loss_tr = least_squares_GD(y_tr, tx_tr, initial_w, max_iters, gamma)
    elif method=='least_squares_SGD':
        w, loss_tr = least_squares_SGD(y_tr, tx_tr, initial_w, max_iters, gamma)
    elif method=='least_squares':
        w, loss_tr = least_squares(y_tr, tx_tr)
    elif method=='logistic_regression':
        w, loss_tr = logistic_regression(y_tr, tx_tr, initial_w, max_iters, gamma)
    elif method=='reg_logistic_regression':
        w, loss_tr = reg_logistic_regression(y_tr, tx_tr, lambda_, initial_w, max_iters, gamma)

    
    # Calculate the loss for train and test data
    loss_te = compute_loss(y_te, tx_te, w)
    
    return loss_tr, loss_te, w
