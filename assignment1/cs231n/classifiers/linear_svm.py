import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        ds = np.zeros(scores.shape)
        
        correct_class_score = scores[y[i]]
        margins = np.maximum(0, scores - correct_class_score + 1)
        margins[y[i]] = 0
        
        loss += np.sum(margins)
        
        ds[margins > 0] = 1
        ds[y[i]] = - np.sum(ds)
        ds /= num_train
        dW += np.outer(X[i], ds)

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    delta = 1
    num_train = X.shape[0]

    S = X.dot(W)
    correct_class_scores = S[np.arange(len(S)), y]
    margins = S - correct_class_scores.reshape((-1,1)) + delta
    margins[np.arange(len(margins)), y] = 0 # set loss for correct class
    
    max_margins = np.maximum(margins, 0)
    loss = np.sum(max_margins)
    
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    
    # dL / dW = (dL / dS) * (dS / dW)  
    dS = np.zeros(S.shape) # NxC
    dS[margins > 0] = 1 # equals to dL / dSij when j != yi
    dS[np.arange(len(dS)), y] = -np.sum(dS, axis=1) # dL / dSij when j == yi
    dS /= num_train

    dW = X.T.dot(dS) # DxN * NxC  => DxC
    
    # Regularization grad
    dW +=reg * W

    return loss, dW
