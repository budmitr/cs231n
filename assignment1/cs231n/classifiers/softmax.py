import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    num_examples = X.shape[0]
    dW = np.zeros_like(W)

    loss = 0.0
    for i in xrange(X.shape[0]): # for every example
        scores = X[i].dot(W)        
        scores -= np.max(scores) # softmax stability
        probs = np.exp(scores) / np.sum(np.exp(scores))
               
        loss -= np.log(probs[y[i]])
        
        ds = probs
        ds[y[i]] -= 1
        dW += np.outer(X[i], ds)
    
    loss /= num_examples
    dW /= num_examples
    
    loss += 0.5 * reg * np.sum(W ** 2)
    dW += reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    num_examples = X.shape[0]
    dW = np.zeros_like(W)

    scores = X.dot(W)
    scores -= np.max(scores, axis=1).reshape((-1,1))
    scores_exp = np.exp(scores)
    probs = scores_exp / np.sum(scores_exp, axis=1).reshape((-1,1))
    
    loss = - np.sum( np.log( probs[np.arange(len(probs)), y] ) ) / num_examples
    loss += 0.5 * reg * np.sum(W ** 2)
    
    probs[np.arange(len(probs)), y] -= 1
    dW = X.T.dot(probs) / num_examples + reg * W

    return loss, dW

