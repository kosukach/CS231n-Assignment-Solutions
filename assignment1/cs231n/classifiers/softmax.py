from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    for i in range(num_train):
      scores = W.T.dot(X[i])
      exp_sum = 0
      grad = np.zeros_like(dW)
      for j in range(num_classes):
        exp_sum += np.exp(scores[j])
        grad[:, j] += np.exp(scores[j])
      loss += -scores[y[i]] + np.log(exp_sum)
      grad /= exp_sum
      grad[:, y[i]] -= 1
      grad *= X[i][:, np.newaxis]
      dW += grad

    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += reg * 2 * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    scores = X.dot(W)
    truths = scores[np.arange(num_train), y]
    exps = np.exp(scores)
    exps_sum = np.sum(exps, axis = 1)
    log_sum = np.log(exps_sum)
    loss = np.sum(log_sum - truths, axis=0)
    loss /= num_train
    loss += reg * np.sum(W * W)

    #znam = np.power(exps_sum, -1)[:, np.newaxis]
    
    grads = exps / exps_sum[:, np.newaxis]
    grads[np.arange(num_train), y] -= 1
    dW = X.T.dot(grads)
    dW /= num_train
    dW += reg * 2 * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
