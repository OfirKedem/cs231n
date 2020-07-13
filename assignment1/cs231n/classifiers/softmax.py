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
        scores = np.zeros(num_classes)  # [C, ]

        # compute scores
        for c in range(num_classes):
            scores[c] = X[i].dot(W[:, c])

        scores -= scores.max()  # shift by max for numeric stability
        probs = np.exp(scores) / np.sum(np.exp(scores))  # [C, ]

        # add data loss
        loss -= np.log(probs[y[i]])

        # add derivative of data
        for c in range(num_classes):
            dW[:, c] += probs[c] * X[i]
            if c == y[i]:
                dW[:, c] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    # Add regularization to the derivative
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

    # -------- COMPUTE LOSS -----------
    num_train = X.shape[0]
    num_classes = W.shape[1]

    scores = X.dot(W)  # [N, C]
    scores -= scores.max()
    probs = (np.exp(scores).T / np.sum(np.exp(scores), axis=1)).T  # [N, C]

    loss -= np.mean(np.log(probs[range(num_train), y]), axis=0)

    # add regularization
    loss += reg * np.sum(W * W)

    # -------- COMPUTE GRADIENT ----------
    dW += X.T.dot(probs) / num_train

    # one-hot y
    one_hot_y = np.zeros([num_train, num_classes])
    one_hot_y[range(num_train), y] = 1
    dW -= X.T.dot(one_hot_y) / num_train

    # add reg to grad
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
