from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, y[i]] -= X[i]
                dW[:, j] += X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Every time the margin is grater the 0 and the class is not the correct class,
    # the gradient at the class is added the X and, and at the correct class is subtracted the X.

    # Right now the gradient is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    dW /= num_train

    # Add regularization to the gradient.
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    delta = 1

    # compute score and margins
    scores = X.dot(W)  # [N,C]
    correct_class_scores = scores[range(num_train), y]  # [N, ]
    margins = (scores.T - correct_class_scores).T + delta  # [N, C]

    mask = margins > 0  # [N, C]
    margins[~mask] = 0  # max
    margins[range(num_train), y] = 0  # don't include the correct class

    # Add data loss
    loss += margins.sum() / num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_features = W.shape[0]

    # Lets start by saying that for every sample (N) every class (C) is added 1 X.
    X_per_class = np.repeat(X[:, :, np.newaxis], num_classes, axis=2)  # [N, D, C]

    # Now for every time the wrong class had a margin grater than zero,
    # we need to subtract one X from the correct class score
    # NOTE: since we added one X for all classes (even the correct class), we need to subtract one more.
    #       but since the margin of the true class is always >0, summing over the class dim of the mask
    #       will give the correct amount of times to subtract X.
    X_per_class[range(num_train), :, y] -= (mask.sum(axis=1) * X.T).T

    # the grad is zero when the max operation output zero.
    mask_per_feature = np.repeat(mask[:, np.newaxis, :], num_features, axis=1)  # [N, D, C]
    X_per_class[~mask_per_feature] = 0

    # compute the mean grad over the samples    
    dW += X_per_class.sum(axis=0) / num_train  # [D,C]

    # Add regularization to the gradient.
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
