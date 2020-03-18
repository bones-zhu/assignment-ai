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
        scores = np.dot(X[i], W)
        scores -= np.max(scores)  #为了避免计算越界进行的处理
        correct_score = scores[y[i]]
        sum_scores = np.sum(np.exp(scores))
        porb = np.exp(correct_score)/sum_scores
        loss += -np.log(porb)
        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] = dW[:, j] - X[i, :]
            porb = np.exp(scores[j])/sum_scores
            dW[:, j] = dW[:, j] + X[i, :] * porb
    
    loss /= num_train
    loss += reg * np.sum(W * W)
    dW = dW / num_train
    dW += 2* reg * W
            

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
    scores = np.dot(X, W)
    scores -= np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    sum_scores = np.sum(exp_scores, axis=1, keepdims=True)
    porb = exp_scores / sum_scores
    real_porb = porb[np.arange(num_train), y]
    loss -= np.sum(np.log(real_porb))
    loss /= num_train
    loss += reg * np.sum(W * W)
    binary = porb
    binary[np.arange(num_train), y] -= 1
    dW = dW + np.dot(X.T, binary)
    dW = dW / num_train
    dW = dW + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW