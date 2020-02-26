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
        scores = np.dot(X[i],W)
        shifted_scores_by_C = scores - np.max(scores)
        correct_class_score = shifted_scores_by_C[y[i]]
        sum_shifted_expscore = np.sum(np.exp(shifted_scores_by_C))
        prob_i = np.exp(correct_class_score)/sum_shifted_expscore
        loss+=-correct_class_score+np.log(sum_shifted_expscore)
        for j in range(num_classes):
            prob_j = np.exp(shifted_scores_by_C[j])/sum_shifted_expscore
            if j == y[i]:
                dW[:,j] += (prob_i-1)*X[i]
            else: 
                dW[:,j] += prob_j*X[i]
    
    dW/=num_train
    dW+=2*reg*W
    loss/=num_train
    loss+=reg*np.sum(W*W)
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
    num_classes = W.shape[1]
    
    scores = np.dot(X,W)
    shifted_scores = scores - np.max(scores,axis=1).reshape(-1,1)
    softmax_out = np.exp(shifted_scores)/np.sum(np.exp(shifted_scores),axis=1).reshape(-1,1)
    loss =np.mean(-np.log(softmax_out)[range(num_train),y])
    

    softmax_out[range(num_train),y]-=1
    dW = np.dot(softmax_out.T,X).T
    dW/=num_train
#     print(softmax_out.T.shape,X.shape,dW.shape)
    dW +=2*reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
