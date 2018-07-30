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
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin

        # Compute gradients (one inner and one outer sum)
        # this is really a sum over j != y_i
        # sums each contribution of the x_i's 
        dW[:,y[i]] -= X[i,:] 
        dW[:,j] += X[i,:]      
       
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
    
  # Add regularization derivative to the gradient  
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """

  ## Semi vectorized implementation from lecture notes
  # for i in xrange(num_train):
  #   scores = X[i].dot(W)
  #   correct_class_score = scores[y[i]]
  #   # compute the margins for all classes in one vector operation
  #   margins = np.maximum(0, scores - correct_class_score + delta)
  #   # on y-th position scores[y] - scores[y] canceled and gave delta. We want
  #   # to ignore the y-th position and only consider margin on max wrong class
  #   margins[y[i]] = 0
  #   loss += np.sum(margins)


  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
   
  correct_class_score = scores[np.arange(num_train),y]
  margin = scores - correct_class_score[:, np.newaxis] + 1
  margin[np.arange(num_train), y] = 0
    
  # Compute margin > 0
  thresh = np.maximum(np.zeros((num_train,num_classes)), margin)

  # Compute loss as double sum
  loss = np.sum(thresh)

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  # Binarize into integers
  binary = thresh
  binary[thresh > 0] = 1

  incorrect_counts = np.sum(binary, axis=1)
  binary[np.arange(num_train), y] = -incorrect_counts[np.arange(num_train)]
  dW = X.T.dot(binary)
        
  dW /= num_train
  dW += reg * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
    
  return loss, dW
