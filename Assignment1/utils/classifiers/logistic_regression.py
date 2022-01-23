import numpy as np
from random import shuffle

def sigmoid(x):
    """Sigmoid function implementation"""
    h = np.zeros_like(x)
    
    #############################################################################
    # TODO: Implement sigmoid function.                                         #         
    #############################################################################
    #############################################################################
    #                          START OF YOUR CODE                               #
    #############################################################################

    h = 1 / (1 + np.exp(-x))
    
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return h 

def logistic_regression_loss_naive(W, X, y, reg):
    """
      Logistic regression loss function, naive implementation (with loops)
      Use this linear classification method to find optimal decision boundary.

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where c can be either 0 or 1.
      - reg: (float) regularization strength. For regularization, we use L2 norm.

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
    """
    # Set the loss to a random number
    loss = 0
    # Initialize the gradient to zero
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.    #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                        #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    
    #𝐿𝑖=−(𝑦𝑖log(𝜎(𝑓𝑖))+(1−𝑦𝑖)log(1−𝜎(𝑓𝑖))) Naive Implementation
    #𝐿=1𝑁∑𝑖𝐿𝑖+𝑟𝑒𝑔×‖𝑊‖2
    
    N,D = X.shape # Number of training samples, Features
    C = W.shape[1] # Number of classes
    
    for i in range(N):
        WX=X[i].dot(W)
        sigmoid=1 / (1 + np.exp(-WX))
        L1=y[i]*(np.log(sigmoid))
        L2=(1-y[i])*(np.log(1-sigmoid))
        loss += (-L1-L2)/N
        
        #∂𝐿𝑖/∂𝑊=−(𝑦𝑖−𝜎(𝑓𝑗))∗𝑥𝑖
        dW += -(y[i]-sigmoid)*(X[i].reshape(-1,1))/N
        
    regW=reg*np.sqrt(sum(W[i].dot(W[i].T) for i in range(D)))
    loss+=regW
    dW+=2*reg*W
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################

    return loss, dW



def logistic_regression_loss_vectorized(W, X, y, reg):
    """
    Logistic regression loss function, vectorized version.
    Use this linear classification method to find optimal decision boundary.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Set the loss to a random number
    loss = 0
    # Initialize the gradient to zero
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the logistic regression loss and its gradient using no     # 
    # explicit loops.                                                          #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the       #
    # regularization!                                                          #
    ############################################################################
    #############################################################################
    #                          START OF YOUR CODE                               #
    #############################################################################
    
    #𝐿𝑖=−(𝑦𝑖log(𝜎(𝑓𝑖))+(1−𝑦𝑖)log(1−𝜎(𝑓𝑖))) Vector Implementation
    #𝐿=1𝑁∑𝑖𝐿𝑖+𝑟𝑒𝑔×‖𝑊‖2
    
    N,D = X.shape # Number of training samples, Features
    C = W.shape[1] # Number of classes
    
    WX = X.dot(W)
    sigmoid = 1 / (1 + np.exp(-WX))
    L1=y.dot(np.log(sigmoid))
    L2=(np.ones(y.shape)-y).dot(np.log(np.ones(sigmoid.shape)-sigmoid))
    L=-L1-L2
    regW=reg*np.sqrt(sum(W[i].dot(W[i].T) for i in range(D)))
    loss=(np.sum(L[range(C)]))/N+regW
    
    #∂𝐿/∂𝑊=−(𝑦−𝜎(𝑓))∗X
    
    #dev = (y-sigmoid).dot(X)
    #dev = dev.sum(axis=0)
    #dW = dev.T
    
    dW = (X.T).dot(y.reshape(-1,1)-sigmoid)
    
    #z=y-sigmoid
    #q=y.reshape(-1,1)-sigmoid
    #dW[1],dW[2]=z.shape
    #dW[3],dW[4]=q.shape
    dW = dW/N
    dW = -dW
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    

    return loss, dW
