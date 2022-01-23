import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
      Softmax loss function, naive implementation (with loops)
      This adjusts the weights to minimize loss.

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength. For regularization, we use L2 norm.

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
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
    #############################################################################
    #                     START OF YOUR CODE                                    #
    #############################################################################
    
    
    N,D = X.shape # Number of training samples, Features
    C = W.shape[1] # Number of classes
    
    for i in range(N):
        WX = X[i].dot(W)
        WX_norm=WX-np.max(WX) # Numerical stability,log𝐶=−max𝑓𝑗
        loss_i = -np.log(np.exp(WX_norm[y[i]])/np.sum(np.exp(WX_norm)))
        loss+=loss_i
        for j in range (C):
            P=np.exp(WX_norm[j])/sum(np.exp(WX_norm))
            if j == y[i]: #when yi, dW=-1
                dW[:,j]+=(P-1)*X[i] 
            else: 
                dW[:,j]+=P*X[i]
                
    loss=loss/N
    loss+=reg*np.sum(W*W) # Regularization for loss
    
    dW = dW/N + 2*reg*W # Regularization for dW
    
    #############################################################################
    #                     END OF YOUR CODE                                      #
    #############################################################################
    

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    This adjusts the weights to minimize loss.

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
    #############################################################################
    #                     START OF YOUR CODE                                    #
    #############################################################################
    
    N,D = X.shape # Number of training samples, Features
    C = W.shape[1] # Number of classes

    WX=X.dot(W)
    
    #logC=-max(fj)    
    #max_WX=np.max(WX,axis=1,keepdims=True).reshape(-1,1)#get max score for 0~N
    WX_norm = WX - np.max(WX, axis = 1).reshape(-1, 1) # Normalize
    sftm = np.exp(WX_norm) / np.sum(np.exp(WX_norm), axis = 1).reshape(-1, 1)
    loss = - np.sum(np.log(sftm[range(N), list(y)]))
    
    sftm[range(N), list(y)] += -1
    
    dW = (X.T).dot(sftm)
  
    loss= loss/N
    loss +=reg *np.sum(W*W)
    
    dW = dW/N
    dW += reg*2*W #regularization
    
    #############################################################################
    #                     END OF YOUR CODE                                      #
    #############################################################################
    

    return loss, dW
