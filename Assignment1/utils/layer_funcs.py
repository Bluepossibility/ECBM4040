from builtins import range
import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine transformation function.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: a numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: a numpy array of weights, of shape (D, M)
    - b: a numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    """
    ############################################################################
    # TODO: Implement the affine forward pass. Store the result in 'out'. You  #
    # will need to reshape the input into rows.                                #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                                     #
    ############################################################################
    N = x.shape[0] # Number of samples
    D = np.prod(x.shape[1:]) # D = d_1*..*d_k
    x = x.reshape((N,D)) # Reshape each input x into a vector of dimension D = d_1 * ... * d_k
    out = x.dot(w)+b 
    
    ############################################################################
    #                    END OF YOUR CODE                                      #
    ############################################################################

    return out


def affine_backward(dout, x, w, b):
    """
    Computes the backward pass of an affine transformation function.

    Inputs:
    - dout: upstream derivative, of shape (N, M)
    - x: input data, of shape (N, d_1, ... d_k)
    - w: weights, of shape (D, M)
    - b: bias, of shape (M,)

    Returns a tuple of:
    - dx: gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: gradient with respect to w, of shape (D, M)
    - db: gradient with respect to b, of shape (M,)
    """
    ############################################################################
    # TODO: Implement the affine backward pass.                                #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                                     #
    ############################################################################

    N = x.shape[0] # Number of samples
    D = np.prod(x.shape[1:]) # D = d_1*..*d_k
    x = x.reshape((N,D)) # Reshape each input x into a vector of dimension D = d_1 * ... * d_k
    
    dx = dout.dot(w.T) # dx=dout/w [N,M]*[M,D]=[N,D]
    dx = dx.reshape(x.shape) # dx is gradient with respect to x, of shape (N, d1, ..., d_k)
    dw = (x.T).dot(dout)# dw=dout/x [D,N] * [N,M]
    db = np.sum(dout,axis=0) # db=dout [M,]
    
    ############################################################################
    #                    END OF YOUR CODE                                      #
    ############################################################################

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for rectified linear units (ReLUs) activation function.

    Input:
    - x: inputs, of any shape

    Returns a tuple of:
    - out: output, of the same shape as x
    """
    ############################################################################
    # TODO: Implement the ReLU forward pass.                                   #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                                     #
    ############################################################################

    out = np.maximum(x,0) # return x if x>=0; else return 0
    
    ############################################################################
    #                    END OF YOUR CODE                                      #
    ############################################################################

    return out


def relu_backward(dout, x):
    """
    Computes the backward pass for rectified linear units (ReLUs) activation function.

    Input:
    - dout: upstream derivatives, of any shape

    Returns:
    - dx: gradient with respect to x
    """
    ############################################################################
    # TODO: Implement the ReLU backward pass.                                  #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                                     #
    ############################################################################

    dx = dout
    dx[x<0] = 0

    ############################################################################
    #                    END OF YOUR CODE                                      #
    ############################################################################

    return dx


def softmax_loss(x, y):
    """
    Softmax loss function, vectorized version.
    This adjusts the weights to minimize loss.
    y_prediction = argmax(softmax(x))

    Inputs:
    - x: (float) a tensor of shape (N, #classes)
    - y: (int) ground truth label, a array of length N

    Returns:
    - loss: the cross-entropy loss
    - dx: gradients wrt input x
    """
    # Initialize the loss.
    loss = 0.0
    dx = np.zeros_like(x)

    # When calculating the cross entropy,
    # you may meet another problem about numerical stability, log(0)
    # to avoid this, you can add a small number to it, log(0+epsilon)
    epsilon = 1e-15


    ############################################################################
    # TODO: You can use the previous softmax loss function here.               #
    # Hint: Be careful on overflow problem                                     #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                                     #
    ############################################################################

    N = x.shape[0] # Number of samples
    D = np.prod(x.shape[1:]) # D = d_1*..*d_k
    x = x.reshape((N,D)) # Reshape each input x into a vector of dimension D = d_1 * ... * d_k
    
    max_X = np.max(x,axis=1,keepdims=True) # Max weighted X for 0~N
    X = x - max_X # Numerical stability
    
    # X: (float) a tensor of shape (N, C)
    sftm = np.exp(X)/np.sum(np.exp(X),axis=1,keepdims=True) # Make use of previous Softmax loss func
    
    # y: (int) ground truth label, a array of length N
    loss_mtrx = -np.log(sftm[np.arange(N),y]+ epsilon) # To avoid log(0), add a small number to it, log(0+epsilon)
    loss = np.sum(loss_mtrx)/N
    
    dx = sftm
    dx[np.arange(N),y] -= 1
    dx = dx/N
    ############################################################################
    #                    END OF YOUR CODE                                      #
    ############################################################################

    return loss, dx
