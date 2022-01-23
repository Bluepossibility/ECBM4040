#!/usr/bin/env/ python
# ECBM E4040 Fall 2021 Assignment 2
# This Python script contains various functions for layer construction.

import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    :param x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    :param w: A numpy array of weights, of shape (D, M)
    :param b: A numpy array of biases, of shape (M,)

    :return:
    - out: output, of shape (N, M)
    - cache: x, w, b for back-propagation
    """
    num_train = x.shape[0]
    x_flatten = x.reshape((num_train, -1))
    out = np.dot(x_flatten, w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    :param dout: Upstream derivative, of shape (N, M)
    :param cache: Tuple of:
                    x: Input data, of shape (N, d_1, ... d_k)
                    w: Weights, of shape (D, M)

    :return: a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache

    N = x.shape[0]
    x_flatten = x.reshape((N, -1))

    dx = np.reshape(np.dot(dout, w.T), x.shape)
    dw = np.dot(x_flatten.T, dout)
    db = np.dot(np.ones((N,)), dout)

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    :param x: Inputs, of any shape
    :return: A tuple of:
    - out: Output, of the same shape as x
    - cache: x for back-propagation
    """
    out = np.zeros_like(x)
    out[np.where(x > 0)] = x[np.where(x > 0)]

    cache = x

    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    :param dout: Upstream derivatives, of any shape
    :param cache: Input x, of same shape as dout

    :return: dx - Gradient with respect to x
    """
    x = cache

    dx = np.zeros_like(x)
    dx[np.where(x > 0)] = dout[np.where(x > 0)]

    return dx


def softmax_loss(x, y):
    """
    Softmax loss function, vectorized version.
    y_prediction = argmax(softmax(x))

    :param x: (float) a tensor of shape (N, #classes)
    :param y: (int) ground truth label, a array of length N

    :return: loss - the loss function
             dx - the gradient wrt x
    """
    loss = 0.0
    num_train = x.shape[0]

    x = x - np.max(x, axis=1, keepdims=True)
    x_exp = np.exp(x)
    loss -= np.sum(x[range(num_train), y])
    loss += np.sum(np.log(np.sum(x_exp, axis=1)))

    loss /= num_train

    neg = np.zeros_like(x)
    neg[range(num_train), y] = -1

    pos = (x_exp.T / np.sum(x_exp, axis=1)).T

    dx = (neg + pos) / num_train

    return loss, dx


def conv2d_forward(x, w, b, pad, stride):
    """
    A Numpy implementation of 2-D image convolution.
    By 'convolution', simple element-wise multiplication and summation will suffice.
    The border mode is 'valid' - Your convolution only happens when your input and your filter fully overlap.
    Another thing to remember is that in TensorFlow, 'padding' means border mode (VALID or SAME). For this practice,
    'pad' means the number rows/columns of zeroes to concatenate before/after the edge of input.

    Inputs:
    :param x: Input data. Should have size (batch, height, width, channels).
    :param w: Filter. Should have size (filter_height, filter_width, channels, num_of_filters).
    :param b: Bias term. Should have size (num_of_filters, ).
    :param pad: Integer. The number of zeroes to pad along the height and width axis.
    :param stride: Integer. The number of pixels to move between 2 neighboring receptive fields.

    :return: A 4-D array. Should have size (batch, new_height, new_width, num_of_filters).

    Note:
    To calculate the output shape of your convolution, you need the following equations:
    new_height = ((height - filter_height + 2 * pad) // stride) + 1
    new_width = ((width - filter_width + 2 * pad) // stride) + 1
    For reference, visit this website:
    https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/
    """
    #######################################################################
    #                         TODO: YOUR CODE HERE                        #
    #######################################################################
    # x, a 4-D array: Input data. Should have size (batch, height, width, channels).(2, 5, 5, 3)
    batch, height, width, channels = x.shape
    # w, a 4-D array: Filter. Should have size (filter_height, filter_width, channels, num_of_filters).(3, 3, 3, 5)
    filter_height, filter_width, flt_channels, num_of_flt = w.shape
    
    # Do padding before convolution
    # Initialize padding with an zero 4-D array bigger than x
    padding=np.zeros([batch,height+pad*2,width+pad*2,channels])
    # Assign values of x to padding respectively
    padding[:,pad:pad+height,pad:pad+width,:]=x
    #p_batch, pad_height, pad_width, p_channels=padding.shape
    
    # Calculate the output shape after convolution, in this case, pad=1, stride=2
    new_height = ((height - filter_height + 2 * pad) // stride) + 1
    new_width = ((width - filter_width + 2 * pad) // stride) + 1
    
    # Initializing output 4-D array
    out=np.zeros([batch,new_height,new_width,num_of_flt])
    # Works for all batches
    for num_batch in range(batch):
        # For every grid in output 2-D array out of output 4-D array
        for i in range(new_height):
            for j in range(new_width):
                # Calculate the right position in 2-D array of x
                x_i=i*stride
                x_j=j*stride
                # Works for all filters
                for num_filter in range(num_of_flt):
                    # Weight for this particular grid
                    W=w[:, :, :, num_filter]
                    # Bias for this particular grid
                    B=b[num_filter,]
                    # Finally, Convolution with fitlers,w&b based on the padding!
                    out[num_batch, i, j, num_filter]=np.sum(padding[num_batch,x_i:x_i+filter_height,x_j:x_j+filter_width, :]*W)+B
    return out

    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################


def conv2d_backward(d_top, x, w, b, pad, stride):
    """
    (Optional, but if you solve it correctly, we give you 5 points for this assignment.)
    A lite Numpy implementation of 2-D image convolution back-propagation.

    Inputs:
    :param d_top: The derivatives of pre-activation values from the previous layer
                       with shape (batch, height_new, width_new, num_of_filters).
    :param x: Input data. Should have size (batch, height, width, channels).
    :param w: Filter. Should have size (filter_height, filter_width, channels, num_of_filters).
    :param b: Bias term. Should have size (num_of_filters, ).
    :param pad: Integer. The number of zeroes to pad along the height and width axis.
    :param stride: Integer. The number of pixels to move between 2 neighboring receptive fields.

    :return: (d_w, d_b), i.e. the derivative with respect to w and b. For example, d_w means how a change of each value
     of weight w would affect the final loss function.

    Note:
    Normally we also need to compute d_x in order to pass the gradients down to lower layers, so this is merely a
    simplified version where we don't need to back-propagate.
    For reference, visit this website:
    http://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
    """
    #######################################################################
    #                         TODO: YOUR CODE HERE                        #
    #######################################################################
    
    
    
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    
def avg_pool_forward(x, pool_size, stride):
    """
    A Numpy implementation of 2-D image average pooling.

    Inputs:
    :params x: Input data. Should have size (batch, height, width, channels).
    :params pool_size: Integer. The size of a window in which you will perform average operations.
    :params stride: Integer. The number of pixels to move between 2 neighboring receptive fields.
    :return :A 4-D array. Should have size (batch, new_height, new_width, num_of_filters).
    """
    #######################################################################
    #                         TODO: YOUR CODE HERE                        #
    #######################################################################
    
    batch, height, width, channels=x.shape

    #calculate output size:
    new_height = int(np.floor(height/stride))
    new_width = int(np.floor(width/stride))
    #init output
    out=np.zeros([batch,new_height,new_width,channels])

    #max pooling
    for num_batch in range(batch):
        for c in range(channels):
            for i in range(new_height):
                for j in range(new_width):
                    x_i=i*stride
                    x_j=j*stride
                    out[num_batch,i,j,c]=np.mean(x[num_batch,x_i:x_i+pool_size,x_j:x_j+pool_size,c])
    return out
    
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    
def avg_pool_backward(dout, x, pool_size, stride):
    """
    (Optional, but if you solve it correctly, we give you +5 points for this assignment.)
    A Numpy implementation of 2-D image average pooling back-propagation.

    Inputs:
    :params dout: The derivatives of values from the previous layer
                       with shape (batch, height_new, width_new, num_of_filters).
    :params x: Input data. Should have size (batch, height, width, channels).
    :params pool_size: Integer. The size of a window in which you will perform average operations.
    :params stride: Integer. The number of pixels to move between 2 neighboring receptive fields.
    
    :return dx: The derivative with respect to x
    You may find this website helpful:
    https://medium.com/the-bioinformatics-press/only-numpy-understanding-back-propagation-for-max-pooling-layer-in-multi-layer-cnn-with-example-f7be891ee4b4
    """
    #######################################################################
    #                         TODO: YOUR CODE HERE                        #
    #######################################################################
    
    print('./utils/layer_funcs.avg_pool_backward() not implemented!') # delete me
    
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################
