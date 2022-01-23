import numpy as np
import tensorflow as tf

# Realize common spectral pooling
def Common_Spectral_Pool(pictures, filter_size):
    """pictures must have a shape of NHWC"""
    input_shape = pictures.get_shape().as_list()
    assert len(input_shape) == 4
    _, H, W, _ = input_shape
    assert H == W
    """input filter_size must be int type"""
    assert type(filter_size) is int
    """input filter_size must be larger than 3"""
    assert filter_size >= 3

    """tf.concat(values, axis, name='concat')"""
    """Concatenates tensors along one dimension."""
    """Negative axis are interpreted as counting from the end of the rank."""

    """tf.expand_dims(input, axis, name=None)"""
    """tf.expand_dims: Returns a tensor with a length 1 axis inserted at index axis."""
    """A negative axis counts from the end so axis=-1 adds an inner most dimension:"""

    """tf.cast(x, dtype, name=None)"""
    """Casts a tensor to a new type."""
    """print(tf.cast(0.5 ** 0.5, tf.complex64))
    ---tf.Tensor((0.70710677+0j), shape=(), dtype=complex64)"""
####################################################################################################################
    # if filter size is  an even number
    if filter_size % 2 == 0:
        m = filter_size / 2
        """When filter size is even, divide pictures to 9 parts, 
        top_left, top_middle, top_right, 
        middle_left, middle_middle, middle_right, 
        bottom_left, bottom_middle, bottom_right respectively"""
        root_point_five=pow(0.5,0.5)
        top_left = pictures[:, :, :m, :m]

        top_middle = \
            tf.expand_dims(tf.cast(root_point_five, tf.complex64) * (pictures[:,:m, m,:] + pictures[:,:m,-m,:]), -1)

        top_right = pictures[:,:m,-(m-1):,:]

        middle_left = \
            tf.expand_dims(tf.cast(root_point_five, tf.complex64) * (pictures[:,m,:m,:] + pictures[:,-m,:m,:]),-2)

        middle_middle = \
            tf.expand_dims(
                tf.expand_dims(
                    tf.cast(0.5, tf.complex64)*(
                            pictures[:,m,m,:] + pictures[:,m,-m,:] +pictures[:,-m,m,:] + pictures[:,-m,-m,:]),-1),
            -1
        )

        middle_right = \
            tf.expand_dims(tf.cast(root_point_five, tf.complex64) * (pictures[:,m,-(m-1):,:] + pictures[:,-m,-(m-1):,:]),-2)

        bottom_left = pictures[:,-(m-1):,:m,:]

        bottom_middle = \
            tf.expand_dims(tf.cast(root_point_five, tf.complex64) * (pictures[:,-(m-1):,m,:] + pictures[:,-(m-1):,-m,:]),-1)

        bottom_right = pictures[:,-(m-1):,-(m-1):,:]
        """Combine all separate 9 parts"""
        top_combined = tf.concat([top_left, top_middle, top_right],axis=-2)  # NHWC, at width axis
        middle_combined = tf.concat([middle_left, middle_middle, middle_right],axis=-2)  # NHWC, at width axis
        bottom_combined = tf.concat([bottom_left, bottom_middle, bottom_right],axis=-2)  # NHWC, at width axis
        combine_all = tf.concat([top_combined, middle_combined, bottom_combined],axis=-3)  # NHWC, at height axis
####################################################################################################################
    # if filter size is an odd number
    if filter_size % 2 == 1:
        m = filter_size // 2
        """When filter size is odd, divide pictures to 4 parts, top_left, top_right, bottom_left, bottom_right respectively"""
        top_left = pictures[:,:m+1,:m+1,:]
        top_right = pictures[:,:m+1,-m:,:]
        bottom_left = pictures[:,-m:,:m+1,:]
        bottom_right = pictures[:,-m:,-m:,:]
        """Combine all 4 separate parts"""
        top_combined = tf.concat([top_left, top_right], axis=-2)  # NHWC, at width axis
        bottom_combined = tf.concat([bottom_left, bottom_right], axis=-2)  # NHWC, at width axis
        combine_all = tf.concat([top_combined, bottom_combined], axis=-3)  # NHWC, at height axis

    return combine_all


"""Shift the zero-frequency component to the center of the spectrum."""
# Fourier Shift
def tf_fftshift(matrix, n):
    """Performs similar function to numpy's fftshift
        Note: Takes image as a channel first numpy array of shape:
            (batch_size, height, width, channels)
        """
    """Fourier Shift, don't inverse, realize shift on axis=1 of the spectrum"""
    cut_point = (n + 1) // 2
    head = [0, 0, cut_point, 0]
    tail = [-1, -1, cut_point, -1]
    slice1 = tf.slice(matrix, head, [-1, -1, -1, -1])
    slice2 = tf.slice(matrix, [0, 0, 0, 0], tail)
    matrix_ = tf.concat([slice1, slice2], axis + 1)
    """Based on the matrix_ realize shift on axis=0 of the spectrum"""
    head = [0, cut_point, 0, 0]
    tail = [-1, cut_point, -1, -1]
    slice1 = tf.slice(matrix_, head, [-1, -1, -1, -1])
    slice2 = tf.slice(matrix_, [0, 0, 0, 0], tail)
    matrix__=tf.concat([slice1, slice2], axis + 1)

    return matrix__

# Inverse Fourier Shift
def tf_ifftshift(matrix, n):
    """Performs similar function to numpy's ifftshift
    Note: Takes image as a channel first numpy array of shape:
        (batch_size, channels, height, width)
    """
    """Fourier Shift, don't inverse, realize shift on axis=1 of the spectrum"""
    cut_point = n - (n + 1) // 2
    head = [0, 0, cut_point, 0]
    tail = [-1, -1, cut_point, -1]
    slice1 = tf.slice(matrix, head, [-1, -1, -1, -1])
    slice2 = tf.slice(matrix, [0, 0, 0, 0], tail)
    matrix_ = tf.concat([slice1, slice2], axis + 1)
    """Based on the matrix_ realize shift on axis=0 of the spectrum"""
    head = [0, cut_point, 0, 0]
    tail = [-1, cut_point, -1, -1]
    slice1 = tf.slice(matrix_, head, [-1, -1, -1, -1])
    slice2 = tf.slice(matrix_, [0, 0, 0, 0], tail)
    matrix__ = tf.concat([slice1, slice2], axis + 1)

    return matrix__

def spectral_pool(image, filter_size=3,
                  return_fft=False,
                  return_transformed=False,
                  ):
    """ Perform a single spectral pool operation.
    Args:
        image: numpy array representing an image, channels last
            shape: (batch_size, height, width, channel)
        filter_size: the final dimension of the filter required
        return_fft: bool, if True function also returns the raw
                          fourier transform
    Returns:
        An image of same shape as input
    """
    # pad zeros to the image
    # this is required only when we're visualizing the image and not in
    # the final spectral layer
    # required to handle odd and even image size
    # offset = int((dim + 1 - filter_size) / 2)
    # im_pad = tf.image.pad_to_bounding_box(im_cropped, offset, offset, dim, dim)
    # im_pad = im_cropped
    img_fft = tf.signal.fft2d(tf.cast(image, tf.complex64))
    img_transformed = Common_Spectral_Pool(img_fft, filter_size)
    # perform ishift and take the inverse fft and throw img part
    # Computes the inverse 2-dimensional discrete Fourier transform
    img_ifft = tf.math.real(tf.signal.ifft2d(img_transformed))
    # normalize image:
    channel_max = tf.math.reduce_max(input_tensor=img_ifft, axis=(0, 1, 2))
    channel_min = tf.math.reduce_min(input_tensor=img_ifft, axis=(0, 1, 2))
    img_out = tf.math.divide(img_ifft - channel_min,
                       channel_max - channel_min)
    #returns result of fast fourier transformation, returns the raw fourier transform
    if return_fft:
        return img_fft, img_out
    #return result of fast fourier transformation
    elif return_transformed:
        return img_transformed, img_out
    #return result of fast fourier transformation
    else:
        return img_out


def max_pool(image, pool_size=(2,2)):
    max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size,strides=(1, 1), padding='valid')
    return max_pool_2d(image)

def max_pool_1(image, pool_size=(2,2)):
    max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size,strides=(1, 1), padding='same')
    return max_pool_2d(image)


def l2_loss_images(orig_images, mod_images):
    """Calculates the loss for a set of modified images vs original
    formular: l2(orig-mod)/l2(orig)
    Args:
        orig_images: numpy array size (batch, dims..)
        mod_images: numpy array of same dim as orig_images
    Returns:
        single value, i.e. loss
    """
    n = orig_images.shape[0]
    # convert to 2d:
    orig_img = orig_images.reshape(n, -1)
    mod_img = mod_images.reshape(n, -1)
    # bring to same scale if the two scales not already
    if orig_img.max() > 2:
        orig_img = orig_img / 255.
    if mod_img.max() > 2:
        mod_img = mod_img / 255.
    # calculate error and base, perform normalization
    error_norm = np.linalg.norm(orig_img - mod_img, axis=0)
    base_norm = np.linalg.norm(orig_img, axis=0)
    return np.mean(error_norm / base_norm)


def l2_loss_images_1(orig_images, mod_images):
    """Calculates the loss for a set of modified images vs original
    formular: l2(orig-mod)/l2(orig)
    Args:
        orig_images: numpy array size (batch, dims..)
        mod_images: numpy array of same dim as orig_images
    Returns:
        single value, i.e. loss
    """
    n = orig_images.shape[0]
    # convert to 2d:
    orig_img = orig_images.reshape(n, -1)
    mod_img = tf.reshape(mod_images,[n, -1])
    # print(type(orig_img),type(mod_img))
    # bring to same scale if the two scales not already
    if orig_img.max() > 2:
        orig_img = orig_img / 255.
    if tf.math.reduce_max(mod_img) > 2:
        mod_img = mod_img / 255.
    # calculate error and base, perform normalization
    error_norm = np.linalg.norm(orig_img - mod_img, axis=0)
    base_norm = np.linalg.norm(orig_img, axis=0)
    return np.mean(error_norm / base_norm)