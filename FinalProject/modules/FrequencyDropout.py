"""Implement a frequency dropout."""
import numpy as np
import tensorflow as tf

def freq_dropout_mask(size, truncate_threshold):
    """Create a mask for frequency dropout.

        Args:
            size: int. the height of the image to create a mask for.
                For a 32x32 image, this should be 32.
            truncate_threshold: scalar. Tensor of shape (,). All
                frequencies above this will be set to zero. For an image with
                a height of 32, a number above 16 will have no effect. For an
                image with a height of 31, an input above 15 will have no effect.

        Returns:
            dropout_mask: Tensor of shape (height, height)
                The result can be multiplied by the FFT of an image to create
                a modified FFT where all frequencies above the cutoff have
                been set to zero. Therefore, the value of the mask will be 1
                for the frequencies below the truncation level, and 0 for the
                frequencies above it. In other words, it is really the mask
                of values to retain, not the mask of values to drop.
        """
    truncate_threshold_shape = truncate_threshold.get_shape().as_list()
    assert len(truncate_threshold_shape) == 0

    half_low = size // 2  # round down
    if size % 2 == 1:
        half_up = half_low + 1
    else:
        half_up = half_low

    indice_mask = np.concatenate((np.arange(half_up) , np.arange(half_low, 0, -1))).astype(np.float32)

    x_spread = np.broadcast_to(indice_mask, (size, size))
    y_spread = np.broadcast_to(np.expand_dims(indice_mask, -1), (size, size))
    highest_freq = np.maximum(x_spread, y_spread)

    dropout_mask = tf.cast(tf.less_equal(highest_freq, truncate_threshold), tf.complex64)

    return dropout_mask


def freq_dropout_test(images, truncate_threshold):
    """Demonstrate the use of _frequency_dropout_mask.

        Args:
            images: n-d array of shape (num_images, height, width, num_channels)
            truncate_threshold: Tensor of shape (,) (i.e. scalar). All
                frequencies above this will be set to zero. For an image with
                a height of 32, a number above 16 will have no effect. For an
                image with a height of 31, an input above 15 will have no effect.

        Returns:
            sample_images: n-d array of shape (num_images, height, width, num_channels).
        """
    assert len(images.shape) == 4
    N, H, W, C = images.shape
    assert H == W

    frq_dp_msk = freq_dropout_mask(H, truncate_threshold)

    tf_images = tf.constant(images, dtype=tf.complex64)
    tf_images = tf.squeeze(tf_images)

    if len(tf_images.shape)==2:
        fft_images = tf.signal.fft2d(tf_images)
        trunc_images = tf.math.multiply(fft_images,frq_dp_msk)
        sample_images = tf.math.real(tf.signal.ifft2d(trunc_images))

    if len(tf_images.shape)==3:
        fft_images1 = tf.signal.fft2d(tf_images[:,:,0])
        fft_images2 = tf.signal.fft2d(tf_images[:,:,1])
        fft_images3 = tf.signal.fft2d(tf_images[:,:,2])
        fft_images=[fft_images1,fft_images2,fft_images3]
        fft_images=np.moveaxis(fft_images, 0, -1)

        trunc_images1 = tf.math.multiply(tf.squeeze(fft_images[:,:,0]),frq_dp_msk)
        trunc_images2 = tf.math.multiply(tf.squeeze(fft_images[:,:,1]),frq_dp_msk)
        trunc_images3 = tf.math.multiply(tf.squeeze(fft_images[:,:,2]),frq_dp_msk)
        trunc_images=[trunc_images1,trunc_images2,trunc_images3]
        trunc_images=np.moveaxis(trunc_images, 0, -1)

        sample_images1 = tf.math.real(tf.signal.ifft2d(trunc_images[:,:,0]))
        sample_images2 = tf.math.real(tf.signal.ifft2d(trunc_images[:,:,1]))
        sample_images3 = tf.math.real(tf.signal.ifft2d(trunc_images[:,:,2]))
        sample_images=[sample_images1,sample_images2,sample_images3]
        sample_images=np.moveaxis(sample_images, 0, -1)
      
    if len(tf_images.shape)==4:
        
        tf_images = tf.experimental.numpy.moveaxis(tf_images, 3, 1)
        images_fft = tf.signal.fft2d(tf_images)
        # print(images_fft.shape)
        images_trunc = images_fft * frq_dp_msk
        images_back = tf.math.real(tf.signal.ifft2d(images_trunc)) 
        sample_images = images_back
        # print(type(sample_images))
        # with tf.compat.v1.Session() as sess:
            # sample_images = sess.run(images_back)

    return sample_images