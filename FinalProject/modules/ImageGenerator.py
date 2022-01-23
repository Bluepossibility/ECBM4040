import numpy as np
from matplotlib import pyplot as plt

class ImageGenerator(object):

    def __init__(self, x, y):
        """
        Initialize an ImageGenerator instance.
        :param x: A Numpy array of input data. It has shape (num_of_samples, height, width, channels).
        :param y: A Numpy vector of labels. It has shape (num_of_samples, ).
        """
        self.samples_count
        self.x = x
        self.y = y
        self.sample_count, self.height, self.width, self.channel_count = self.x.shape  # N,H,W,C
        self.height_shift = 0  # the amount shift in height
        self.width_shift = 0  # the amount shift in width
        self.angle = 0  # default no angle
        self.flip_horizontal = False   # whether a horizontal flip is needed
        self.flip_vertical = False  # whether a vertical flip is needed
        self.add_noise = False  # whether to add noise

    def next_batch_gen(self, batch_size, shuffle=True):
        """
        A python generator function that yields a batch of data indefinitely.
        :param batch_size: The number of samples to return for each batch.
        :param shuffle: If True, shuffle the entire dataset after every sample has been returned once.
                        If False, the order or data samples stays the same.
        :return: A batch of data with size (batch_size, width, height, channels).
        """
        total_batch_num = self.sample_count // batch_size  # number of batches in total
        batch_num = 0  # type = int
        while True:
            if batch_num < total_batch_num:
                batch_num += 1
                yield (self.x[(batch_num - 1) * batch_size : batch_num * batch_size],
                       self.y[(batch_num - 1) * batch_size : batch_num * batch_size])
            else:
                if shuffle:
                    perm = np.random.permutation(self.samples_count)
                    self.x = self.x[perm]
                    self.y = self.y[perm]
                batch_num = 0

    def show(self):
        """
        Plot the top 16 images (index 0~15) of self.x for visualization.
        """
        X_sample = self.x[:16] # take all before index 16

        # imshow() one channel of images
        r = 4  # dimension of plots matrix 4 * 4
        f, axarr = plt.subplots(r, r, figsize=(8,8))  # plot matrix and figure size
        for i in range(r):
            for j in range(r):
                img = X_sample[r*i+j]
                axarr[i][j].imshow(img, cmap="gray")  # grey images

    def shift(self, height_shift, width_shift):
        """
        Translate self.x by the values given in shift.
        :param height_shift: the number of pixels to shift along height direction. Can be negative.
        :param width_shift: the number of pixels to shift along width direction. Can be negative.
        :return:
        """
        self.height_shift = height_shift
        self.width_shift = width_shift

        # shift the values along the axis, 1 and 2 means height and width
        # Roll array elements along a given axis.
        # Elements that roll beyond the last position are re-introduced at the first.
        self.x = np.roll(self.x, height_shift, axis=1)  # shift in height, axis = 1, horizontal
        self.x = np.roll(self.x, width_shift, axis=2)  # shift in width, axis = 2, vertical

    def flip(self, mode='h'):
        """
        Flip self.x according to the mode specified
        :param mode: 'h' or 'v' or 'hv'. 'h' means horizontal and 'v' means vertical.
        Do the augmentation.
        """
        self.flip_horizontal = 'h' in mode  # h = horizontal, default horizontal
        self.flip_vertical = 'v' in mode  # v = vertical
        if self.flip_horizontal:
            self.x = np.flip(self.x, axis=2)  # flip horizontally is to flip upon axis 2
        if self.flip_vertical:
            self.x = np.flip(self.x, axis=1)  # flip vertically is to flip upon axis 1