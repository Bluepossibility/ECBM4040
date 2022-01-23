"""Create scaled and shifted images for exploration."""
from PIL import Image
import os
import numpy as np
# The path we store all the pictures
__DEFAULT_PATH = '/home/ecbm4040/Spectral_Representation_for_CNN/Images'


def open_image(filename, path=__DEFAULT_PATH):
    """Open an image file with Pillow."""
    # To read or write files see open(), and for accessing the filesystem see the os module.
    # The path parameters can be passed as either strings, or bytes.
    if filename is None:
        raise ValueError('Filename is required.')
    full_path = os.path.join(path, filename)
    #The Image module provides a class with the same name which is used to represent a PIL image.
    #The module also provides a number of factory functions, including functions to load images from files, and to create new images.
    im = Image.open(full_path).convert('RGBA')
    return im


def save_derived_image(im, filename=None, path=__DEFAULT_PATH):
    """Save a pillow image as a PNG."""
    if filename is None:
        # define the filename
        filename = 'Derived/{0:08x}.png'.format(np.random.randint(2 ** 31))
    full_path = os.path.join(path, filename)
    # OS module in Python provides functions for interacting with the operating system.
    # OS comes under Python’s standard utility modules. This module provides a portable way of using operating system dependent functionality.
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    im.save(full_path, 'PNG')


def downscale_image(orig_image, max_width, max_height):
    """Rescale an image to a smaller image."""
    orig_width = orig_image.width
    orig_height = orig_image.height

    # Compute how much to multiply the existing dimensions by
    width_multo = max_width / orig_width
    height_multo = max_height / orig_height
    multo = min(height_multo, width_multo)

    # Create the new image
    new_width = int(orig_width * multo)
    new_height = int(orig_height * multo)
    # Lanczos filtering and Lanczos resampling are two applications of a mathematical formula.
    # It can be used as a low-pass filter or used to smoothly interpolate the value of a digital signal between its samples.
    # In the latter case it maps each sample of the given signal to a translated and scaled copy of the Lanczos kernel, which is a sinc function windowed by the central lobe of a second, longer, sinc function. The sum of these translated and scaled kernels is then evaluated at the desired points.
    new_image = orig_image.resize((new_width, new_height),resample=Image.LANCZOS)

    return new_image


def add_to_background(
    foreground_image,
    destination_left,
    destination_top,
    destination_max_width,
    destination_max_height,
    background_image=None,
    background_width=128,
    background_height=128,
):
    """Add an image to a set image on the jupyter notebook.
    If background_image == None, the function will create a solid grey background image of dimensions
    with the form of (background_width, background_height)and paste the image onto that.
    """
    if background_image is None:
        # PIL.Image.new() method creates a new image with the given mode and size.
        # Size is given as a (width, height)-tuple, in pixels.
        # The color is given as a single value for single-band images, and a tuple for multi-band images (with one value for each band).
        new_background_image = Image.new('RGBA',(background_width, background_height),'#7f7f7f')
    else:
        # Copy part of an image
        new_background_image = background_image.copy()

    rescaled_foreground_image = downscale_image(foreground_image,destination_max_width,destination_max_height,)
    new_background_image.paste(rescaled_foreground_image,box=(destination_left, destination_top),mask=rescaled_foreground_image)

    return new_background_image


def make_random_size(destination_width=128, destination_height=128):

    # define the scale and new destination of an image
    scale = np.random.randint(16,1 + min(destination_width, destination_height))
    left = np.random.randint(0, 1 + destination_width - scale)
    top = np.random.randint(0, 1 + destination_height - scale)
    width = scale
    height = scale

    return left, top, width, height
