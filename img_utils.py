import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import openslide

def count_non_zero_pixels(image_file):
    """Calculate the number of non-zero pixels in an image.

    Parameter:
        image_file (str): path to image file (compatible with plt.imread)

    Returns:
        int: the number of non-zero pixels in the image
    """
    im = plt.imread(image_file)
    nz_pixels = np.count_nonzero(im)
    return nz_pixels

def extract_tile(image_file, minx, miny, width, height):
    """Extract a tile from an image. 

    Parameter:
        image_file (str): path to image file (compatible with openslide.open_slide)
        minx, miny: top left pixel coordinate in the level 0 reference frame
        width, height: width and height in pixels in level 0 reference frame

    Returns:
        image: tile 
    """
    slide = openslide.open_slide(image_file)
    im = slide.read_region((minx, miny), 0, (width, height))
    return im
