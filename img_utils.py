import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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

