# import aicsimage.io as io
import aicsimageio as io
import os
import pdb
import numpy as np
from tifffile import imread

class TifReader(object):
    def __init__(self, filepath):
        self.tif_np = imread(filepath)
        #  with io.tifReader.TifReader(filepath) as reader:
        #     """Keeping it this way in order to extend it further for multi-channel tifs"""
        #     self.tif_np = reader.tif.asarray()
        #     self.tif_np = np.squeeze(self.tif_np, axis = 0)

    def get_image(self):
        """Returns the image for the specified channel."""
        """Keeping it this way in order to extend it further for multi-channel tifs"""

        return self.tif_np
