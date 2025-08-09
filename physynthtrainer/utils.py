
from PIL import Image
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def paint_arr_to_jpg(arr, filename='test.jpg', do_norm=True):
    """
    Saves a 2D numpy array as a jpg image.

    Parameters:
    -----------
    arr : np.ndarray
        The 2D array to save.
    filename : str
        The name of the file to save.
    do_norm : bool
        Whether to normalize the array before saving.
    """
    norm = mcolors.Normalize(vmin=arr.min(), vmax=arr.max())
    cmap = plt.get_cmap('CMRmap') 
    img = cmap(norm(arr.T))
    imgsave = (img * 255).astype(np.uint8)[:,:,0:3]
    im = Image.fromarray(imgsave)
    im.save(filename)