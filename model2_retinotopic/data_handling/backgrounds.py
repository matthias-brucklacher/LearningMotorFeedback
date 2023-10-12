from skimage import color, io
import numpy as np
import torch

def grass_background(height, width, to_torch=False):
    """Creates a grass background of size (height, width).

    Args:
        height (int): Height of background
        width (int): Width of background
        to_torch (bool, optional): Whether to convert to torch tensor. Defaults to False.   
    
    Returns:
        background (np.array/torch.tensor): Background of size (height, width)
    """
    background = color.rgb2gray(io.imread('data/original/grass_texture.jpg'))
    background = background[:height, :width]
    background = background.astype(np.float32)
    if to_torch:
        return torch.tensor(background)
    return background