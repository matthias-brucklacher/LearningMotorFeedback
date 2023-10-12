import numpy as np
import os
import torch

def make_tensor_plottable(img_width):
    """Returns a function that can be used to detach and convert a tensor to a 2d numpy array of shape (img_width, img_width)

    Args:
        img_width (int): Width of the image

    Returns:
        function: Function that can be used to convert a tensor to a 2d numpy array of shape (img_width, img_width)

    """
    return lambda tensor: tensor.detach().cpu().numpy().reshape((img_width, img_width))

def delete_all_files_in_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)