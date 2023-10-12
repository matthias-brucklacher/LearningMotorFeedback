from typing import Any
import cv2
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from skimage import color, io
import torchvision.transforms as T
from model2_retinotopic.data_handling.backgrounds import grass_background
from model2_retinotopic.data_handling.movement import movement_alternating, movement_constant


def png_to_tensor(path):
    """Convert png image to tensor.

    Args:
        path (str): Path to image
    
    Returns:
        tensor (torch.tensor): Tensor of image

    """
    img = torch.tensor(io.imread(path, as_gray=True))
    return img

def frame_touched(img):
    # Check whether non-zero pixels are touching the edge of the image
    if np.sum(img[0, :]) > 0 or np.sum(img[-1, :]) > 0 or np.sum(img[:, 0]) > 0 or np.sum(img[:, -1]) > 0:
        return True

def compute_flow(visual_input):
    num_seqs, sequence_length, img_height, img_width = visual_input.shape
    flow = np.zeros((num_seqs, sequence_length, img_height, img_width, 2), dtype=np.float32)
    for i in range(num_seqs):
        for j in range(1, sequence_length):
            flow[i, j] = cv2.calcOpticalFlowFarneback(255 * visual_input[i, j - 1], 
                                                     255 * visual_input[i, j], 
                                                     flow=None, pyr_scale=0.5, 
                                                     levels=3, 
                                                     winsize=10, 
                                                     iterations=3, 
                                                     poly_n=5, 
                                                     poly_sigma=1.2, 
                                                     flags=0)
        flow[i, 0] = flow[i, 1] # Assume padding
    return flow

def resize(img):
    """Takes 2D tensor and resizes the larger dimension to 50, keeping the aspect ratio.

    Args:
        img (torch.tensor): 2D tensor
    
    Returns:
        img (torch.tensor): Resized 2D tensor

    """
    img = img.unsqueeze(0) # Resize expects >=3D tensor
    if img.shape[1] > img.shape[2]:
        img = T.Resize(size=(50, int(50 * img.shape[2] / img.shape[1])), antialias=None)(img)
    else:
        img = T.Resize(size=(int(50 * img.shape[1] / img.shape[2]), 50), antialias=None)(img)
    return img[0]

def create_moving_background(img_width, sequence_length, movement):
    """Create a moving background pattern.
    
    Args:
        img_width (int): Width of image
        sequence_length (int): Length of sequence
        movement (np.array): Array of shape (sequence_length, 1) with 1s indicating movement

    Returns:
        visual_input (torch.tensor): Tensor of shape (1, sequence_length, img_width, img_width) with moving background pattern

    """
    assert movement.shape == (sequence_length, 1), 'Movement must be a 1D array of length sequence_length'
    # Create background patterns
    background = grass_background(height=img_width, width=img_width + sequence_length - 1)
    background = np.repeat(background[np.newaxis, :, :], 10, axis=0)

    # Create frames by shifting background pattern
    visual_input = np.zeros((1, sequence_length, img_width, img_width), dtype=np.float32)
    lateral_shift = 0
    for j in range(sequence_length):
        if movement[j, 0] == 1:
            lateral_shift += 1
        visual_input[0, j] = np.roll(background[0], -lateral_shift, axis=1)[:, 0:img_width]

    return torch.tensor(visual_input)

def place_object_in_sequence(sequence, img, removal_paradigm=False, relative_speed=0):
    """Takes an empty sequence and places an image in it.

    Args:
        sequence (np.array): Array of shape (sequence_length, img_width, img_width)
        img (np.array): Array of shape (img_width, img_width)
        removal_paradigm (bool, optional): Whether to remove the object after a few frames. Defaults to False. 

    Returns:
        sequence (np.array): Array of shape (sequence_length, img_width, img_width) with object placed

    """

    img_width = sequence.shape[2]

    sequence[:, int((img_width - img.shape[0]) / 2):int((img_width - img.shape[0]) / 2) + img.shape[0],
                        int((img_width - img.shape[1]) / 2):int((img_width - img.shape[1]) / 2) + img.shape[1]] = img
    if relative_speed != 0:
        if True: #img.shape[0] > img.shape[1]: # if image is taller than wide, move horizontally
            movement_axis = 1
        else:
            movement_axis = 0
        forward_backward_switch = 1 # -1
        shift = relative_speed
        sequence[1] = np.roll(sequence[1], forward_backward_switch * shift, axis=movement_axis) # Second frame is shifted by 1
        for frame_it in range(1, sequence.shape[0]):
            sequence[frame_it] = np.roll(sequence[frame_it - 1], forward_backward_switch * shift, axis=movement_axis)
            # If the image is at the edge of the frame, switch direction
            if frame_touched(sequence[frame_it]):
                pass
                #forward_backward_switch *= -1

    if removal_paradigm:
        removal_period = [5, 6, 7, 8, 9]
        sequence[removal_period] = 0
    return sequence

def create_animals_sequence(removal_paradigm=False, empty=False, coupled=True, relative_speed=0):
    """Create animal sequences.

    Args:
        removal_paradigm (bool, optional): Whether to remove the animal after a few frames. Defaults to False.
        empty (bool, optional): Whether to create empty sequences. Defaults to False.
        coupled (bool, optional): Whether to create coupled sequences. Defaults to True.

    Returns:
        sequences (np.array): Array of shape (n_samples, sequence_length, img_width, img_width)
        segmented_object (np.array): Array of shape (n_samples, sequence_length, img_width, img_width)
        movement (np.array): Array of shape (sequence_length, 1)
        flow (np.array): Array of shape (n_samples, sequence_length, img_width, img_width, 2)

    """
    # make list of all animal files
    n_samples = 10
    img_width = 80
    sequence_length = 60
    visual = np.zeros((n_samples, sequence_length, img_width, img_width), dtype=np.float32)
    segmentation_mask = np.zeros_like(visual)
    movement = movement_alternating()

    # Whether the movement signal controlling the visual feedback is the same depends on paradigm
    if coupled:
        movement_for_vr = movement_alternating() 
    else:
        movement_for_vr = np.random.binomial(1, 0.5, size=(sequence_length, 1))

    background = create_moving_background(img_width, sequence_length, movement_for_vr)
    for i, file in enumerate(os.listdir('data/original/animals')):
        if file.endswith('.png'):
            img = resize(png_to_tensor('data/original/animals/' + file))

            img = np.where(img < 0.99, img, 0.) # Remove white background
            single_segmentation_mask = np.where(img != 0, 1, 0)
            
            if empty:
                pass
            else:
                visual[i] = place_object_in_sequence(visual[i], img, removal_paradigm=removal_paradigm, relative_speed=relative_speed)
                segmentation_mask[i] = place_object_in_sequence(segmentation_mask[i], single_segmentation_mask, relative_speed=relative_speed)
            
            # Where the image is not 0, place background
            visual[i] = np.where(visual[i] > 0., visual[i], background)

    flow = compute_flow(visual)
    return flow, movement, visual, segmentation_mask



            
            

