import matplotlib.pyplot as plt
from model2_retinotopic.data_handling.movement import movement_alternating
import numpy as np
import torch

def expanding_flow(img_width=40, invert=False):
    """ Creates a flow field that expands from the center of the image.

    Args:
        img_width (int): Width of the image
        invert (bool): If True, the flow field will contract towards the center of the image.

    Returns:
        optic_flow (np.array): Flow field of shape (img_width, img_width, 2)

    """
    optic_flow = np.zeros((img_width, img_width, 2), dtype=np.float32)
    center = int(img_width/2)
    linear_field = np.arange(-center, img_width-center, dtype=np.float32) # (W,)
    optic_flow[:, :, 0] = np.stack([linear_field for i in range(img_width)], axis=0) # (W, H)
    optic_flow[:, :, 1] = np.stack([linear_field for i in range(img_width)], axis=1) # (W, H)
    optic_flow = optic_flow / np.max(optic_flow)
    if invert:
        optic_flow = -optic_flow
    return optic_flow 

def linear_flow(direction, img_width=40):
    """ Creates a flow field that is linear in one direction.
    
    Args:
        direction (str): 'x', 'y', or '-xy'
        img_width (int): Width of the image
    
    Returns:
        optic_flow (np.array): Flow field of shape (img_width, img_width, 2)
        
    """
    optic_flow = np.zeros((img_width, img_width, 2), dtype=np.float32)
    if direction == 'x':
        optic_flow[:, :, 0] = np.ones((img_width, img_width), dtype=np.float32) 
    elif direction == 'y':
        optic_flow[:, :, 1] = np.ones((img_width, img_width), dtype=np.float32)
    elif direction == '-xy':
        optic_flow[:, :, 0] = -np.ones((img_width, img_width), dtype=np.float32) 
        optic_flow[:, :, 1] = -np.ones((img_width, img_width), dtype=np.float32)
        optic_flow = optic_flow / np.sqrt(2)
    return optic_flow

def gradient_flow(img_width=40, direction='x'):
    """ Creates a flow field that is linearly increasing in one direction.
    
    Args:
        direction (str): 'x' or 'y'
        img_width (int): Width of the image
    
    Returns:
        optic_flow (np.array): Flow field of shape (img_width, img_width, 2)
    
    """
    optic_flow = np.zeros((img_width, img_width, 2), dtype=np.float32)
    linear_field = np.arange(0, img_width, dtype=np.float32) # (W,)
    if direction == 'x':
        optic_flow[:, :, 0] = np.stack([linear_field for i in range(img_width)], axis=0)
    elif direction == 'y':
        optic_flow[:, :, 1] = np.stack([linear_field for i in range(img_width)], axis=1)
    optic_flow = optic_flow / np.max(optic_flow)
    return optic_flow

def create_sequence_multimove(img_width=40):
    """Create sequences with multiple movement dimensions and different optic flow patterns. 
    
    The visual input is left blank for simplicity and the optic flow directly created.

    Args:
        img_width (int): Width of the image
    
    Returns:
        optic_flow (np.array): Flow field of shape (n_sequences, sequence_length, img_width, img_width, 2)
        movement (np.array): Movement of shape (sequence_length, mov_dim)
        visual (np.array): Visual input of shape (n_sequences, sequence_length, img_width, img_width)
        segmentation_label (np.array): Segmentation label of shape (n_sequences, sequence_length, img_width, img_width)

    """
    mov_dim = 6
    n_sequences = 1 # One sequence per movement dimension
    movement = movement_alternating(mov_dim=mov_dim)
    sequence_length = movement.shape[0]

    # visual and segmentation_label: all zeros
    visual = np.zeros((n_sequences, sequence_length, img_width, img_width), dtype=np.float32)
    segmentation_label = np.zeros((n_sequences, sequence_length, img_width, img_width), dtype=np.float32)

    # Optic flow
    optic_flow = np.zeros((1, sequence_length, img_width, img_width, 2), dtype=np.float32)
    optic_flow[0, :10] = linear_flow(img_width=img_width, direction='x')
    optic_flow[0, 10:20] = linear_flow(img_width=img_width, direction='y') 
    optic_flow[0, 20:30] = linear_flow(img_width=img_width, direction='-xy') 
    optic_flow[0, 30:40] = gradient_flow(img_width=img_width, direction='x')
    optic_flow[0, 40:50] = expanding_flow(img_width=img_width) 
    optic_flow[0, 50:60] = expanding_flow(img_width=img_width, invert=True) 

    return optic_flow, movement, visual, segmentation_label

if __name__ == '__main__':
    optic_flow, movement, visual, segmentation_label = create_sequence_multimove(img_width=40)

    # Plot all optic flow patterns
    for i in range(optic_flow.shape[0]):
        plt.subplot(2, 3, i+1)
        plt.quiver(optic_flow[i, 0, :, :, 0][::2, ::2], optic_flow[i, 0, :, :, 1][::2, ::2], scale=30) # Plotting only every second arrow looks cleaner
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xticks([])  
        plt.yticks([])
        plt.box(False) 
        plt.title('ABCDEFGH'[i], fontsize=16)
    
    plt.show()
