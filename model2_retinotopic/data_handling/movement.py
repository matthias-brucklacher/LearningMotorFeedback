import numpy as np
import math
import warnings


def movement_constant(mov_dim=1):
    # Create a movement signal consisting of constant one-hot states.

    sequence_length = 60
    movement = np.zeros((sequence_length, mov_dim), dtype=np.float32)
    movement[:, 0] = 1

def movement_alternating(mov_dim=1):
    """Create a movement signal consisting of blocks of one-hot states.

    Args:
        mov_dim (int, optional): Number of movement dimensions. Defaults to 1.

    Returns:
        movement (np.array): Movement signal of size (sequence_length, mov_dim)

    Examples:
        >>> movement_alternating(mov_dim=1)
        array([[1.],
                [1.],
                ...,
                [0.],
                [0.]])
        >>> movement_alternating(mov_dim=2)
        array([[0., 1.],
                [0., 1.],
                ...,
                [0., 0.],
                [0., 0.]])

    """
    sequence_length = 60
    blocksize = 10
    n_blocks = sequence_length // blocksize
    if n_blocks < mov_dim:
        warnings.warn(f'The number of movement dimensions ({mov_dim}) is larger than the number of blocks ({n_blocks}). Some dimensions will not be used.')  
    movement = np.zeros((sequence_length, mov_dim), dtype=np.float32)
    for i in range(sequence_length):
        movement_id = math.ceil((i+1) / blocksize) % (mov_dim + 1)
        if movement_id == 0:
            movement[i] = np.zeros((mov_dim))
        else:
            movement[i, movement_id - 1] = 1 # Place 1 in the correct dimension


    return movement


