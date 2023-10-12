import cv2
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from model2_retinotopic.data_handling.backgrounds import grass_background


class Fashionmnist_to_sequence():

    def __init__(self, empty=False):
        """Transformation of a single sample into a sequence in front of a moving background.

        The transformation is applied during data loading: 
        The input sample is a single image from the fashionmnist dataset.
        The output is a sequence of images, where the object is moving in front of a moving background.

        Args:
            empty (bool, optional): If True, the sequence will be empty. Defaults to False.

        Example:
            >>> from model2_retinotopic.data_handling.fashionmnist_preprocessing_transform import Fashionmnist_to_sequence
            >>> from torch.utils.data import DataLoader
            >>> dataset = FashionMNIST(root='data', train=True, download=True, transform=Fashionmnist_to_sequence(emmpty=False))
            >>> dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

        """
        self.empty = empty
        self.img_width = 40
        self.sequence_length = 50
        self.background = grass_background(height=self.img_width, width=self.img_width + self.sequence_length)
        self.background = torch.from_numpy(self.background)
        assert self.background.shape[0] >= self.img_width, 'Background not high enough'
        assert self.background.shape[1] >= self.img_width + self.sequence_length - 1, 'Background not wide enough'
    
    def __call__(self, sample):
        """Transforms a single sample into a sequence in front of a moving background.
        Args:
            sample (torch.tensor): Single image from fashionmnist dataset
        
        Returns:
            optical_flow (torch.tensor): Sequence of optical flow vectors
            visual_sequence (torch.tensor): Sequence of images
            segmentation_labels (torch.tensor): Sequence of segmentation labels
            movement (np.array): Sequence of movement vectors

        """
        visual_sequence = torch.zeros((self.sequence_length, self.img_width, self.img_width), dtype=torch.float32)
        mov_dim = 1
        movement = np.zeros((self.sequence_length, mov_dim), dtype=np.float32)
        blocksize = 10
        for i in range(self.sequence_length):
            if math.floor(i / blocksize) % 2 == 0:
                movement[i] = 1

        # Roll background 
        lateral_shift = 0
        for frame_it in range(self.sequence_length):
            visual_sequence[frame_it] = torch.roll(self.background, -lateral_shift, dims=1)[:, :self.img_width]
            if movement[frame_it, 0] == 1:
                lateral_shift += 1

        # Place object in center
        segmentation_labels = torch.zeros_like(visual_sequence)
 
        if self.empty == False:
            segmentation_labels[:, 6:34, 6:34] += torch.where(sample > 0.02, 1.0, 0.0)
            visual_sequence[:, 6:34, 6:34] = torch.where(segmentation_labels[:, 6:34, 6:34] == 1, sample, visual_sequence[:, 6:34, 6:34])


        # Compute optic flow
        optic_flow = np.zeros((self.sequence_length, self.img_width, self.img_width, 2), dtype=np.float32)
        visual_sequence_np = visual_sequence.cpu().numpy()
        for j in range(1, self.sequence_length):
            optic_flow[j] = cv2.calcOpticalFlowFarneback(255*visual_sequence_np[j - 1], 
                                                     255*visual_sequence_np[j], 
                                                     flow=None, 
                                                     pyr_scale=0.5, 
                                                     levels=3, 
                                                     winsize=6, 
                                                     iterations=3, 
                                                     poly_n=5, 
                                                     poly_sigma=1.1, 
                                                     flags=0)
            optic_flow[0] = optic_flow[1]
        optic_flow = torch.from_numpy(optic_flow)
        movement = torch.from_numpy(movement)
        return optic_flow, movement, visual_sequence, segmentation_labels
    
