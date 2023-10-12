"""
Build torch dataset and dataloader from the generated optic flow and motor state files
Based on: https://towardsdatascience.com/dataloader-for-sequential-data-using-pytorch-deep-learning-framework-part-2-ed3ad5f6ad82
"""

from model2_retinotopic.data_handling.animals_sequence_generation import create_animals_sequence
import torch
from torch.utils.data import Dataset

class AnimalsDataset(Dataset):
    def __init__(self, empty=False, removal_paradigm=False, coupled=True, relative_speed=0, interleaved_with_empty=False):
        """Build torch dataset for the animals dataset. 

        Args:
            empty (bool, optional): If True, returns dataset with empty sequences of animals images. Used for pretraining. Defaults to False.
            removal_paradigm (bool, optional): If True, returns dataset with sequences of animals images with removal paradigm. Defaults to False.
            coupled (bool, optional): If True, returns dataset with sequences of animals images with coupled paradigm. Defaults to True.
        
        Example:
            >>> animals_empty = AnimalsDataset(empty=True)
            >>> animals_removal = AnimalsDataset(removal_paradigm=True)
            >>> animals_coupled = AnimalsDataset()
            >>> data, labels = animals_empty()

        """
        self.interleaved_with_empty = interleaved_with_empty
        if interleaved_with_empty:
    
            assert empty == False, "Cannot have empty and interleaved_with_empty at the same time."
            empty_data = create_animals_sequence(empty=True, removal_paradigm=removal_paradigm, coupled=coupled, relative_speed=relative_speed)
            self.empty_opticflow, self.empty_movement, self.empty_visual, self.empty_segmentation_label = [torch.from_numpy(array) for array in empty_data]

        animals_data = create_animals_sequence(empty=empty, removal_paradigm=removal_paradigm, coupled=coupled, relative_speed=relative_speed)
        self.opticflow, self.movement, self.visual, self.segmentation_label = [torch.from_numpy(array) for array in animals_data]

    def __len__(self):
        return len(self.opticflow)
    
    def __getitem__(self, idx):
        """Returns a tuple of the form (data, label), where data is a list containing the sequences optic_flow, movement, visual_sequence, segmentation_labels.

        Args:
            idx (int): sample index in batch (of the sequence to be returned)

        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = [self.opticflow[idx], self.movement, self.visual[idx], self.segmentation_label[idx]]
        if self.interleaved_with_empty:
            if idx not in [3]:
                data = [self.empty_opticflow[idx], self.empty_movement, self.empty_visual[idx], self.empty_segmentation_label[idx]]
            else:
                data = [self.opticflow[idx], self.movement, self.visual[idx], self.segmentation_label[idx]]
        labels = 0 # Also return a dummy label to be have the same format as the modified fashionmnist dataset
        return data, labels

def empty_like_animals_coupled():
    # Returns dataset with empty sequences (of the same dimension as the sequences with animals). Used for pretraining.
    return AnimalsDataset(empty=True)

def empty_like_animals_noncoupled():
    # Returns dataset with empty sequences (of the same dimension as the sequences with animals). Used as control for pretraining.
    return AnimalsDataset(empty=True, coupled=False)

def animals_removal():
    # Returns dataset with sequences of animals images with removal paradigm in which the object disappears suddenly.
    return AnimalsDataset(removal_paradigm=True)

def animals_training_coupled():
    # Returns dataset with sequences of animals images with coupled paradigm.
    return AnimalsDataset()

def animals_training_noncoupled():
    # Returns dataset with sequences of animals images with non-coupled paradigm.
    return AnimalsDataset(coupled=False)

def animals_independently_moving(relative_speed):
    # Returns dataset with sequences of animals images with non-gaze following paradigm.
    return AnimalsDataset(relative_speed=relative_speed)

def animals_interleaved_with_empty():
    # Returns dataset with sequences of animals images interleaved with empty sequences.
    return AnimalsDataset(interleaved_with_empty=True)
