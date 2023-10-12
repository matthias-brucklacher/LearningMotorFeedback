from model2_retinotopic.data_handling.movement import movement_alternating
from model2_retinotopic.data_handling.multimove_sequence_generation import create_sequence_multimove
import torch
from torch.utils.data import Dataset

class MultimoveDataset(Dataset):

    def __init__(self):
        """ Build torch dataset for the multimove dataset.

        """
        multimove_data = create_sequence_multimove(img_width=40)
        self.opticflow, self.movement, self.visual, self.segmentation_label = [torch.from_numpy(array) for array in multimove_data]

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
        labels = 0 # dummy label to be compatible with fashionmnist dataset
        return data, labels
    
def empty_multimove():
    # Returns dataset with empty sequences (of the same dimension as the sequences with animals). Used for pretraining.
    return MultimoveDataset()