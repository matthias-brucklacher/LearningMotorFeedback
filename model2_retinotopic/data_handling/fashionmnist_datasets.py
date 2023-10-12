
import torchvision.transforms as T
import torchvision.datasets
from model2_retinotopic.data_handling.fashionmnist_preprocessing_transform import Fashionmnist_to_sequence

def empty_like_fashionmnist_train():
    """Returns dataset with empty sequences of fashionmnist images. Used for pretraining.

    """
    return torchvision.datasets.FashionMNIST("./data/original", download=True, train=True, transform= 
                                             T.Compose([T.ToTensor(), Fashionmnist_to_sequence(empty=True)]))

def fashionmnist_train():
    """Returns dataset with sequences of fashionmnist images.

    Calling next(enumerate()) on the returned dataset will return a tuple of the form (data, label), 
    where data is a list containing optic_flow, movement, visual_sequence, segmentation_labels.

    """
    return torchvision.datasets.FashionMNIST("./data/original", download=True, train=True, transform= 
                                             T.Compose([T.ToTensor(), Fashionmnist_to_sequence(empty=False)]))

def fashionmnist_test():
    """Returns dataset with sequences of fashionmnist images.

    Calling next(enumerate()) on the returned dataset will return a tuple of the form (data, label), 
    where data is a list containing optic_flow, movement, visual_sequence, segmentation_labels.

    """
    return torchvision.datasets.FashionMNIST("./data/original", download=True, train=False, transform= 
                                             T.Compose([T.ToTensor(), Fashionmnist_to_sequence(empty=False)]))