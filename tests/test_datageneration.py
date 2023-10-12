"""Collection of tests to check whether data creation has changed.

"""
from model2_retinotopic.data_handling.animals_datasets import AnimalsDataset
from model2_retinotopic.data_handling.animals_sequence_generation import create_animals_sequence
import numpy as np

def setup_module():
    """Rerun data creation.
    
    """
    create_animals_sequence('animals')
    create_animals_sequence('animals_removal')
    create_animals_sequence('animals_segmentation')
    create_animals_sequence('homogeneous')
    
def test_animals_removal():
    """Test 'animals_removal' dataset.
    
    """
    dataset_test(
        path_to_data='data/preprocessed',
        dataset_name='animals_removal',
        opticflow_target_shape=(10, 100, 80, 80, 2),
        segmentation_labels_target_checksum=632500,
        visual_seq_target_checksum=800770432
    )

def test_animals():
    """Test 'animals' dataset.
    
    """
    dataset_test(
        path_to_data='data/preprocessed',
        dataset_name='animals',
        opticflow_target_shape=(10, 100, 80, 80, 2),
        segmentation_labels_target_checksum=632500,
        visual_seq_target_checksum=796136320  
    )

def test_animals_segmentation():
    """Test 'animals_segmentation' dataset.
    
    """
    dataset_test(
        path_to_data='data/preprocessed',
        dataset_name='animals_segmentation',
        opticflow_target_shape=(10, 10, 80, 80, 2),
        segmentation_labels_target_checksum=63250,
        visual_seq_target_checksum=80110832  
    )

def test_homogeneous():
    """Test 'homogeneous' dataset. Slightly different than other tests, because no boundary file exists for these empty sequences.
    
    """
    path_to_data = 'data/preprocessed'
    dataset_name = 'homogeneous'
    train_dataset = AnimalsDataset(movement_file=path_to_data + f'/movement_{dataset_name}.npy',
                            opticflow_file=path_to_data + f'/opticflow_{dataset_name}.npy', 
                            visual_input_file=path_to_data + f'/visual_{dataset_name}.npy')

    assert train_dataset.opticflow_data.shape == (10, 100, 80, 80, 2)
    assert int(int(np.sum(train_dataset.visual_seq))) == 813834432

def dataset_test(path_to_data, dataset_name, opticflow_target_shape, segmentation_labels_target_checksum, visual_seq_target_checksum):
    """Test whether this dataset differs from the specified target values.
    
    """
    train_dataset = AnimalsDataset(movement_file=path_to_data + f'/movement_{dataset_name}.npy',
                                opticflow_file=path_to_data + f'/opticflow_{dataset_name}.npy', 
                                visual_input_file=path_to_data + f'/visual_{dataset_name}.npy',
                                segmentation_label=path_to_data + f'/segmentation_labels_{dataset_name}.npy')

    assert train_dataset.opticflow_data.shape == opticflow_target_shape
    assert int(np.sum(train_dataset.segmentation_label)) == segmentation_labels_target_checksum
    assert int(int(np.sum(train_dataset.visual_seq))) == visual_seq_target_checksum