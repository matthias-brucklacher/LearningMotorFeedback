import matplotlib.pyplot as plt
from model2_retinotopic.data_handling.animals_datasets import animals_training_coupled
import numpy as np
from skimage import segmentation
import torch
from torch.utils.data import DataLoader
    
def segmentation_performance(net, test_dataset, eval_frame, run_id):
    """Evaluate segmentation performance of a network on a fixed test set.

    Args:
        net (VisualMotoricNetwork): Network to be evaluated.
        eval_frame (int): After which frame of the sequence to evaluate. At this frame, there needs to be relative speed !=0 between the object and the background.
    
    Returns:
        IoU (float): Intersection over union between [0, 1]. The higher, the better.

    """
    if net.inference_mode == 'stream_1':
        print('Network cannot be in motor-to-visual mode for segmentation. Skipping.')
        return 0
    
    # Set up data loader
    testloader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  

    # Initialize some variables
    IoU_sum = 0
    IoU_summands = 0

    # Iterate through data
    for i, data in enumerate(testloader):
        contains_class_labels = (len(data) == 2)
        if contains_class_labels:
            data = data[0] # Don't take class labels (these are data[1])
        opticflow_input, motor_state, visual_input, segmentation_labels = data[0].to(device), data[1].to(device), data[2], data[3]
        sequence_length = opticflow_input.shape[1]
        batch_size = opticflow_input.shape[0]
        net.reset_activity(batch_size)
        for frame_it in range(eval_frame + 1):
            nPE0, pPE0, _, _  = net(opticflow_input[:, frame_it], motor_state[:, frame_it])
    
        # Get the prediction by the model
        predicted_area = net.segment()
        
        # Get the ground truth segmentation
        img_width = opticflow_input.shape[-2]
        true_area = np.zeros((batch_size, img_width, img_width))
        for i in range(batch_size):
            true_area[i] = segmentation_labels[i, eval_frame]
        if batch_size >= 10:
            n_plots = 10
        else:
            n_plots = batch_size
        fig, axs = plt.subplots(n_plots, 2)
        for i in range(n_plots):
            axs[i, 0].imshow(predicted_area[i])
            axs[i, 1].imshow(true_area[i])
        axs[0, 0].set_title('Predicted')
        axs[0, 1].set_title('Ground truth')
        for ax in axs.flat:
            # remove the x and y ticks
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        
        plt.savefig(f'results/debugging/segmentation_{run_id}.png') 
        plt.close()
        # Compute intersection over union for this batch
        IoU_this_batch = compute_IoU(true_area, predicted_area)
        IoU_sum += IoU_this_batch
        IoU_summands += 1
    IoU = IoU_sum / IoU_summands # Averaged across all batches
    return IoU

def compute_IoU(true_area, predicted_area):
    """Compute intersection over union for given true and predicted area.

    Args:
        true_area (numpy.ndarray): Ground truth.
        predicted_area (numpy.ndarray): Prediction (e.g. by a model).

    Returns:
        IoU (float): Intersection over union between [0, 1]. The higher, the better.
        
    """
    assert true_area.shape == predicted_area.shape

    # Make sure both are binary
    true_area = np.where(true_area > 0, 1, 0)
    predicted_area = np.where(predicted_area > 0, 1, 0)

    # Find intersection (where both are 1) and union (where at least one is 1)
    intersection = np.sum(np.where(predicted_area + true_area > 1, 1, 0))
    union = np.sum(np.where(predicted_area+true_area > 0, 1, 0))
    
    IoU = intersection / union
    return IoU

def get_boundary(segmented_object):
    """Return the boundary between an object and the background that is zero.

    Args:
        segmented_object (torch.tensor): Shape is (height, width).

    Returns:
        boundary (numpy.ndarray): Shape is (width, height).

    """
    segmented_object = segmented_object.cpu().numpy()
    segmented_object = np.swapaxes(segmented_object, 0, 1) # Between torch and numpy, ordering of x and y is inverted.
    boundary = np.zeros_like(segmented_object)
    boundary = segmentation.find_boundaries(segmented_object.astype(np.uint8), mode='outer')
    return boundary

