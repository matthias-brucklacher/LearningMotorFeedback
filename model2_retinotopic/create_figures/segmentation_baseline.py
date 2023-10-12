from matplotlib import pyplot as plt
from model2_retinotopic.data_handling.animals_datasets import animals_training_coupled
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model2_retinotopic.during_run_analysis.segmentation import compute_IoU
from model2_retinotopic.data_handling.animals_datasets import animals_training_coupled
import torch

def segmentation_baseline_performance(dataset, eval_frame, thresh, OF_direction=0, mode='larger'):
    # Use selected OF_direction (o for horizontal, 1 for vertical) and assume object where smaller or larger than thresh
    train_loader = DataLoader(dataset=dataset, batch_size=10, shuffle=False)

    # Iterate through dataset
    baseline_performance = 0
    count = 0
    for data in iter(train_loader):
        count += 1
        data = data[0] # Remove labels

        optic_flow, motor_state, segmentation_ground_truth = data[0][:, eval_frame], data[1][:, eval_frame], data[3][:, eval_frame]
        optic_flow = optic_flow[:, :, :, OF_direction]

        if mode == 'smaller':
            baseline_segmentation = torch.where(optic_flow < thresh, 1, 0)
        elif mode == 'larger':
            baseline_segmentation = torch.where(optic_flow > thresh, 1, 0)
        baseline_performance += compute_IoU(segmentation_ground_truth, baseline_segmentation)
    return (baseline_performance / count)

def segmentation_optimal_baseline(dataset, eval_frame):
    optimal_IoU = 0
    for thresh in [i / 50 for i in range(-100, 100, 10)]:
        baseline_performance = segmentation_baseline_performance(dataset=dataset, eval_frame=eval_frame, thresh=thresh)
        #print(f'thresh {thresh}: IoU {baseline_performance:.2f}')
        if baseline_performance > optimal_IoU:
            optimal_IoU = baseline_performance
    return optimal_IoU


