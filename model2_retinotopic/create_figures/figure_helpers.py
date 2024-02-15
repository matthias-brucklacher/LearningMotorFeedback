import matplotlib.pyplot as plt
from model2_retinotopic.data_handling.animals_datasets import empty_like_animals_coupled, animals_training_coupled, animals_independently_moving
from model2_retinotopic.network.network_hierarchical import HierarchicalNetwork
from model2_retinotopic.during_run_analysis.segmentation import get_boundary
from model2_retinotopic.network.train import train
from model2_retinotopic.helpers import make_tensor_plottable
import numpy as np
import torch
from torch.utils.data import DataLoader

def plot_errorband(ax, x, y, yerr, color=None, label=None, alpha=0.2):
    x = np.array(x)
    y = np.array(y)
    yerr = np.array(yerr)
    ax.plot(x, y, color=color, label=label)
    ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=alpha)

def snapshot(load_path_pretrained, load_path_trained, dataset, eval_frame, show_idcs=[2, 4], mode=None):
    # Take pretrained and trained network, perform inference until a screenshot of the patterns is taken
    BATCH_SIZE = 10
    data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)
    data = next(iter(data_loader))
    data = data[0] # Remove class labels
    optic_flow, movement, visual, segmentation_label = data
    assert optic_flow.shape[1] > eval_frame, f'eval_frame must be in sequence (<{optic_flow.shape[1]}, but was {eval_frame})'
    img_width, movements_dim = visual.shape[-1], movement.shape[-1]
    trained_net = HierarchicalNetwork(img_width=img_width, movement_dim=movements_dim)
    pretrained_net = HierarchicalNetwork(img_width=img_width, movement_dim=movements_dim)
    
    pretrained_net.load_state_dict(torch.load(load_path_pretrained + '_run-0'))
    trained_net.load_state_dict(torch.load(load_path_trained + '_run-0'))
    trained_net.infer_all()
    pretrained_net.infer_stream_1()
    for net in [pretrained_net, trained_net]:
        net.reset_activity(batch_size=BATCH_SIZE)

    for frame_it in range(eval_frame + 1):
        nPE0_pretrained, _, _, _ = pretrained_net(optic_flow[:, frame_it], movement[:, frame_it])
        nPE0_trained, _, _, _ = trained_net(optic_flow[:, frame_it], movement[:, frame_it])
    direction_idx = 1

    fig, axs = plt.subplots(len(show_idcs), 4, figsize=(8, 2.4))
    segmented_areas = []
    ground_truths = []
    for i, show_idx in enumerate(show_idcs):
        min_val = torch.min(torch.stack((visual[show_idx, frame_it], nPE0_pretrained[show_idx, direction_idx], nPE0_trained[show_idx, direction_idx])))
        max_val = torch.max(torch.stack((visual[show_idx, frame_it], nPE0_pretrained[show_idx, direction_idx], nPE0_trained[show_idx, direction_idx])))
        image_for_cbar = axs[i, 0].imshow(visual[show_idx, frame_it], vmin=min_val, vmax=max_val)
        axs[i, 1].imshow(net.PE_actfun(nPE0_pretrained[show_idx, direction_idx]), vmin=min_val, vmax=max_val)
        axs[i, 2].imshow(net.PE_actfun(nPE0_trained[show_idx, direction_idx]), vmin=min_val, vmax=max_val)
        if torch.sum(movement[show_idx, frame_it]) != 0:
            predicted_segmentation_area = trained_net.segment()[show_idx]
            segmented_areas.append(predicted_segmentation_area)
            axs[i, 3].imshow(predicted_segmentation_area, vmin=min_val, vmax=max_val)
            object_boundary_groundtruth = get_boundary(segmentation_label[show_idx, frame_it])
            ground_truths.append(object_boundary_groundtruth)
            axs[i, 3].scatter(np.nonzero(object_boundary_groundtruth)[0],
                                np.nonzero(object_boundary_groundtruth)[1],
                                                    marker='.',
                                                    linewidths=0.001,
                                                    color='red')
        for ax in axs.flat:
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            ax.set_xticks([])
            ax.set_yticks([])

    axs[0, 0].set_ylabel(f'Sample {show_idcs[0]}/10')
    axs[1, 0].set_ylabel(f'Sample {show_idcs[1]}/10')
    axs[0, 0].set_xlabel('Retina\nVisual input')
    axs[0, 1].set_xlabel('V1-nPE\n(mHVA off)')
    axs[0, 2].set_xlabel('V1-nPE\n(mHVA on)')
    axs[0, 3].set_xlabel('Segmentation\n(mHVA on)')

    for ax in [axs[0, 0], axs[0, 1], axs[0, 2], axs[0, 3]]:
        ax.xaxis.set_label_position('top')
    
    plt.subplots_adjust(wspace=-0.5)
    fig.subplots_adjust(right=0.8)

    cbar_ax = fig.add_axes([0.75, 0.35, 0.02, 0.3])
    cbar = fig.colorbar(mappable=image_for_cbar, cax=cbar_ax, shrink=0.1)
    cbar.set_label('Neural activity')

    if mode != 'joint_training':
        plt.savefig('results/figures/fig6A_snapshot.png', dpi=800, bbox_inches='tight')
    return segmented_areas, ground_truths
