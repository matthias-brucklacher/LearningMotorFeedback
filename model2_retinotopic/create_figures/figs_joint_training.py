import matplotlib.pyplot as plt
from model2_retinotopic.network.training_configurations import train_animals_jointly, train_animals_separately
from model2_retinotopic.network.network_paths import network_paths
from model2_retinotopic.data_handling.animals_datasets import animals_interleaved_with_empty, animals_training_coupled
from model2_retinotopic.create_figures.figure_helpers import snapshot
from model2_retinotopic.during_run_analysis.segmentation import segmentation_performance
from model2_retinotopic.network.network_hierarchical import HierarchicalNetwork
import numpy as np
import torch

def figs_joint_training():
    plt.style.use('model2_retinotopic.create_figures.plotting_style')

    # Train networks
    train_animals_jointly(n_runs=1)
    train_animals_separately()

    # Plot segmentation snapshot
    eval_dataset = animals_training_coupled()
    show_idcs = [3, 4, 6]
    segmented_areas_joint_training, ground_truths = snapshot(load_path_trained=network_paths['trained_animals_jointly'],
                                            load_path_pretrained=network_paths['trained_animals_jointly'],
                                            dataset=eval_dataset,
                                            eval_frame=28,
                                            show_idcs=show_idcs,
                                            mode='joint_training')
    segmented_areas_separate_training, ground_truths = snapshot(load_path_trained=network_paths['trained_animals'],
                                            load_path_pretrained=network_paths['pretrained_animals'],
                                            dataset=eval_dataset,
                                            eval_frame=28,
                                            show_idcs=show_idcs,
                                            mode='joint_training')

    fig, axs = plt.subplots(2, len(show_idcs))
    for col_it, show_idx in enumerate(show_idcs):
        axs[0, col_it].imshow(segmented_areas_joint_training[col_it])
        axs[1, col_it].imshow(segmented_areas_separate_training[col_it])
        for row_it in range(2):
            axs[row_it, col_it].scatter(np.nonzero(ground_truths[col_it])[0],
                                            np.nonzero(ground_truths[col_it])[1],
                                                                marker='.',
                                                                linewidths=0.001,
                                                                color='red')
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    axs[0, 0].set_ylabel('Joint\n training')
    axs[1, 0].set_ylabel('Separate\n training')
    plt.savefig('results/figures/figS4_joint_training')


    # Load trained networks
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net_jointly_trained = HierarchicalNetwork(img_width=80, movement_dim=1)
    net_jointly_trained.load_state_dict(torch.load(network_paths['trained_animals_jointly'] + '_run-0'))
    net_jointly_trained.infer_all()
    net_jointly_trained.to(device)

    net_separately_trained = HierarchicalNetwork(img_width=80, movement_dim=1)
    net_separately_trained.load_state_dict(torch.load(network_paths['trained_animals'] + '_run-0'))
    net_separately_trained.infer_all()
    net_separately_trained.to(device)

    # Compute segmentation performance
    net_jointly_trained.train_all()
    IoU_joint_training = segmentation_performance(net_jointly_trained, eval_dataset, eval_frame=3, run_id='joint_training')
    IoU_separate_training = segmentation_performance(net_separately_trained, eval_dataset, eval_frame=3, run_id='separate_training')
    print(f'IoU jointly trained: {IoU_joint_training:.2f}')
    print(f'IoU separately trained: {IoU_separate_training:.2f}')

if __name__ == '__main__':
    figs_joint_training()
    plt.show()  






