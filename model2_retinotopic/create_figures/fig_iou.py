"""Trains the motor-to-visual streams of two networks, one in the non-coupled paradigm and one in the coupled paradigm. 
    Subsequently, the visual-to-visual stream of the coupled network is activated and trained. 

"""

import matplotlib.pyplot as plt
from model2_retinotopic.data_handling.animals_datasets import animals_training_coupled
from model2_retinotopic.create_figures.figure_helpers import plot_errorband
from model2_retinotopic.network.training_configurations import train_animals_separately
from model2_retinotopic.create_figures.segmentation_baseline import segmentation_optimal_baseline

def fig_iou(n_runs=1):
    train_dataset = animals_training_coupled()
    _, train_recording = train_animals_separately(n_runs=n_runs)

    # Get optimal baseline
    baseline = segmentation_optimal_baseline(dataset=train_dataset, eval_frame=9)

    # # Plot IoU over training time
    plt.style.use('model2_retinotopic.create_figures.plotting_style')
    fig, ax = plt.subplots()
    epochs = train_recording['timestamp'][:, 0]

    plot_errorband(ax, epochs, train_recording['IoU'][:, 0], train_recording['IoU'][:, 1], color='blue', label='Sensorimotor PC')
    ax.hlines(y=baseline, xmin=0, xmax=len(epochs) - 1, colors='purple', label='Best instantaneous')

    ax.set_xlabel('Training epochs')
    ax.set_ylabel('Segmentation\noverlap (IoU)')
    ax.tick_params(axis='both', which='major')
    plt.xticks(epochs[::5])
    ax.set_ylim(bottom=0, top=0.73)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('results/figures/fig4_IoU')

if __name__ == '__main__':
    fig_iou(n_runs=2)
    plt.show()