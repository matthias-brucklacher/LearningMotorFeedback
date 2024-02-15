"""Trains the motor-to-visual streams of two networks, one in the non-coupled paradigm and one in the coupled paradigm. 
    Subsequently, the visual-to-visual stream of the coupled network is activated and trained. 

"""

import matplotlib.pyplot as plt
from model2_retinotopic.create_figures.figure_helpers import plot_errorband
from model2_retinotopic.network.training_configurations import pretrain_animals_noncoupled, train_animals_nonempty, params_learning_curve
import numpy as np


def fig_learning_curves(n_runs=1):
    plt.style.use('model2_retinotopic.create_figures.plotting_style')

    args = {
        'n_runs': n_runs,
        'n_epochs_pretrain': params_learning_curve['epochs_pretrain'],
    }

    # 1A: Training motor-to-visual pathway (non-coupled)
    noncoupled_recording = pretrain_animals_noncoupled(n_runs=args['n_runs'])
    noncoupled_error = noncoupled_recording['error']

    # 2A-B: Training motor-to-visual pathway (coupled) and Training visual-to-visual pathway (coupled)
    pretrain_recording, train_recording = train_animals_nonempty(n_runs=args['n_runs'])
    pretrain_error = pretrain_recording['error']
    train_error = train_recording['error']
    coupled_error = np.concatenate((pretrain_error, train_error[1:]), axis=0)

    # 3: Plot reconstruction errors
    fig, ax = plt.subplots(1)
    plot_errorband(ax, range(1, len(coupled_error)), coupled_error[1:, 0], coupled_error[1:, 1], color='blue', label='CT')
    plot_errorband(ax, range(1, args['n_epochs_pretrain'] + 1), noncoupled_error[1:, 0], noncoupled_error[1:, 1], color='r', label='NT')

    ax.vlines(x=args['n_epochs_pretrain'], ymin=0.1, ymax=ax.get_ylim()[1], colors='purple')
    ax.annotate(' Activation of\n mHVA-V1 stream',
            xy=(10, 1),  
            xytext=(14, 1),  
            arrowprops=dict(facecolor='black', shrink=0.01),
            horizontalalignment='left',
            verticalalignment='center'
    )
    ax.set_yscale('log')
    ax.set_ylim(bottom=0.9 * np.min(coupled_error[1:, 0]))
    ax.set_xlabel('Training epochs')
    ax.set_ylabel('Prediction error (MSE)')
    ax.tick_params(axis='both', which='major')
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig('results/figures/fig5B_learning.png', dpi=400)

    # Print relevant values
    print(f'Non-coupled training error: {noncoupled_error[-1, 0]:.3f} +- {noncoupled_error[-1, 1]:.3f}')
    print(f'Coupled training error: {coupled_error[args["n_epochs_pretrain"], 0]:.3f} +- {coupled_error[args["n_epochs_pretrain"], 1]:.3f}')
    print(f'Coupled training error after activation of visual stream: {train_error[-1, 0]:.3f} +- {train_error[-1, 1]:.3f}')

if __name__ == '__main__':
    fig_learning_curves(n_runs=1)
    plt.show()