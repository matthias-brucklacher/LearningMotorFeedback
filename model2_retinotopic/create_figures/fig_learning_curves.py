import matplotlib.pyplot as plt
from model2_retinotopic.create_figures.figure_helpers import plot_errorband
from model2_retinotopic.network.training_configurations import pretrain_animals_noncoupled, train_animals_nonempty, params_learning_curve, train_animals_separately, pretrain_animals
import numpy as np
from matplotlib.lines import Line2D

def fig_learning_curves(n_runs=1):
    plt.style.use('model2_retinotopic.create_figures.plotting_style')

    args = {
        'n_runs': n_runs,
        'n_epochs_pretrain': params_learning_curve['epochs_pretrain'],
    }
    if True: # Disable for faster plotting from saved data
        # 1A: Pretraining motor-to-visual pathway (non-coupled)
        pretrain_noncoupled_recording = pretrain_animals_noncoupled(n_runs=args['n_runs'])
        pretrain_noncoupled_error = pretrain_noncoupled_recording['error']

        # 2B: Pretraining motor-to-visual pathway (coupled) on empty sequences, compute prediction error with objects
        pretrain_recording_tested_with_objects = pretrain_animals(n_runs=args['n_runs'], test_with_animals=True)
        pretrain_error_tested_with_objects = pretrain_recording_tested_with_objects['error']

        # 2C: Training motor-to-visual pathway (coupled) and Training visual-to-visual pathway (coupled). Pretraining is evaluated without objects.
        pretrain_recording, train_recording = train_animals_separately(n_runs=args['n_runs'])
        pretrain_error = pretrain_recording['error']
        train_error = train_recording['error']

        # Combine pretraining and training errors evaluated with objects
        separate_training_evaluated_with_objects = np.concatenate((pretrain_error_tested_with_objects, train_error[1:]), axis=0)

        # 2D: Additional control: Training motor-to-visual stream with objects
        single_stream_with_objects_recording, _ = train_animals_nonempty(n_runs=args['n_runs'], single_stream=True)
        single_stream_with_objects_error = single_stream_with_objects_recording['error']
        print(f'Single stream with objects: {single_stream_with_objects_error[-1, 0]:.3f} +- {single_stream_with_objects_error[-1, 1]:.3f}')

        # Save data
        np.save('results/fig5B_learning_curves.npy', {
        'noncoupled_error': pretrain_noncoupled_error,
        'pretrain_error': pretrain_error,
        'separate_training_evaluated_with_objects': separate_training_evaluated_with_objects
        })
    else:
        #Load data
        data = np.load('results/fig5B_learning_curves.npy', allow_pickle=True)
        pretrain_noncoupled_error = data.item().get('noncoupled_error')
        pretrain_error = data.item().get('pretrain_error')
        single_stream_with_objects_error = data.item().get('single_stream_with_objects_error')
        separate_training_evaluated_with_objects = data.item().get('separate_training_evaluated_with_objects')

    # 3: Plot reconstruction errors
    fig, ax = plt.subplots(1)
    plot_errorband(ax, range(1, len(separate_training_evaluated_with_objects)), separate_training_evaluated_with_objects[1:, 0], separate_training_evaluated_with_objects[1:, 1], color='blue', label='CT')
    plot_errorband(ax, range(1, args['n_epochs_pretrain'] + 1), pretrain_error[1:, 0], pretrain_error[1:, 1], color='blue', label='CT', linestyle = '--')
    plot_errorband(ax, range(1, args['n_epochs_pretrain'] + 1), pretrain_noncoupled_error[1:, 0], pretrain_noncoupled_error[1:, 1], color='r', linestyle = '--', label='NT')

    ax.vlines(x=args['n_epochs_pretrain'], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], colors='purple')
    ax.annotate(' Activation\n mHVA-V1\n stream',
            xy=(10, 2),  
            xytext=(14, 2),  
            arrowprops=dict(facecolor='black', shrink=0.01),
            horizontalalignment='left',
            verticalalignment='center'
    )
    ax.set_yscale('log')
    ax.set_ylim(bottom=0.9 * np.min(pretrain_error[1:, 0]))
    ax.set_xlabel('Training epochs')
    ax.set_ylabel('Prediction error (MSE)')
    ax.tick_params(axis='both', which='major')
    neutral_color = 'grey'
    custom_lines = [Line2D([0], [0], color='r', lw=3),
                    Line2D([0], [0], color='blue', lw=3),
                    Line2D([0], [0], color=neutral_color, linestyle='--', lw=3),
                    Line2D([0], [0], color=neutral_color, lw=3)]
    ax.legend(custom_lines, ['Non-coupled', 'Coupled', 'No object at test time', 'Object at test time'], bbox_to_anchor=(0.7, 1.35), loc='upper left')# bbox_to_anchor=(1.04, 1.04)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.savefig('results/figures/fig5B_learning.png', dpi=400)

    # Print relevant values
    print(f'Non-coupled pretraining error: {pretrain_noncoupled_error[-1, 0]:.3f} +- {pretrain_noncoupled_error[-1, 1]:.3f}')
    print(f'Coupled pretraining error: {pretrain_error[-1, 0]:.3f} +- {pretrain_error[-1, 1]:.3f}')
    print(f'Coupled pretraining error eval w/ objects: {separate_training_evaluated_with_objects[args["n_epochs_pretrain"], 0]:.3f} +- {separate_training_evaluated_with_objects[args["n_epochs_pretrain"], 1]:.3f}')
    print(f'Coupled training error after activation of visual stream: {separate_training_evaluated_with_objects[-1, 0]:.3f} +- {separate_training_evaluated_with_objects[-1, 1]:.3f}')

if __name__ == '__main__':
    fig_learning_curves(n_runs=1)
    plt.show()