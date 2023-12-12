from model2_retinotopic.network.train import train
from model2_retinotopic.data_handling.animals_datasets import animals_independently_moving
from model2_retinotopic.network.training_configurations import pretrain_animals
from model2_retinotopic.network.network_paths import network_paths
from model2_retinotopic.create_figures.segmentation_baseline import segmentation_optimal_baseline
from model2_retinotopic.create_figures.figure_helpers import plot_errorband 
import matplotlib.pyplot as plt
import numpy as np

def figs_independent_movement(n_runs=1):
    plt.style.use('model2_retinotopic.create_figures.plotting_style')

    if True:
        args = {
            'n_runs': n_runs, 
            'n_epochs_pretrain': 10,
            'n_epochs_train': 4,
        }

        path_trained = network_paths['trained_animals_independently_moving']

        pretrain_animals(n_runs=args['n_runs'])

        IoU_mean = []
        IoU_std = []
        speed = []
        baseline = []
        eval_frame = {0: 9, 1: 9, 2: 6, 4: 4}
        for relative_speed in eval_frame.keys():
            speed.append(relative_speed)
            train_dataset = animals_independently_moving(relative_speed=relative_speed)
            train_recording = train(n_runs=args['n_runs'],
                                    n_epochs=args['n_epochs_train'],
                                    learning_rate=0.02,
                                    load_path=network_paths['pretrained_animals'],
                                    save_path=path_trained,
                                    train_dataset=train_dataset,
                                    run_id=f'speed-{relative_speed}',
                                    eval_frame=eval_frame[relative_speed]
                                    )
            IoU_mean.append(train_recording['IoU'][-1, 0])
            IoU_std.append(train_recording['IoU'][-1, 1])
            optimal_baseline = segmentation_optimal_baseline(dataset=train_dataset, eval_frame=3)
            baseline.append(optimal_baseline)
    else:
        # Dummy data to more quickly improve plot
        IoU_mean = [0.70, 0.65, 0.60, 0.55]
        IoU_std = [0.01, 0.01, 0.01, 0.01]
        speed = [0, 1, 2, 4]
        baseline = [0.74, 0.68, 0.64, 0.58]

    fig, ax = plt.subplots()
    plot_errorband(ax, speed, IoU_mean, IoU_std, label='Sensorimotor PC')
    ax.plot(speed, baseline, c='purple', label='Best instantaneous')
    ax.set_xticks(speed)
    ax.set_ylim(bottom=0.4, top=0.8)
    ax.set_xlabel('Relative speed \nobject vs. observer')
    ax.set_ylabel('Segmentation\noverlap (IoU)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/figures/fig6D_independent_movement')

if __name__ == '__main__':
    figs_independent_movement()
    plt.show()