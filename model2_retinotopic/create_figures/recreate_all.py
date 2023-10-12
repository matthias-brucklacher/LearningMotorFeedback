import os
import matplotlib.pyplot as plt
from model2_retinotopic.create_figures.fig_learning_curves import fig_learning_curves
from model2_retinotopic.create_figures.fig_snapshot import fig_snapshot
from model2_retinotopic.create_figures.fig_iou import fig_iou
from model2_retinotopic.create_figures.figs_joint_training import figs_joint_training
from model2_retinotopic.create_figures.figs_multimove import figs_multimove
from model2_retinotopic.create_figures.figs_independent_movement import figs_independent_movement
from model2_retinotopic.create_figures.fig_retinotopic_recording import fig_retinotopic_recording
from model2_retinotopic.create_figures.tab_readout import tab_readout
from model2_retinotopic.helpers import delete_all_files_in_folder

# If desired, delete the following folders
folders_to_delete = [#'results/figures', 
                #'results/trained_models', 
                'results/debugging', 
                #'results/intermediate'
    ]

# Skip retraining query in training_configuration to speed up things

for folder in folders_to_delete:
    delete_all_files_in_folder(folder)
n_scripts = 2
n_runs = 4

# fig_learning_curves(n_runs=n_runs)
# print(f'1/{n_scripts} Learning curve plot done\n')

# fig_iou(n_runs=n_runs)
# print(f'2/{n_scripts} IoU plot done\n')

# fig_snapshot()
# print(f'3/{n_scripts} Snapshot plot done\n')

# figs_independent_movement(n_runs=n_runs)
# print(f'6/{n_scripts} Independent movement plot done\n')

# Slow!
# tab_readout(n_runs=n_runs)
# print(f'4/{n_scripts} Readout table done\n')

# fig_retinotopic_recording(condition='onset', n_runs=n_runs)
# fig_retinotopic_recording(condition='removal', n_runs=n_runs)
# print(f'5/{n_scripts} Retinotopic recording plot done\n')

# figs_multimove()
# print(f'7/{n_scripts} Multimove plot done\n')

k

#plt.show()
 