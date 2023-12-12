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

# If desired, delete all weights and figures
start_from_scratch = True
if start_from_scratch:
    folders_to_delete = ['results/figures', 
                    'results/trained_models', 
                    'results/debugging', 
                    'results/intermediate'
        ]

    for folder in folders_to_delete:
        delete_all_files_in_folder(folder)

n_runs = 1# Average across this many randomly seeded simulations. 1: quick results, 4: reproduce paper

fig_learning_curves(n_runs=n_runs)
print('Learning curve plot done\n')

fig_snapshot()
print('Snapshot plot done\n')

figs_independent_movement(n_runs=n_runs)
print('Independent movement plot done\n')

# Slow!
# tab_readout(n_runs=n_runs)
# print('Readout table done\n')

fig_retinotopic_recording(condition='onset', n_runs=n_runs)
fig_retinotopic_recording(condition='removal', n_runs=n_runs)
print('Retinotopic recording plot done\n')

figs_multimove()
print('Multimove plot done\n')

