import numpy as np
import copy
from model1_global.global_model import GlobalNet as GlobalNet
from model1_global.fig2_train import get_trained_nets_ct_nt, get_data
from model1_global.mismatch_functions import calcium_imaging_on_sequence, mismatch_plot, create_mismatch_inputs
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Prepare recordings
    deltaf_ct = {'mismatch': [], 'playback_halt_standing': [], 'playback_halt_stopping': []}
    deltaf_nt = copy.deepcopy(deltaf_ct)
    deltaf_nt_day5 = copy.deepcopy(deltaf_ct)
    deltaf_ct_day5 = copy.deepcopy(deltaf_ct)

    # Input sequences for the different experiments
    input_visual, motor = create_mismatch_inputs()

    n_runs = 15
    for run_it in range(n_runs):
        # Append new trial to recordings
        for recording in [deltaf_ct, deltaf_nt, deltaf_nt_day5, deltaf_ct_day5]:
            for test_paradigm in recording.keys():
                recording[test_paradigm].append([])

        # Create and train networks
        ct_net, nt_net = get_trained_nets_ct_nt(fixed_seed=False)
        nt_net_day5 = copy.deepcopy(nt_net)
        ct_net_day5 = copy.deepcopy(ct_net)
        optic_flow, motor_ct, motor_nt = get_data(data_type='binary')
        nt_net_day5.training_epoch(optic_flow, motor_ct)
        ct_net_day5.training_epoch(optic_flow, motor_ct)

        # Run tests on trained networks, append to recordings
        for test_paradigm in deltaf_ct.keys():
            deltaf_ct[test_paradigm][-1].append(calcium_imaging_on_sequence(ct_net, input_visual[test_paradigm], motor[test_paradigm]))
            deltaf_nt[test_paradigm][-1].append(calcium_imaging_on_sequence(nt_net, input_visual[test_paradigm], motor[test_paradigm]))
            deltaf_nt_day5[test_paradigm][-1].append(calcium_imaging_on_sequence(nt_net_day5, input_visual[test_paradigm], motor[test_paradigm]))
            deltaf_ct_day5[test_paradigm][-1].append(calcium_imaging_on_sequence(ct_net_day5, input_visual[test_paradigm], motor[test_paradigm]))

    # Convert to numpy arrays, compute mean and std
    for test_paradigm in deltaf_ct.keys():
        deltaf_ct[test_paradigm] = np.array(deltaf_ct[test_paradigm])
        deltaf_nt[test_paradigm] = np.array(deltaf_nt[test_paradigm])
        deltaf_nt_day5[test_paradigm] = np.array(deltaf_nt_day5[test_paradigm])
        deltaf_ct_day5[test_paradigm] = np.array(deltaf_ct_day5[test_paradigm])

        deltaf_ct[test_paradigm] = (np.mean(deltaf_ct[test_paradigm], axis=0)[0], np.std(deltaf_ct[test_paradigm], axis=0)[0])
        deltaf_nt[test_paradigm] = (np.mean(deltaf_nt[test_paradigm], axis=0)[0], np.std(deltaf_nt[test_paradigm], axis=0)[0])
        deltaf_nt_day5[test_paradigm] = (np.mean(deltaf_nt_day5[test_paradigm], axis=0)[0], np.std(deltaf_nt_day5[test_paradigm], axis=0)[0])
        deltaf_ct_day5[test_paradigm] = (np.mean(deltaf_ct_day5[test_paradigm], axis=0)[0], np.std(deltaf_ct_day5[test_paradigm], axis=0)[0])

    # Plotting
    mismatch_plot(deltaf_ct, deltaf_nt, deltaf_nt_day5, deltaf_ct_day5, input_visual, motor)
    plt.show()