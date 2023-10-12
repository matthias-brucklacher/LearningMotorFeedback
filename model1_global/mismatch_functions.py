import numpy as np
from model1_global.global_model import GlobalNet as GlobalNet
import matplotlib.pyplot as plt

def create_mismatch_inputs():
    """Creates input sequences for the mismatch experiment.

    Returns:
        input_visual (dict): Dictionary of visual input sequences.
        motor (dict): Dictionary of motor input sequences.

    """
    # Segments of which to compose the input sequences
    constant_on = np.ones(10)
    constant_off_short = np.zeros(5)
    constant_on_short = np.ones(5)

    input_visual = {'mismatch': [], 'playback_halt_standing': [], 'playback_halt_stopping': []}
    for test_paradigm in input_visual.keys():
        input_visual[test_paradigm] = np.concatenate([constant_on, constant_off_short, constant_on]) # Visual input always the same (on-off-on)
    motor = dict.fromkeys(input_visual)
    motor['mismatch'] = np.concatenate([constant_on, constant_on_short, constant_on])
    motor['playback_halt_standing'] = np.zeros(len(motor['mismatch']))
    motor['playback_halt_stopping'] = np.concatenate([constant_on, constant_off_short, constant_on])
    return input_visual, motor

def preprocessing(recorded_responses,):
    """
    Preprocesses the recorded responses by subtracting the baseline response from each signal.

    Args:
        recorded_responses (list): List of recorded responses.

    Returns:
        delta_f (list): List of delta_f responses.
    """
    delta_f = []
    baseline = recorded_responses[-1] # Under the condition that the last element is the average case
    for signal in recorded_responses:
        delta_f.append(np.absolute(signal) - baseline)
    return delta_f

def calcium_imaging_on_sequence(net, visflow_seq, motor_seq):
    """
    Runs the network on a sequence of optic flow and motor commands and returns the calcium imaging response.

    Args:
        net (GlobalNet): Network to run.
        visflow_seq (list): Sequence of optic flow values.
        motor_seq (list): Sequence of motor values.

    Returns:
        ca2_response (list): List of calcium imaging responses.
    """
    assert len(visflow_seq) == len(motor_seq), "Optic flow sequence of different length than motor sequence"
    timesteps = len(visflow_seq)
    error_list = []
    for timeit in range(timesteps):
        net.infer_step(visflow_seq[timeit], motor_seq[timeit])
        error_list.append(np.abs(net.pPE[0, 0]) + np.abs(net.nPE[0, 0]))
    ca2_response = preprocessing(error_list)
    return ca2_response

def mismatch_plot(deltaf_ct, deltaf_nt, deltaf_nt_day5, deltaf_ct_day5, input_visual, motor):
    """Plots the mismatch experiment.

    Args:
        deltaf_ct (dict): Dictionary of calcium imaging responses for CT.
        deltaf_nt (dict): Dictionary of calcium imaging responses for NT.
        input_visual (dict): Dictionary of visual input sequences.
        motor (dict): Dictionary of motor input sequences.
    
    """
    # Set up plotting
    plot_range_lower = 9
    plot_range_upper = 20
    time_range = [i/5 for i in range(-1, plot_range_upper - plot_range_lower - 1)] # Scale to match the time axis of the experimental data

    # Plot 1: Mismatch CT vs NT
    fig, axs = plt.subplots(3, 3, sharey=True)

    # 1.1 Mismatch (first column)
    axs[0, 0].fill_between(time_range, deltaf_ct['mismatch'][0][plot_range_lower:plot_range_upper] - 0.5 * deltaf_ct['mismatch'][1][plot_range_lower:plot_range_upper], deltaf_ct['mismatch'][0][plot_range_lower:plot_range_upper] + 0.5 * deltaf_ct['mismatch'][1][plot_range_lower:plot_range_upper], color='blue', label='CT', alpha=1)
    axs[0, 0].plot(time_range, deltaf_nt['mismatch'][0][plot_range_lower:plot_range_upper], 'r', label='NT')
    axs[0, 0].fill_between(time_range, deltaf_nt['mismatch'][0][plot_range_lower:plot_range_upper] - 0.5 * deltaf_nt['mismatch'][1][plot_range_lower:plot_range_upper], deltaf_nt['mismatch'][0][plot_range_lower:plot_range_upper] + 0.5 * deltaf_nt['mismatch'][1][plot_range_lower:plot_range_upper], color='red', label='NT', alpha=0.5)
    axs[0, 0].set_xticklabels([])
    axs[0, 0].set_ylim([-0.1, 1.1])
    axs[0, 0].set_ylabel('$\Delta$ F')
    axs[0, 0].set_title('Mismatch', weight='bold')

    axs[1, 0].plot(time_range, input_visual['mismatch'][plot_range_lower:plot_range_upper], 'tab:green')
    axs[1, 0].set_ylabel('Optic Flow')
    axs[1, 0].set_xticklabels([])

    axs[2, 0].plot(time_range, motor['mismatch'][plot_range_lower:plot_range_upper], 'tab:purple')
    axs[2, 0].set_ylabel('Running')
    axs[2, 0].set_xlabel('Time [s]')

    # 1.2 Playback halt standing (second column)
    axs[0, 1].plot(time_range, deltaf_ct['playback_halt_standing'][0][plot_range_lower:plot_range_upper], 'b--', label='CT', alpha=0.7)
    axs[0, 1].plot(time_range, deltaf_nt['playback_halt_standing'][0][plot_range_lower:plot_range_upper], 'r--', dashes=(5, 1), label='NT', alpha=0.7)
    axs[0, 1].set_xticklabels([])
    axs[0, 1].legend(frameon=False)
    axs[0, 1].set_title('Playback halt', weight='bold')

    axs[1, 1].plot(time_range, input_visual['playback_halt_standing'][plot_range_lower:plot_range_upper], 'tab:green')
    axs[1, 1].set_xticklabels([])

    axs[2, 1].plot(time_range, motor['playback_halt_standing'][plot_range_lower:plot_range_upper], 'tab:purple')

    # 1.3 Playback halt stopping (third column)
    # axs[0, 2].plot(time_range, deltaf_ct['playback_halt_stopping'][0][plot_range_lower:plot_range_upper], 'b--', label='CT')
    # axs[0, 2].plot(time_range, deltaf_nt['playback_halt_stopping'][0][plot_range_lower:plot_range_upper], 'r--', label='NT')

    # axs[0, 2].set_xticklabels([])
    # axs[0, 2].set_title('Playback halt 2', weight='bold')

    # axs[1, 2].plot(time_range, input_visual['playback_halt_stopping'][plot_range_lower:plot_range_upper], 'tab:green')
    # axs[1, 2].set_xticklabels([])

    # axs[2, 2].plot(time_range, motor['playback_halt_stopping'][plot_range_lower:plot_range_upper], 'tab:purple')
    for i in range(3):
        axs[i, 2].axis('off')

    for ax in axs.reshape(-1):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig('results/figures/fig3_mismatch.png', dpi=400)

    # Plot 2: Recovery
    fig, axs = plt.subplots(3, 3, sharey=True)

    # 2.1 Day 1 (first column)
    axs[0, 0].plot(time_range, deltaf_nt['mismatch'][0][plot_range_lower:plot_range_upper], color='red', label='Mismatch')
    axs[0, 0].fill_between(time_range, deltaf_nt['mismatch'][0][plot_range_lower:plot_range_upper] - 0.5 * deltaf_nt['mismatch'][1][plot_range_lower:plot_range_upper], deltaf_nt['mismatch'][0][plot_range_lower:plot_range_upper] + 0.5 * deltaf_nt['mismatch'][1][plot_range_lower:plot_range_upper], color='red', alpha=0.5)
    axs[0, 0].plot(time_range, deltaf_nt['playback_halt_standing'][0][plot_range_lower:plot_range_upper], 'r--', label='Playback halt')

    axs[0, 0].set_xticklabels([])
    axs[0, 0].set_ylabel('$\Delta$ F')
    axs[0, 0].set_ylim([-0.1, 1.1])
    axs[0, 0].set_title('Day 1', weight='bold')


    axs[1, 0].plot(time_range, deltaf_ct['mismatch'][0][plot_range_lower:plot_range_upper], color='blue', label='Mismatch')
    axs[1, 0].fill_between(time_range, deltaf_ct['mismatch'][0][plot_range_lower:plot_range_upper] - 0.5 * deltaf_ct['mismatch'][1][plot_range_lower:plot_range_upper], deltaf_ct['mismatch'][0][plot_range_lower:plot_range_upper] + 0.5 * deltaf_ct['mismatch'][1][plot_range_lower:plot_range_upper], color='blue', alpha=1)
    axs[1, 0].plot(time_range, deltaf_ct['playback_halt_standing'][0][plot_range_lower:plot_range_upper], 'b--', label='Playback halt')
    axs[1, 0].set_ylabel('$\Delta$ F')
    axs[1, 0].set_xlabel('Time [s]')
    #axs[1, 0].set_xticklabels([])


    # 2.2 Day 5 (second column)
    axs[0, 1].plot(time_range, deltaf_nt_day5['playback_halt_standing'][0][plot_range_lower:plot_range_upper], 'r--', label='Playback halt')
    axs[0, 1].set_xticklabels([])
    axs[0, 1].set_title('Day 5', weight='bold')
    axs[0, 1].plot(time_range, deltaf_nt_day5['mismatch'][0][plot_range_lower:plot_range_upper], color='red', label='Mismatch')
    axs[0, 1].legend(frameon=False)
    axs[0, 1].fill_between(time_range, deltaf_nt_day5['mismatch'][0][plot_range_lower:plot_range_upper] - 0.5 * deltaf_nt_day5['mismatch'][1][plot_range_lower:plot_range_upper], deltaf_nt_day5['mismatch'][0][plot_range_lower:plot_range_upper] + 0.5 * deltaf_nt_day5['mismatch'][1][plot_range_lower:plot_range_upper], color='red', alpha=0.5)

    axs[1, 1].plot(time_range, deltaf_ct_day5['playback_halt_standing'][0][plot_range_lower:plot_range_upper], 'b--', label='Playback halt')
    axs[1, 1].plot(time_range, deltaf_ct_day5['mismatch'][0][plot_range_lower:plot_range_upper], color='blue', label='Mismatch')
    axs[1, 1].fill_between(time_range, deltaf_ct_day5['mismatch'][0][plot_range_lower:plot_range_upper] - 0.5 * deltaf_ct_day5['mismatch'][1][plot_range_lower:plot_range_upper], deltaf_ct_day5['mismatch'][0][plot_range_lower:plot_range_upper] + 0.5 * deltaf_ct_day5['mismatch'][1][plot_range_lower:plot_range_upper], color='blue', alpha=1)
    
    for i in range(3):
        axs[2, i].axis('off')
        axs[i, 2].axis('off')
    #axs[1, 1].set_xticklabels([])


    for ax in axs.reshape(-1):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.savefig('results/figures/fig3_recovery.png', dpi=400)


