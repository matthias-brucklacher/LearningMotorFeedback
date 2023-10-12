import copy
from model2_retinotopic.network.network_hierarchical import HierarchicalNetwork
from model2_retinotopic.create_figures.figure_helpers import plot_errorband
from model2_retinotopic.network.training_configurations import train_animals_separately
from model2_retinotopic.network.network_paths import network_paths
from model2_retinotopic.network.network_helpers import OF_to_V1_DS
from model2_retinotopic.data_handling.animals_datasets import animals_training_coupled
from model2_retinotopic.data_handling.animals_datasets import animals_removal
from model2_retinotopic.during_run_analysis.analysis_functions import Recording
from model2_retinotopic.helpers import make_tensor_plottable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import torch
from torch.utils.data import DataLoader

def to_numpy_and_average(tensor):
    tensor = tensor.detach().cpu().numpy()
    tensor = np.average(tensor)
    return tensor

def get_V1_response(opticflow_input, DS_channels, show_idx, rec_x_range, rec_y_range):
    response = OF_to_V1_DS(opticflow_input)
    response = response[show_idx, DS_channels, rec_x_range[0]:rec_x_range[1] , rec_y_range[0]:rec_y_range[1]] # Select channels and spatial range
    response = to_numpy_and_average(response) # Sum over the four directional channels and the selected spatial range
    return response

def get_MT_response(net, show_idx, rec_range):
    response = net.MT_actfun(net.MT_pre)
    response = response[show_idx]
    response = response[:, rec_range[0][0]:rec_range[0][1], rec_range[1][0]:rec_range[1][1]]
    response = to_numpy_and_average(response)
    return response

def get_pe_response(pPE0, nPE0, ds_channels, show_idx, rec_range):
    response = torch.relu(pPE0)+ torch.relu(nPE0)
    response = response[show_idx, ds_channels, rec_range[0][0]:rec_range[0][1], rec_range[1][0]:rec_range[1][1]]
    response = to_numpy_and_average(response)
    return response

def fig_retinotopic_recording(n_runs=1, condition='onset'):
    assert condition in ['onset', 'removal'], f'condition must be either "onset" or "removal" but was {condition}'
    torch.manual_seed(0)

    # Train net
    train_animals_separately(n_runs=n_runs)

    # Prepare recording of neural responses
    recording = Recording('timestamp', 
                            'motor',
                            'v1_in',
                            'v1_out',
                            'mt_in',
                            'mt_out',
                            'pe_in',
                            'pe_out')
    
    # Get data
    if condition == 'onset':  
        test_dataset = animals_training_coupled()
    elif condition == 'removal':
        test_dataset = animals_removal()
    testloader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=False)
    show_idx = 0

    # Recording window
    x_start_in = 39
    y_start_in = 39
    x_start_out = 70
    y_start_out = 70

    d = 4 # width and height of window
    rec_range_v1_in = [[x_start_in, x_start_in + d], [y_start_in, y_start_in + d]] # [[x_start, x_end], [y_start, y_end]]
    rec_range_v1_out = [[x_start_out, x_start_out + d], [y_start_out, y_start_out + d]]
    rec_range_mt_in = [[x_start_in - 1, x_start_in + 3], [y_start_in - 1, y_start_in + 3]]
    rec_range_mt_out = [[x_start_out - 3, x_start_out + 1], [y_start_out - 3, y_start_out + 1]]

    ds_channels = [1]#[0, 1, 2, 3] # which of the four directions. 

    # Define some helper functions
    get_V1_response_in = lambda of_input: get_V1_response(of_input, ds_channels, show_idx, rec_range_v1_in[0], rec_range_v1_in[1])
    get_V1_response_out = lambda of_input: get_V1_response(of_input, ds_channels, show_idx, rec_range_v1_out[0], rec_range_v1_out[1])
    make_plottable = make_tensor_plottable(img_width=80)
    
    for run_it in range(n_runs):
        recording.add_run()

        # Get trained net
        net = HierarchicalNetwork(img_width=80, movement_dim=1)
        PATH_TO_TRAINED = f'{network_paths["trained_animals"]}_run-{run_it}' 
        train_animals_separately(n_runs=n_runs)
        net.load_state_dict(torch.load(PATH_TO_TRAINED))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        net.to(device)
        net.infer_all()
        i_steps = copy.deepcopy(net.I_STEPS)
        net.I_STEPS = 1 # Conduct inference one step at a time to allow temporally precise recording

        if condition == 'onset':
            rec_frame_limits = [18, 28]
        elif condition == 'removal':
            rec_frame_limits = [4, 9]
        sequence_frame_range = range(0, rec_frame_limits[1]) # No need to present beyond recording duration
        rec_istep_range = range(rec_frame_limits[0] * i_steps, rec_frame_limits[1] * i_steps)

        # Rough conversion of inference steps to ms
        isteps_to_ms = 5
        rec_timesteps = [i * isteps_to_ms for i in rec_istep_range] 
        
        # Go through one sequence of the dataset
        for i, sequences in enumerate(testloader):
            sequence = sequences[0] # Remove labels
            opticflow_input, motor_state, visual_input, _ = sequence[0].to(device), sequence[1].to(device), sequence[2], sequence[3]
            net.reset_activity(batch_size=10)
            for frame_it in sequence_frame_range:
                if condition == 'onset' and frame_it == rec_frame_limits[0]:
                    # necessary to reset activity here, as there is still object-related activity in MT from previous movement (frames 0-10) that is not
                    # washed out due to the gating, which prevents V1 errors from being calculated during passive state and thus prevents
                    # updating of MT 
                    net.reset_activity(batch_size=10)

                for istep_it in range(i_steps):
                    nPE0_pre, pPE0_pre, _, _ = net(opticflow_input[:, frame_it], motor_state[:, frame_it])
                    if frame_it in range(rec_frame_limits[0], rec_frame_limits[1]):
                        recording.update(timestamp=frame_it,
                                        v1_in=get_V1_response_in(opticflow_input[:, frame_it]),
                                        v1_out=get_V1_response_out(opticflow_input[:, frame_it]),
                                        mt_in=get_MT_response(net, show_idx, rec_range_mt_in),
                                        mt_out=get_MT_response(net, show_idx, rec_range_mt_out),
                                        motor=motor_state[show_idx, frame_it].detach().cpu().numpy()[0],
                                        pe_in=get_pe_response(nPE0_pre, pPE0_pre, ds_channels, show_idx, rec_range_v1_in),
                                        pe_out=get_pe_response(nPE0_pre, pPE0_pre, ds_channels, show_idx, rec_range_v1_out)
                        )

    rec_mean_std = recording.compute_mean_std()

    # Plot neural recordings
    fig, axs = plt.subplots(4, 2, figsize=(4.5, 5))
    axs[0, 0].axis('off')

    plot_errorband(axs[0, 1], rec_timesteps, rec_mean_std['motor'][:, 0], rec_mean_std['motor'][:, 1])
    axs[0, 1].set_ylim(top=1.1)
    axs[0, 1].set_ylabel('Speed of\ngaze and cat')

    plot_errorband(axs[1, 0], rec_timesteps, rec_mean_std['v1_in'][:, 0], rec_mean_std['v1_in'][:, 1])
    axs[1, 0].set_ylabel('V1-DS\n(on cat)')

    plot_errorband(axs[1, 1], rec_timesteps, rec_mean_std['v1_out'][:, 0], rec_mean_std['v1_out'][:, 1])
    axs[1, 1].set_ylabel('V1-DS\n(background)')
    axs[1, 1].set_ylim(top=1.1)
    axs[1, 1].sharey(axs[1, 0])

    plot_errorband(axs[2, 0], rec_timesteps, rec_mean_std['pe_in'][:, 0], rec_mean_std['pe_in'][:, 1])
    axs[2, 0].set_ylabel('V1-PE\n(on cat)')

    plot_errorband(axs[2, 1], rec_timesteps, rec_mean_std['pe_out'][:, 0], rec_mean_std['pe_out'][:, 1])
    axs[2, 1].set_ylabel('V1-PE\n(background)')
    axs[2, 1].sharey(axs[2, 0])

    plot_errorband(axs[3, 0], rec_timesteps, rec_mean_std['mt_in'][:, 0], rec_mean_std['mt_in'][:, 1])
    axs[3, 0].set_ylabel('MT-RN\n(on cat)')
    axs[3, 0].set_xlabel('Time [ms]')

    plot_errorband(axs[3, 1], rec_timesteps, rec_mean_std['mt_out'][:, 0], rec_mean_std['mt_out'][:, 1])
    axs[3, 1].set_ylabel('MT-RN\n(background)')
    axs[3, 1].sharey(axs[3, 0])
    axs[3, 1].set_xlabel('Time [ms]')

    # Remove upper and right frames
    for ax in axs.flat[1:]:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim(bottom=-0.1)
        if condition == 'removal':
            # Manually indicate area of object removal by shading
            ax.axvspan(5 * i_steps * isteps_to_ms, 6 * i_steps * isteps_to_ms, alpha=0.2, color='lightgrey')
            ax.axvspan(4 * i_steps * isteps_to_ms, 5 * i_steps * isteps_to_ms, alpha=0.2, color='darkgrey')
    plt.tight_layout()
    plt.savefig(f'results/figures/fig_recording_{condition}.png', dpi=400)

    # Plot recording window
    fig2, axs2 = plt.subplots(1, 1, figsize=(2, 2))
    axs2.matshow(make_plottable(visual_input[show_idx, 0]))
    axs2.set_ylabel('Visual input')
    anchor_point_in = [rec_range_v1_in[0][0], rec_range_v1_in[1][0]]
    anchor_point_out = [rec_range_v1_out[0][0], rec_range_v1_out[1][0]]
    rec_width = rec_range_v1_in[0][1] - rec_range_v1_in[0][0]
    rec_height = rec_range_v1_in[1][1] - rec_range_v1_in[1][0]
    rec_window_in = patches.Rectangle((anchor_point_in[0], anchor_point_in[1]), rec_width, rec_height, linewidth=2, edgecolor='r', facecolor='none')
    rec_window_out = patches.Rectangle((anchor_point_out[0], anchor_point_out[1]), rec_width, rec_height, linewidth=2, edgecolor='r', facecolor='none')
    axs2.add_patch(rec_window_in)
    axs2.add_patch(rec_window_out)

    # Cosmetics for image plot
    axs2.xaxis.set_tick_params(labelbottom=False)
    axs2.yaxis.set_tick_params(labelleft=False)
    axs2.set_xticks([])
    axs2.set_yticks([])
    plt.savefig(f'results/figures/fig_recording_window.png', dpi=400)

if __name__ == '__main__':
    fig_retinotopic_recording(condition='onset', n_runs=2)
    fig_retinotopic_recording(condition='removal', n_runs=2)    
    plt.show()