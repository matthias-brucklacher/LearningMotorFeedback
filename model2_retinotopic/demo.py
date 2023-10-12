"""Evaluates a network (with fixed weights) on the segmentation task.

"""

from model2_retinotopic.network.network_hierarchical import HierarchicalNetwork
from model2_retinotopic.network.train import train
from model2_retinotopic.during_run_analysis.segmentation import get_boundary
from model2_retinotopic.data_handling.animals_datasets import empty_like_animals_coupled, animals_training_coupled, animals_independently_moving, animals_interleaved_with_empty
from model2_retinotopic.data_handling.fashionmnist_datasets import fashionmnist_train, empty_like_fashionmnist_train
from model2_retinotopic.data_handling.multimove_dataset import empty_multimove
from model2_retinotopic.network.network_paths import network_paths
from model2_retinotopic.helpers import make_tensor_plottable
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
import torch
from torch.utils.data import DataLoader

def fig4_segmentation(path_to_model, inference_mode, dataset, show_idx, V1_channel, show_details=False):
    """Animated network performance.
    """
    args = {
        'batch_size': 10
        }

    # Set random seed
    torch.manual_seed(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Construct dataloader
    data_loader = DataLoader(dataset=dataset, batch_size=args['batch_size'], shuffle=False)

    # Load trained network, put on GPU
    img_width = next(enumerate(data_loader))[1][0][0].shape[-2]
    movements_dim = next(enumerate(data_loader))[1][0][1].shape[-1]
    net = HierarchicalNetwork(img_width=img_width, movement_dim=movements_dim)
    net.load_state_dict(torch.load(path_to_model))
    if inference_mode == 'all':
        net.infer_all()
    elif inference_mode == 'stream_1':
        net.infer_stream_1()
    net.to(device)

    recording_for_animation = {'optic_flow': [],
                               'visual_input': [],
                               'predicted_segmentation_area': [],
                               'object_boundary_groundtruth': [],
                               'motor_pred_pos': [],
                               'motor_pred_neg': [],
                               'vis_pred_pos': [],
                               'vis_pred_neg': [],
                               'nPE0': [],
                               'pPE0': []
                               }

    make_plottable = make_tensor_plottable(img_width=img_width)

    # Go through dataset
    with torch.no_grad():
        for batch_it, data in enumerate(data_loader):
            data = data[0] # Remove class labels
            if batch_it > 0: # Only do one batch, this suffices to illustrate the segmentation
                break
            net.reset_activity(batch_size=data[0].shape[0])
            optic_flow, movement, visual, segmentation_label = data[0].to(device), data[1].to(device), data[2], data[3]
            for frame_it in range(data[0].shape[1]):
                # Propagate through network
                nPE0, pPE0, _, _ = net(optic_flow[:, frame_it], movement[:, frame_it])

                predicted_segmentation_area = np.zeros((net.img_width, net.img_width))
                if torch.sum(movement[show_idx, frame_it]) != 0:
                    predicted_segmentation_area = net.segment()[show_idx]
                #V1_channel = 1 # 0 to 4
                if V1_channel in [0, 1]:
                    OF_channel = 0
                else:
                    OF_channel = 1
                object_boundary_groundtruth = get_boundary(segmentation_label[show_idx, frame_it])
                optic_flow_plottable = make_plottable(optic_flow[show_idx, frame_it, :, :, OF_channel])
                
                recording_for_animation['optic_flow'].append(optic_flow_plottable)
                recording_for_animation['predicted_segmentation_area'].append(predicted_segmentation_area)
                recording_for_animation['object_boundary_groundtruth'].append(object_boundary_groundtruth)

                nPE0_plottable = make_plottable(nPE0[show_idx, V1_channel])
                pPE0_plottable = make_plottable(pPE0[show_idx, V1_channel])

                visual_input_plottable = make_plottable(visual[show_idx, frame_it, :, :])

                motor_pred_pos = net.motor_predictions(type='pos')
                motor_pred_pos = make_plottable(motor_pred_pos[show_idx, V1_channel])
                motor_pred_neg = net.motor_predictions(type='neg')
                motor_pred_neg = make_plottable(motor_pred_neg[show_idx, V1_channel])

                vis_pred_pos = net.predictions_from_MT(visual_prediction_relay=True, MT=net.MT_actfun(net.MT_pre), type='pos')
                vis_pred_pos = make_plottable(vis_pred_pos[show_idx, V1_channel])
                vis_pred_neg = net.predictions_from_MT(visual_prediction_relay=True, MT=net.MT_actfun(net.MT_pre), type='neg')
                vis_pred_neg = make_plottable(vis_pred_neg[show_idx, V1_channel])

                recording_for_animation['visual_input'].append(visual_input_plottable)
                recording_for_animation['motor_pred_pos'].append(motor_pred_pos)
                recording_for_animation['motor_pred_neg'].append(motor_pred_neg)
                recording_for_animation['vis_pred_pos'].append(vis_pred_pos)
                recording_for_animation['vis_pred_neg'].append(vis_pred_neg)
                recording_for_animation['nPE0'].append(nPE0_plottable)
                recording_for_animation['pPE0'].append(pPE0_plottable)

    return recording_for_animation

def remove_ticks(axs):
    for ax in axs.flat:
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)

        # Hide X and Y axes tick marks
        ax.set_xticks([])
        ax.set_yticks([])
    return axs

def update_fig1(i):
    # Pass through network and update plots
    axs1[0].clear()
    axs1[0].imshow(recording_for_animation['optic_flow'][i])
    axs1[1].clear()
    axs1[1].imshow(recording_for_animation['predicted_segmentation_area'][i])
    axs1[1].scatter(np.nonzero(recording_for_animation['object_boundary_groundtruth'][i])[0],
                                np.nonzero(recording_for_animation['object_boundary_groundtruth'][i])[1],
                                                    marker='.',
                                                    linewidths=0.001,
                                                    color='black')
    axs1[0].set_title(f'Frame {i}')

def update_fig2(i):
    axs2[0, 0].imshow(recording_for_animation['visual_input'][i])
    axs2[0, 0].set_title(f'Frame {i}')
    of_plot.set_data(recording_for_animation['optic_flow'][i])

    motor_pred_pos_plot.set_data(recording_for_animation['motor_pred_pos'][i])
    motor_pred_neg_plot.set_data(recording_for_animation['motor_pred_neg'][i])

    vis_pred_pos_plot.set_data(recording_for_animation['vis_pred_pos'][i])
    vis_pred_neg_plot.set_data(recording_for_animation['vis_pred_neg'][i])

    pPE_plot.set_data(recording_for_animation['pPE0'][i])
    nPE_plot.set_data(recording_for_animation['nPE0'][i])

if __name__ == '__main__':  
    show_details = False
    dataset = 'animals' # 'animals', 'fashionmnist', 'multimove', 'animals_independently_moving'
    training_paradigm = 'separate' # Train streams in 'joint' or 'separate' manner
    show_idx = 3 # index of the sample to show. 
    V1_channel = 1 # Which direction-selective subpopulation to display. 0 to 4.
    if dataset == 'animals':
        pretrain_dataset = empty_like_animals_coupled()
        train_dataset = animals_training_coupled()
        path_pretrained = 'results/intermediate/pretrained_net_animals'
        path_trained = 'results/intermediate/trained_net_animals'
        PATH_TO_TRAINED = network_paths['trained_animals'] + '_run-0' 
    elif dataset == 'animals_independently_moving':
        pretrain_dataset = empty_like_animals_coupled()
        train_dataset = animals_independently_moving(relative_speed=4)
        path_pretrained = 'results/intermediate/pretrained_net_animals'
        path_trained = 'results/intermediate/trained_net_animals_independently_moving'
    elif dataset == 'fashionmnist':
        pretrain_dataset = torch.utils.data.Subset(empty_like_fashionmnist_train(), range(0, 10))
        train_dataset = torch.utils.data.Subset(fashionmnist_train(), range(0, 10))
        path_pretrained = 'results/intermediate/pretrained_net_fashionmnist'
        path_trained = 'results/intermediate/trained_net_fashionmnist'
    elif dataset == 'multimove':
        pretrain_dataset = empty_multimove()
        train_dataset = pretrain_dataset
        path_pretrained = 'results/intermediate/pretrained_net_multimove'
        path_trained = 'results/intermediate/trained_net_multimove'

    if training_paradigm == 'separate':

        # train(n_runs=1,
        #         n_epochs=7,#7,
        #         learning_rate=200, 
        #         save_path=path_pretrained,
        #         train_dataset=pretrain_dataset,
        #         )

        # train(n_runs=1,
        #     n_epochs=10,
        #     learning_rate=0.02,
        #     load_path=path_pretrained,
        #     save_path=path_trained,
        #     train_dataset=train_dataset,
        #     )
        pass

    elif training_paradigm == 'joint':
        train_dataset = animals_interleaved_with_empty()
        # train(n_runs=1, 
        #     n_epochs=10,
        #     learning_rate=0.01,
        #     train_simultaneously=True,
        #     save_path=path_pretrained,
        #     train_dataset=pretrain_dataset,
        #     )
        
        train(n_runs=1, 
            n_epochs=90, #90, # 85
            learning_rate=0.02, #0.02
            train_simultaneously=True,
            #load_path=path_trained,
            save_path=path_trained,
            train_dataset=train_dataset,
           )
        pass

    recording_for_animation = fig4_segmentation(path_to_model=PATH_TO_TRAINED, #path_trained + '_run-0',
                                                inference_mode='all',
                                                dataset=train_dataset,
                                                V1_channel=V1_channel,
                                                show_idx=show_idx, show_details=show_details)

    # Set up plotting and animations
    if show_details:
        fig2, axs2 = plt.subplots(2, 4)
        # Plot first frame in all subplots and add colorbars
        vis_plot = axs2[0, 0].imshow(recording_for_animation['visual_input'][0], 
                                    vmin=np.min(recording_for_animation['visual_input']),
                                        vmax=np.max(recording_for_animation['visual_input']))

        of_plot = axs2[1, 0].imshow(recording_for_animation['optic_flow'][0],
                                    vmin=-1,#np.min(recording_for_animation['optic_flow']),
                                    vmax=1)#np.max(recording_for_animation['optic_flow']))
        axs2[1, 0].set_title('Optic flow')
        
        motor_pred_pos_plot = axs2[0, 1].imshow(recording_for_animation['motor_pred_pos'][0],
                                                vmin=-1,#np.min(recording_for_animation['motor_pred_pos']),
                                                vmax=1)#np.max(recording_for_animation['motor_pred_pos']))
        axs2[0, 1].set_title('MtrPred Pos')
        motor_pred_neg_plot = axs2[1, 1].imshow(recording_for_animation['motor_pred_neg'][0],
                                                vmin=-1,#np.min(recording_for_animation['motor_pred_neg']),
                                                vmax=1)#np.max(recording_for_animation['motor_pred_neg']))
        axs2[1, 1].set_title('MtrPred Neg')

        vis_pred_pos_plot = axs2[0, 2].imshow(recording_for_animation['vis_pred_pos'][0],
                                            vmin=np.min(recording_for_animation['vis_pred_pos']),
                                            vmax=np.max(recording_for_animation['vis_pred_pos']))
        axs2[0, 2].set_title('VisPred Pos')
        vis_pred_neg_plot = axs2[1, 2].imshow(recording_for_animation['vis_pred_neg'][0],
                                            vmin=np.min(recording_for_animation['vis_pred_neg']),
                                            vmax=np.max(recording_for_animation['vis_pred_neg']))
        axs2[1, 2].set_title('VisPred Neg')
        
        pPE_plot = axs2[0, 3].imshow(recording_for_animation['pPE0'][0],
                                            vmin=np.min(recording_for_animation['pPE0']),
                                            vmax=np.max(recording_for_animation['pPE0']))
        axs2[0, 3].set_title('pPE')
        
        nPE_plot = axs2[1, 3].imshow(recording_for_animation['nPE0'][0],
                                            vmin=np.min(recording_for_animation['nPE0']),
                                            vmax=np.max(recording_for_animation['nPE0']))
        axs2[1, 3].set_title('nPE')

        list_of_axes = axs2.flat
        list_of_subplots = [vis_plot, motor_pred_pos_plot, vis_pred_pos_plot, pPE_plot, of_plot, motor_pred_neg_plot, vis_pred_neg_plot, nPE_plot]
        for ax, subplot in zip(list_of_axes, list_of_subplots):
            fig2.colorbar(subplot, ax=ax, location='bottom', fraction=0.1, aspect=15, anchor=(0.5, 0.5))
        axs2 = remove_ticks(axs2)
        ani2 = animation.FuncAnimation(fig2, update_fig2, interval=1000, frames=60)

    else:
        fig1, axs1 = plt.subplots(1, 2)
        ani1 = animation.FuncAnimation(fig1, update_fig1, frames=60, interval=1000)

    plt.show()