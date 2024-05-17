""" Figure S2: Predicting optic flow for a multimove sequence.

    This script trains a network on a (empty) multimove sequence and then predicts the optic flow for each movement in the sequence.
    The optic flow is predicted by summing the motor predictions for each movement dimension.
    The predicted optic flow is compared to the true optic flow.

    Before running, (1) redefine the recording in train.py as
        recording = Recording('timestamp', 'error', 'IoU', 'linear', 'forward + turn', 'expanding/contracting')

    and (2) replace the recording update in train.py with these lines:
        multimove_errors = eval_multimove(net, device)
        recording.update(error=running_loss, timestamp=epoch_it + 1, IoU=IoU, **multimove_errors)

"""

from model2_retinotopic.data_handling.multimove_dataset import empty_multimove
from model2_retinotopic.network.network_hierarchical import HierarchicalNetwork
from model2_retinotopic.network.training_configurations import pretrain_multimove
from model2_retinotopic.network.network_paths import network_paths
from model2_retinotopic.create_figures.figure_helpers import plot_errorband
import matplotlib.pyplot as plt
#import pylustrator
import numpy as np
import torch

def optic_flow_prediction(net):
    """ Predicts optic flow for a given network.

    In each direction (+x, -x, +y, -y), the optic flow is predicted by summing the motor predictions for the positive and negative movements.
    Then, optic flow in 2D (x/y) is computed by subtracting opposing directions.

    Args:
        net (nn.Module): Network to predict optic flow for.

    Returns:
        predicted_OF (np.array): Predicted optic flow of shape (batch_dim, img_width, img_width, 2)

    """
    motor_pred_horizontal = net.motor_predictions(type='pos')[:, 0] + net.motor_predictions(type='neg')[:, 0] \
        - net.motor_predictions(type='pos')[:, 1] - net.motor_predictions(type='neg')[:, 1]
    motor_pred_vertical = net.motor_predictions(type='pos')[:, 2] + net.motor_predictions(type='neg')[:, 2] \
        - net.motor_predictions(type='pos')[:, 3] - net.motor_predictions(type='neg')[:, 3]
    predicted_OF = torch.stack([motor_pred_horizontal, motor_pred_vertical], dim=-1)
    predicted_OF = predicted_OF / 2 # Take average of positive and negative motor predictions.
    return predicted_OF

def figs_multimove():
    #pylustrator.start()

    # Pretrain network (we only need the motor predictions here)
    recording = pretrain_multimove()
    
    # Plot learning curve
    with plt.style.context('model2_retinotopic.create_figures.plotting_style'):
        fig1, ax1 = plt.subplots()
        epochs = recording['timestamp'][1:, 0]
        plot_errorband(ax1, epochs, recording['expanding/contracting'][:, 0], recording['expanding/contracting'][:, 1], label='expanding/contracting')
        plot_errorband(ax1, epochs, recording['forward + turn'][:, 0], recording['forward + turn'][:, 1], label='forward + turn')
        plot_errorband(ax1, epochs, recording['linear'][:, 0], recording['linear'][:, 1], label='linear')
        plt.xlabel('Epochs')
        plt.xlim(0.5, 10.5)
        plt.ylabel('Prediction error (MSE)')
        plt.legend()
        plt.savefig('results/figures/figS_learning_curve_multimove.png', dpi=600, bbox_inches='tight')

    # Create dataset for evaluation
    dataset = empty_multimove()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Load trained model
    path_trained = network_paths['pretrained_multimove']
    img_width = 40
    movements_dim = 6 # Fixed for multimove dataset
    net = HierarchicalNetwork(img_width=img_width, movement_dim=movements_dim)
    net.load_state_dict(torch.load(path_trained + '_run-0'))

    # Go through dataset and record predicted and true optic flow
    true_OF = np.zeros((movements_dim, img_width, img_width, 2), dtype=np.float32)
    predicted_OF = np.zeros_like(true_OF)
    net.infer_stream_1()
    net.reset_activity(batch_size=len(dataset.opticflow))
    data = next(iter(data_loader))
    data = data[0] # Remove labels
    optic_flow, movement, _, _ = data # Dataset contains only one sequence
    for frame_it in range(optic_flow.shape[1]):
        _, _, _, _ = net(optic_flow[:, frame_it], movement[:, frame_it])
        if (frame_it) % 10 == 0: # Every 10th frame, a new movement is started. Record OF for this movement.
            true_OF[frame_it // 10] = optic_flow[:, frame_it]
            predicted_OF[frame_it // 10] = optic_flow_prediction(net)[0]

    # Plot optic flow
    fig, axs = plt.subplots(4, 3)
    
    for row_it in range(2):
        for col_it in range(3):
            axs[0 + 2 * row_it, col_it].set_title(['ABC', 'DEF'][row_it][col_it], fontsize=16)
            d = 6
            scale = 10
            if col_it == 0 and row_it == 1:
                scale = 7 # Draw larger arrows here for visibility
            axs[0 + 2 * row_it, col_it].quiver((true_OF[col_it + row_it * 3, :, :, 0])[::d, ::d], (true_OF[col_it + row_it * 3, :, :, 1])[::d, ::d], scale=scale, width=0.01)
            axs[1 + 2 * row_it, col_it].quiver(predicted_OF[col_it + row_it * 3, :, :, 0][::d, ::d], predicted_OF[col_it + row_it * 3, :, :, 1][::d, ::d], scale=scale, width=0.01)

    plt.subplots_adjust(hspace=.7)

    for ax in axs.flat:
        ax.set(aspect='equal')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()

    #% start: automatic generated code from pylustrator
    plt.figure(2).text(0.1201, 0.7769, 'Original', transform=plt.figure(2).transFigure, fontsize=12., rotation=90.)  # id=plt.figure(2).texts[0].new
    plt.figure(2).text(0.1201, 0.5556, 'Predicted', transform=plt.figure(2).transFigure, fontsize=12., rotation=90.)  # id=plt.figure(2).texts[2].new
    plt.figure(2).text(0.1201, 0.2911, 'Original', transform=plt.figure(2).transFigure, fontsize=12., rotation=90.)  # id=plt.figure(2).texts[3].new
    plt.figure(2).text(0.1201, 0.0675, 'Predicted', transform=plt.figure(2).transFigure, fontsize=12., rotation=90.)  # id=plt.figure(2).texts[4].new
    plt.figure(2).ax_dict = {ax.get_label(): ax for ax in plt.figure(2).axes}
    getattr(plt.figure(2), '_pylustrator_init', lambda: ...)()
    plt.figure(2).texts[0].set(position=(0.1198, 0.7789))
    plt.figure(2).texts[1].set(position=(0.1198, 0.5285))
    plt.figure(2).texts[2].set(position=(0.1198, 0.2987))
    plt.figure(2).texts[3].set(position=(0.1198, 0.0395))
    #% end: automatic generated code from pylustrator

    plt.savefig('results/figures/figS2_multimove.png', dpi=600, bbox_inches='tight')

if __name__ == '__main__':
    figs_multimove()
    plt.show()

