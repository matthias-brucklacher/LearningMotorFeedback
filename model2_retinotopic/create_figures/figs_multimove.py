""" Figure S2: Predicting optic flow for a multimove sequence.

    This script trains a network on a (empty) multimove sequence and then predicts the optic flow for each movement in the sequence.
    The optic flow is predicted by summing the motor predictions for each movement dimension.
    The predicted optic flow is compared to the true optic flow.

"""

from model2_retinotopic.data_handling.multimove_dataset import empty_multimove
from model2_retinotopic.network.network_hierarchical import HierarchicalNetwork
from model2_retinotopic.network.training_configurations import pretrain_multimove
from model2_retinotopic.network.network_paths import network_paths
import matplotlib.pyplot as plt
import pylustrator
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
    pretrain_multimove()
    

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
    for col_it in range(3):
        for i in range(2):
            axs[0 + 2 * i, col_it].set_title(['ABC', 'DEF'][i][col_it], fontsize=16)
            axs[0 + 2 * i, col_it].quiver((true_OF[col_it + i * 3, :, :, 0])[::4, ::4], (true_OF[col_it + i * 3, :, :, 1])[::4, ::4], scale=15)
            axs[1 + 2 * i, col_it].quiver(predicted_OF[col_it + i * 3, :, :, 0][::4, ::4], predicted_OF[col_it + i * 3, :, :, 1][::4, ::4], scale=15)
    plt.subplots_adjust(hspace=.7)

    for ax in axs.flat:
        ax.set(aspect='equal')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()

    #% start: automatic generated code from pylustrator
    plt.figure(1).text(0.1201, 0.7769, 'Original', transform=plt.figure(1).transFigure, fontsize=12., rotation=90.)  # id=plt.figure(1).texts[0].new
    plt.figure(1).text(0.1201, 0.5556, 'Predicted', transform=plt.figure(1).transFigure, fontsize=12., rotation=90.)  # id=plt.figure(1).texts[2].new
    plt.figure(1).text(0.1201, 0.2911, 'Original', transform=plt.figure(1).transFigure, fontsize=12., rotation=90.)  # id=plt.figure(1).texts[3].new
    plt.figure(1).text(0.1201, 0.0675, 'Predicted', transform=plt.figure(1).transFigure, fontsize=12., rotation=90.)  # id=plt.figure(1).texts[4].new
    plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
    getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
    plt.figure(1).texts[0].set(position=(0.1198, 0.7789))
    plt.figure(1).texts[1].set(position=(0.1198, 0.5285))
    plt.figure(1).texts[2].set(position=(0.1198, 0.2987))
    plt.figure(1).texts[3].set(position=(0.1198, 0.0395))
    #% end: automatic generated code from pylustrator

    plt.savefig('results/figures/figS2_multimove.png')

if __name__ == '__main__':
    figs_multimove()
    plt.show()

