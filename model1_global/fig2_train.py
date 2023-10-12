import numpy as np
import matplotlib.pyplot as plt
from model1_global.global_model import GlobalNet as GlobalNet

def get_data(data_type='binary'):
    # 1-D Data
    data_len = 5000
    if data_type == 'binary':
        optic_flow = np.random.randint(2, size=data_len) 
        motor_nt = np.random.randint(2, size=data_len)
    elif data_type == 'binary_negative':
        optic_flow = -np.random.randint(2, size=data_len) 
        motor_nt = np.random.randint(2, size=data_len)
    elif data_type == 'continuous':
        optic_flow = np.random.rand(data_len)
        motor_nt = np.random.rand(data_len) 
    motor_ct = np.abs(optic_flow)
    return optic_flow, motor_ct, motor_nt

def get_trained_nets_ct_nt(fixed_seed=True, data_type='binary', do_plot=False):
    """Produces the trained networks for the CT and NT paradigm.

    Args:
        fixed_seed (bool, optional): Whether to use a fixed seed for the random number generator. Defaults to True.
        data_type (str, optional): Type of data to use for training. Defaults to 'binary'.
        do_plot (bool, optional): Whether to plot the results. Defaults to False.
    
    Returns:
        ct_net (GlobalNet): Trained network for the CT paradigm.
        nt_net (GlobalNet): Trained network for the NT paradigm.    
    """
    if fixed_seed:
        np.random.seed(0)

    optic_flow, motor_ct, motor_nt = get_data(data_type)

    # Instatiate networks
    #init_weights = {'mtr_to_ppe': 0.7, 'mtr_to_npe': 0.1}
    init_weights = {'mtr_to_ppe': 0.2, 'mtr_to_npe': 0.6}
    ct_net = GlobalNet(init_weights=init_weights)
    nt_net = GlobalNet(init_weights=init_weights)

    # Train networks for a single epoch (suffices with this long data)
    ct_weights_pos, ct_weights_neg = ct_net.training_epoch(optic_flow, motor_ct)
    nt_weights_pos, nt_weights_neg = nt_net.training_epoch(optic_flow, motor_nt)

    # Plotting
    if do_plot==True:
        fontsize = 18
        plt.plot(ct_weights_pos, label='$w_{+}$ (CT)', c='#1f77b4')
        plt.plot(ct_weights_neg, label='$w_{-}$ (CT)', c='#ff7f0e')
        plt.plot(nt_weights_pos, linestyle=(0, (1, 5)), label='$w_{+}$ (NT)', c='#1f77b4')
        plt.plot(nt_weights_neg, linestyle=(0, (1, 5)),  label='$w_{-}$ (NT)', c='#ff7f0e')
        plt.ylabel('Synaptic weight', fontsize=fontsize)
        plt.xlabel('Training timestep', fontsize=fontsize)
        plt.legend(frameon=False, prop={'size': fontsize - 2})
        plt.xticks(fontsize=fontsize - 6, rotation=0)
        plt.yticks(fontsize=fontsize - 6, rotation=0)
        plt.ylim([0, 1.1])
        plt.tight_layout()
        plt.savefig('results/figures/weights.png')
    return ct_net, nt_net

if __name__ == '__main__':
    get_trained_nets_ct_nt(do_plot=True)
    plt.show()