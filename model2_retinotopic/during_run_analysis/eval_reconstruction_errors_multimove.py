from model2_retinotopic.data_handling.multimove_dataset import empty_multimove
from torch.utils.data import DataLoader

def eval_subsequence(net, sequence, device, start_idx=0, stop_idx=10):
    """Evaluate reconstruction errors for a subsequence.

    Args:
        net (nn.Module): Network to evaluate.
        sequence (tuple): Tuple containing optic flow and motor state tensors.
        device (torch.device): Device to run the network on.
        start_idx (int): Start index of the subsequence.
        stop_idx (int): Stop index of the subsequence.
    
    Returns:
        monitoring_loss (float): Reconstruction error of the subsequence.

    """
    net.reset_activity(batch_size = len(sequence[0]))
    optic_flow, motor_state = sequence[0].to(device), sequence[1].to(device)
    assert stop_idx <= optic_flow.shape[1], 'stop_idx must be smaller than sequence length'
    for frame_it in range(start_idx, start_idx + 10):
        _, _, _, monitoring_loss = net(optic_flow[:, frame_it], motor_state[:, frame_it])
    return monitoring_loss.item()

def eval_multimove(net, device):
    """Evaluate reconstruction errors for individual subsequences of the multimove dataset for supplementary figure.

    The goal is to compare the reconstruction errors of homogeneous and non-homogeneous optic-flows.
    
    Returns:
        reconstruction_errors (dict): Dictionary containing (mean, std) reconstruction errors for linear, forward + turn, and expanding/contracting sequences.
    
    """
    dataset = empty_multimove()
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    data = next(iter(data_loader))
    reconstruction_errors = []
    data = data[0] # Remove labels
    for i in [0, 30, 40]: # The relevant sequences start at these time points
        reconstruction_errors.append(eval_subsequence(net, data, device, start_idx=i, stop_idx=i+10))
    reconstruction_errors = {'linear': reconstruction_errors[0], 'forward + turn': reconstruction_errors[1], 'expanding/contracting': reconstruction_errors[2]}
    return reconstruction_errors