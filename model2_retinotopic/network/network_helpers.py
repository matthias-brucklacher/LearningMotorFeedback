import torch
import torch.nn as nn

def OF_to_V1_DS(OF_input):
    # convert 2-channel OF to 4-channel OF split by direction
    OF_horizontal = OF_input[:, :, :, 0]
    OF_horizontal_dir0 = torch.where(OF_horizontal > 0, OF_horizontal, 0)
    OF_horizontal_dir1 = torch.where(OF_horizontal < 0, -OF_horizontal, 0)
    OF_vertical = OF_input[:, :, :, 1]
    OF_vertical_dir0 = torch.where(OF_vertical > 0, OF_vertical, 0)
    OF_vertical_dir1 = torch.where(OF_vertical < 0, -OF_vertical, 0)    
    OF_input = torch.stack([OF_horizontal_dir0, OF_horizontal_dir1, OF_vertical_dir0, OF_vertical_dir1], dim=1)
    return OF_input

def bipartite_loss(nPE0_pre, pPE0_pre, device):
    # Loss with linear and quadratic components as derived in the paper
    quadratic_criterion = nn.MSELoss()
    linear_criterion = nn.L1Loss()

    # Keep only positive and negative parts of nPE0_pre and pPE0_pre using relu
    nPE0_pre_active = torch.relu(nPE0_pre) # Above threshold, i.e. in sloped region of actfun
    pPE0_pre_active = torch.relu(pPE0_pre)
    nPE0_pre_inactive = -torch.relu(-nPE0_pre) # In flat region of actfun
    pPE0_pre_inactive = -torch.relu(-pPE0_pre)

    PE_pre_target = torch.zeros_like(nPE0_pre).to(device)
    quadratic_loss = quadratic_criterion(nPE0_pre_active, PE_pre_target) + quadratic_criterion(pPE0_pre_active, PE_pre_target)
    linear_loss = linear_criterion(nPE0_pre_inactive, PE_pre_target) + linear_criterion(pPE0_pre_inactive, PE_pre_target)
    return quadratic_loss + linear_loss
    