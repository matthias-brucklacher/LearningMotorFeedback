import matplotlib.pyplot as plt
from model2_retinotopic.network.network_helpers import OF_to_V1_DS, bipartite_loss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalNetwork(nn.Module):
    def __init__(self, img_width, movement_dim=1):
        """Hierarchical network with 3 layers: V1, MT, MST

        Args:
            img_width (int): width of the input image
            movements_dim (int, optional): number of movement dimensions. Defaults to 1.

        """
        super().__init__()
        # Hyperparameters
        self.I_RATE = 0.05 
        self.I_STEPS = 15
        self.PE_SHIFT = 0.5 # Shift ReLU activation function of PE neurons for baseline activity. 
        self.PE_BASELINE = self.PE_SHIFT # Due to the chosen shift, the baseline activity is 0.5.
        self.segmentation_threshold_factor = 1 / 1.4 

        # Network parameters
        self.img_width = img_width
        self.inference_mode = None  # 'all', 'stream_1'
        self.plasticity_mode = None # 'all', 'stream_1', 'stream_2', 'frozen'
        self.device_param = nn.Parameter(torch.empty(0))

        # Motor area connects downwards to V1
        self.movements_dim = movement_dim
        self.mtr_to_pPE0 = nn.Conv2d(in_channels=4, out_channels=self.movements_dim, stride=1, kernel_size=img_width, bias=False)
        self.mtr_to_pPE0.weight.data.normal_(0.6, 0.6)
        self.mtr_to_pPE0.weight.data = torch.clamp(self.mtr_to_pPE0.weight.data, min=0)

        self.mtr_to_nPE0 = nn.Conv2d(in_channels=4, out_channels=self.movements_dim, stride=1, kernel_size=img_width, bias=False)
        self.mtr_to_nPE0.weight.data.normal_(0.6, 0.6)
        self.mtr_to_nPE0.weight.data = torch.clamp(self.mtr_to_nPE0.weight.data, min=0)

        # Area MT connects downwards to V1
        self.kernel_size_MT = 4
        self.stride_MT = 1
        self.n_channels_MT = 4 # 8
        self.MT_width = (img_width - self.kernel_size_MT + self.stride_MT) // self.stride_MT
        self.MT_to_nPE0 = nn.Conv2d(in_channels=4, out_channels=self.n_channels_MT, stride=self.stride_MT, kernel_size=self.kernel_size_MT, bias=False)
        x = np.sqrt(1 / (self.n_channels_MT * self.kernel_size_MT ** 2))
        self.MT_to_nPE0.weight.data.uniform_(-x, x)
        #self.MT_to_nPE0.weight.data.normal_(0, x)
        # self.MT_to_nPE0.weight.data = torch.clamp(self.MT_to_nPE0.weight.data, min=0)
        self.MT_to_pPE0 = nn.Conv2d(in_channels=4, out_channels=self.n_channels_MT, stride=self.stride_MT, kernel_size=self.kernel_size_MT, bias=False)
        self.MT_to_pPE0.weight.data.uniform_(-x, x)
        #self.MT_to_pPE0.weight.data.normal_(x, x)
        # self.MT_to_pPE0.weight.data = torch.clamp(self.MT_to_pPE0.weight.data, min=0)

        # Area MST connects downwards to MT
        self.kernel_size_MST = 4
        self.stride_MST = 1
        self.n_channels_MST = 4
        self.MST_width = (self.MT_width - self.kernel_size_MST + self.stride_MST) // self.stride_MST
        self.MST_to_MT = nn.Conv2d(in_channels=self.n_channels_MT, out_channels=self.n_channels_MST, stride=self.stride_MST, kernel_size=self.kernel_size_MST, bias=False)

    def set_grad_stream_1(self, grad_enabled):
        self.mtr_to_pPE0.weight.requires_grad = grad_enabled
        self.mtr_to_nPE0.weight.requires_grad = grad_enabled

    def set_grad_stream_2(self, grad_enabled):
        self.MT_to_pPE0.weight.requires_grad = grad_enabled
        self.MT_to_nPE0.weight.requires_grad = grad_enabled
        self.MST_to_MT.weight.requires_grad = grad_enabled
    
    def freeze_weights(self):
        self.set_grad_stream_1(False)
        self.set_grad_stream_2(False)

    def infer_all(self):
        # Set model to inference mode: All weights frozen
        self.inference_mode = 'all'
        self.plasticity_mode = 'frozen'
        self.freeze_weights()

    def infer_stream_1(self):
        # Set model to inference mode: All weights frozen
        self.inference_mode = 'stream_1'
        self.plasticity_mode = 'frozen'
        self.freeze_weights()

    def train_all(self):
        # Simultanously train all weights
        self.inference_mode = 'all'
        self.plasticity_mode = 'all'
        self.set_grad_stream_1(True)
        self.set_grad_stream_2(True)
        self.segmentation_threshold_factor = 1 / 2 # Yields better segmentation in joint paradigm 

    def train_stream_1(self):
        # Set model to pretrain mode: Only motor-to-sensory forward model plastic
        self.inference_mode = 'stream_1'
        self.plasticity_mode = 'stream_1'
        self.set_grad_stream_1(True)
        self.set_grad_stream_2(False)

    def train_stream_2(self):
        # Set model to visual train mode: Only sensory-to-sensory stream plastic
        self.inference_mode = 'all'
        self.plasticity_mode = 'stream_2'
        self.set_grad_stream_1(False)
        self.set_grad_stream_2(True)

    def reset_activity(self, batch_size):
        self.MT_pre = 0.1 * torch.ones((batch_size, self.MT_to_pPE0.out_channels, self.MT_width, self.MT_width), requires_grad=False).to(self.device_param.device)
        self.MST_pre = 0.1 * torch.ones((batch_size, self.MST_to_MT.out_channels, self.MST_width, self.MST_width), requires_grad=False).to(self.device_param.device)
        self.to(self.device_param.device)
    
    def PE_actfun(self, x):
        return torch.relu(x + self.PE_SHIFT)
    
    def MT_actfun(self, x):
        return torch.relu(x)
    
    def MST_actfun(self, x):
        return torch.sigmoid(x)
    
    def clip_weights(self):
        # Make weights non-negative
        self.MT_to_pPE0.weight.data = torch.clamp(self.MT_to_pPE0.weight.data, min=0)
        self.MT_to_nPE0.weight.data = torch.clamp(self.MT_to_nPE0.weight.data, min=0)
    
    def calculate_PE0(self, V1_DS, vis_pred, motor_pred, type):
        """Calculate the prediction error for the input area

        Args:
            V1_DS (torch.Tensor): Input to the area
            vis_pred (torch.Tensor): Prediction from the area MT
            motor_pred (torch.Tensor): Prediction from the motor system
            type (str): Type of prediction error to calculate. Either 'nPE' or 'pPE'

        Returns:
            PE_pre (torch.Tensor): Prediction error before rectification
            PE (torch.Tensor): Rectified prediction error

        """
        assert type in ['nPE', 'pPE'], f'Invalid type provided. Must be "nPE" or "pPE" but was "{type}"'
        if type == 'nPE':
            sign = -1. 
        elif type == 'pPE':
            sign = 1.
        PE_pre = sign * (V1_DS - motor_pred) - vis_pred
        gating = torch.sum(self.mtr, dim=1, keepdim=False).reshape(-1, 1, 1, 1)
        PE_pre = PE_pre * gating # torch.where(torch.sum(self.mtr, dim=1, keepdim=False) != 0, PE_pre, 0) # gating: only calculate PE if motor is active
        PE = self.PE_actfun(PE_pre)
        return PE_pre, PE

    def motor_predictions(self, type):
        """Calculate the prediction from the motor system to the input area.


        """
        if type == 'pos':
            layer = self.mtr_to_pPE0
        elif type == 'neg':
            layer = self.mtr_to_nPE0
        motor_pred = F.conv_transpose2d(self.mtr.unsqueeze(2).unsqueeze(3), layer.weight, stride=1) # Unsqueeze: (B, M) -> (B, M, 1, 1) necesarry for conv_transpose2d  
        return motor_pred
    
    def predictions_from_MT(self, visual_prediction_relay, MT, type):
        if type == 'neg':
            return visual_prediction_relay * F.conv_transpose2d(MT, self.MT_to_nPE0.weight, stride=self.stride_MT)
        elif type == 'pos':
            return visual_prediction_relay * F.conv_transpose2d(MT, self.MT_to_pPE0.weight, stride=self.stride_MT)
        else:
            raise ValueError(f'Invalid type provided. Must be "neg" or "pos" but was "{type}"')
    
    def forward(self, OF_input, mov_state): 
        """Inference on a single input frame
        """
        assert self.inference_mode is not None, 'inference_mode must be set before training loop'
        assert torch.numel(self.MT_pre) != 0, 'Network not initialized. Use net.reset_activity.'
        V1_DS = OF_to_V1_DS(OF_input)

        # Predictions from motor system to V1
        self.mtr = mov_state.detach() # (B, M)
        motor_pred_neg = self.motor_predictions(type='neg')
        motor_pred_pos = self.motor_predictions(type='pos')

        for i in range(self.I_STEPS):
            with torch.set_grad_enabled(i == self.I_STEPS - 1):
                # Predictions of MT to V1
                visual_prediction_relay = int(self.inference_mode in ['all', 'stream_2']) # Predictions from stream_2 (MT/MST)
                MT = self.MT_actfun(self.MT_pre)
                vis_pred_neg = self.predictions_from_MT(visual_prediction_relay, MT, type='neg')
                vis_pred_pos = self.predictions_from_MT(visual_prediction_relay, MT, type='pos')

                # Calculate errors in V1
                nPE0_pre, nPE0 = self.calculate_PE0(V1_DS, vis_pred_neg, motor_pred_neg, type='nPE')
                pPE0_pre, pPE0 = self.calculate_PE0(V1_DS, vis_pred_pos, motor_pred_pos, type='pPE')

                # Update activity of areas in MT and MST ('stream_2')
                if self.inference_mode != 'stream_1':
                    self.MT_pre = self.MT_pre + self.I_RATE / 2 * (self.MT_to_pPE0(pPE0 - self.PE_BASELINE) + self.MT_to_nPE0(nPE0 - self.PE_BASELINE)).detach() # Detach prevents torch from building a computational graph for this attribute

                    # Predict to MT     
                    MST = self.MST_actfun(self.MST_pre)          
                    vis_pred_1 = visual_prediction_relay * F.conv_transpose2d(MST, self.MST_to_MT.weight, stride=self.stride_MST)

                    # Calculate errors in MT
                    MT_PE = self.MT_pre - vis_pred_1

                    # Update activity of MST 
                    self.MST_pre = self.MST_pre + self.I_RATE * self.MST_to_MT(MT_PE).detach()
                else:
                    MT_PE = 0
        training_loss, monitoring_loss = self.loss(nPE0_pre, nPE0, pPE0_pre, pPE0, MT_PE)

        return nPE0_pre, pPE0_pre, training_loss, monitoring_loss
    
    def loss(self, nPE0_pre, nPE0, pPE0_pre, pPE0, PE1):
        """Calculate the training and monitoring loss.

        """
        assert self.plasticity_mode is not None, 'plasticity_mode must be set'
        criterion = nn.MSELoss()
        PE0_target = torch.zeros_like(nPE0).to(self.device_param.device)
        input_reconstruction_loss_pre = criterion(nPE0_pre, PE0_target) + criterion(pPE0_pre, PE0_target)
        input_reconstruction_loss_post = criterion(pPE0, PE0_target) + criterion(nPE0, PE0_target)
        input_reconstruction_loss_bipartite = bipartite_loss(nPE0_pre=nPE0_pre, pPE0_pre=pPE0_pre, device=self.device_param.device)
        if self.plasticity_mode == 'stream_1':
            training_loss = input_reconstruction_loss_bipartite
            monitoring_loss = input_reconstruction_loss_pre
        elif self.plasticity_mode in ['stream_2', 'all']:
            MT_reconstruction_loss = criterion(PE1, torch.zeros_like(PE1).to(self.device_param.device)) # Should also be trainable with bipartite loss
            training_loss = input_reconstruction_loss_bipartite  + MT_reconstruction_loss 
            monitoring_loss = input_reconstruction_loss_pre
        elif self.plasticity_mode == 'frozen':
            training_loss = 0
            monitoring_loss = input_reconstruction_loss_post
        else:
            raise ValueError(f'Invalid plasticity mode provided. Must be "stream_1", "stream_2" but was "{self.plasticity_mode}"')
        return training_loss, monitoring_loss

    def segment(self):
        """Return mask where the optic flow is not self-generated.

        Returns:
            predicted_area (numpy.ndarray): Binary mask, 1 where non-self generated. Shape is (batch_size, img_height, img_width)

        """
        vis_pred_pos = self.predictions_from_MT(MT=self.MT_actfun(self.MT_pre), visual_prediction_relay=True, type='pos') # (B, 4, H, W)
        vis_pred_neg = self.predictions_from_MT(MT=self.MT_actfun(self.MT_pre), visual_prediction_relay=True, type='neg') # (B, 4, H, W)

        # Keep only positive predictions
        vis_pred_pos = torch.where(vis_pred_pos > 0, vis_pred_pos, torch.zeros_like(vis_pred_pos))
        vis_pred_neg = torch.where(vis_pred_neg > 0, vis_pred_neg, torch.zeros_like(vis_pred_neg))

        # Add channels up
        vis_pred_pos = torch.sum(vis_pred_pos, dim=1, keepdim=True)
        vis_pred_neg = torch.sum(vis_pred_neg, dim=1, keepdim=True)

        # Mark with 1 where either positive or negative prediction is above threshold
        segmentation_threshold = torch.amax(vis_pred_pos + vis_pred_neg, dim=(2, 3), keepdim=True) * self.segmentation_threshold_factor
        predicted_area = torch.where(vis_pred_pos + vis_pred_neg > segmentation_threshold, 1, 0).cpu().numpy()
        predicted_area = predicted_area[:, 0, :, :] # (B, 1, H, W) -> (B, H, W)

        return predicted_area

    