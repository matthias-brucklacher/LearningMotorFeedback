
import numpy as np
import matplotlib.pyplot as plt

class GlobalNet:
    def __init__(self, motor_dim=1, init_weights={'mtr_to_ppe': 0.7, 'mtr_to_npe': 0.1}):
        """Network class

        Args:
            motor_dim (int, optional): Dimensionality of motor state. Defaults to 1.
        """
        self.size = motor_dim 
        self.l_rate = 0.01
        self.e0 = 0.5 # baseline activity of error neurons

        # Initialize weights
        self.mtr_to_ppe = init_weights['mtr_to_ppe'] * np.ones((motor_dim, motor_dim))
        self.mtr_to_npe = init_weights['mtr_to_npe'] * np.ones((motor_dim, motor_dim))
        self.vis_wt = 1.0 * np.ones((motor_dim, motor_dim)) # The weight of the visual input to the PE is assumed to be fixed.
        
        # Initialize state variables
        self.y_mtr = 0
        self.y_vis = 0
        self.nPE = 0
        self.pPE = 0

        # Initialize prediction variables
        self.vis_pred_pos = self.y_mtr * self.mtr_to_ppe
        self.vis_pred_neg = self.y_mtr * self.mtr_to_npe

    def actfun(self, x):
        # Shifted ReLU activation function.
        return np.maximum(0, x + self.e0)
    
    def compute_errors(self, prediction, optic_flow, type):
        """Compute error signals

        Args:
            prediction (float): Predicted optic flow
            optic_flow (float): Actual optic flow
            type (str): Either 'nPE' or 'pPE'

        Returns:
            float: Error signal
        """
        gating = (self.y_mtr != 0)
        sign = 1 if type == 'pPE' else -1 # inverse exc./inh. wiring of nPE and pPE
        net_in = gating * sign * (self.vis_wt * optic_flow - prediction) 
        err = self.actfun(net_in)
        return err

    def infer_step(self, optic_flow, movement):
        """Inference step to compute error signals in the network based on present optic flow and motor command.

        Args:
            optic_flow (float): Optic flow
            movement (float): Motor command
        """
        self.y_vis = optic_flow
        self.y_mtr = movement
        self.vis_pred_pos = self.y_mtr * self.mtr_to_ppe
        self.vis_pred_neg = self.y_mtr * self.mtr_to_npe
        self.nPE = self.compute_errors(self.vis_pred_neg, self.y_vis, type='nPE') 
        self.pPE = self.compute_errors(self.vis_pred_pos, self.y_vis, type='pPE')

    def learn_step(self):
        """Learning step to update weights based on error signals in the network.
        
        """
        self.mtr_to_ppe += self.l_rate * (self.pPE - self.e0) * self.y_mtr
        self.mtr_to_npe -= self.l_rate * (self.nPE - self.e0) * self.y_mtr

    def training_epoch(self, optic_flow, motor):
        """Runs a training epoch on the network.

        Args:
            optic_flow (list): List of optic flow values.
            motor (list): List of motor values. 

        Returns:
            weights_pos (list): List of positive weights.
            weights_neg (list): List of negative weights.
        """
        weights_pos, weights_neg = [], []
        for timestep in range(len(optic_flow)):
            self.infer_step(optic_flow[timestep], motor[timestep])
            self.learn_step()
            weights_neg.append((self.mtr_to_npe[0, 0]))
            weights_pos.append((self.mtr_to_ppe[0, 0]))
        return weights_pos, weights_neg