import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir('..')

class net:
    def __init__(self, size):
        self.size = size # number of neurons along one dimension
        self.wts = 1.*np.ones((size,size))
        self.wts2 = 0.5*np.ones((size,size))
        self.l_rate = 0.05
        self.motor = 0
        self.visflow = 0
        self.visflow_pred = self.motor*self.wts
        self.err = 0

    def infer_step(self, visflow, motor):
        self.visflow = visflow
        self.motor = motor
        self.visflow_pred = self.motor*self.wts
 
        self.err = np.maximum(0,self.visflow_pred - self.wts2*self.visflow) # following Attinger et al., the signal is subtracted from prediction and not vice versa
        self.err = 1/(1 + np.exp(-self.err))


    def learn_step(self):
        error_baseline = self.wts*0.5-self.wts2*0.5
        error_baseline = 1/(1 + np.exp(-error_baseline))
        #self.wts -= self.l_rate*(self.err -error_baseline )*(self.motor)
        
        self.wts2 += self.l_rate*(self.err -error_baseline )*(self.visflow)
        
        #self.wts2 += self.l_rate*self.err*self.visflow
        
        #self.wts -= self.l_rate*(self.err-error_baseline)*self.motor
        
        #self.wts2 += self.l_rate*(self.err-0.5)*self.visflow

# Load training data
seq_idx = 3 # For now, train only on one sequence
train_data_visual= np.load('in_data/data_visual_condition-train.npy')[seq_idx]
train_data_visflow = np.load('in_data/data_opticflow_condition-train.npy')[seq_idx,...,0]
train_data_motor_ct = np.load('in_data/data_running_ct_condition-train.npy')[seq_idx]
train_data_motor_nt = np.load('in_data/data_running_nt_condition-train.npy')[seq_idx]

# Instatiate networks
size = train_data_visual.shape[-1]
ct_net = net(size=1)
nt_net = net(size=1)
updates=np.zeros(train_data_visual.shape[0])
# Training loop

train_data_visual= np.load('in_data/data_visual_condition-train.npy')[seq_idx]
train_data_visflow = np.load('in_data/data_opticflow_condition-train.npy')[seq_idx,...,0]
train_data_motor_ct = np.load('in_data/data_running_ct_condition-train.npy')[seq_idx]
train_data_motor_nt = np.load('in_data/data_running_nt_condition-train.npy')[seq_idx]
ct_weights = []
nt_weights=[]


# 1-D Data
train_data_visflow = np.random.rand(10000)
train_data_motor_ct = train_data_visflow
train_data_motor_nt = np.random.rand(10000)


for timestep in range(train_data_visflow.shape[0]):
    nt_net.infer_step(train_data_visflow[timestep], train_data_motor_nt[timestep])
    ct_net.infer_step(train_data_visflow[timestep], train_data_motor_ct[timestep])
    ct_net.learn_step()
    nt_net.learn_step()
    ct_weights.append((ct_net.wts2[0,0]))
    nt_weights.append((nt_net.wts2[0,0]))
fontsize = 18

plt.plot(ct_weights, label='CT weights')
plt.plot(nt_weights, label='NT weights')
plt.ylabel('Synaptic weight', fontsize = fontsize)
plt.xlabel('Training timestep', fontsize = fontsize)
plt.legend(frameon=False, prop={'size': fontsize-2})
plt.xticks(fontsize=fontsize - 6, rotation=0)
plt.yticks(fontsize=fontsize - 6, rotation=0)
#plt.imshow(ct_net.wts)
#plt.colorbar()
plt.tight_layout()
plt.savefig('weights.svg')
plt.show()