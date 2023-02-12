import numpy as np
import queue
import matplotlib.pyplot as plt

class network:
    def __init__(self, dim_1, dim_2, w1_scaling, w2_scaling):
        self.w1 = w1_scaling #np.random.uniform(0.1,w1_scaling,(dim_2, dim_1))
        self.w2 = w2_scaling #np.random.uniform(0.1,w2_scaling,(dim_2))
        self.act_1 = None
        self.act_2 = None
        self.err = None
        self.err_list = []
        self.err_avg = None
        self.l_rate = 0.1

    def infer_step(self, in_1, in_2):
        # Determine activity of the neurons
        self.act_1 = in_1
        self.act_2 = in_2
        self.err = -self.w1*in_1+ self.w2*in_2 # -np.dot(self.w1,in_1) + np.multiply(self.w2.T,in_2)

        # To get Keller-like responses (otherwise not necessary)
        #self.err = np.maximum(0, self.err) 
        #self.err = 1/(1 + np.exp(-self.err)) 

    def learn_step(self):
        # Calculate moving average of the error neuron 
        sliding_window = 10
        if len(self.err_list) == sliding_window: 
            self.err_list.pop(0)
        self.err_list.append(self.err)
        self.err_avg = np.sum(np.array(self.err_list), axis = 0)/len(self.err_list)

        # Update weights
        #self.err_avg = 0.5
        # self.w1 += self.l_rate * (self.err -self.err_avg)*self.act_1 #self.l_rate * np.dot((self.err -self.err_avg),np.transpose(self.act_1))
        # self.w2 -= self.l_rate * (self.err - self.err_avg)*self.act_2 #self.l_rate * np.dot((self.err -self.err_avg),np.transpose(self.act_2))
        self.w1 += self.l_rate * self.err*self.act_1
        self.w2 -= self.l_rate * self.err*self.act_2
        self.w1 = np.maximum(0,self.w1)
        self.w2 = np.maximum(0,self.w2)

np.random.seed(0)
seq_length = 1000
in_dim = 1
uni_1 = np.random.uniform(0,1,size  = (seq_length,in_dim))
uni_1_noisy = uni_1 + 0.*np.random.standard_normal((seq_length,in_dim))
uni_1_noisy = np.clip(uni_1_noisy, a_min = 0, a_max= 1)

uni_2 = np.random.uniform(0,1,size  = (seq_length, in_dim))

# Multi-dim patters: sequences of one-hot vectors 
multibin_1 = np.array([np.eye(in_dim)[act] for act in np.random.randint(0,in_dim,seq_length) ])
multibin_2 = np.array([np.eye(in_dim)[act] for act in np.random.randint(0,in_dim,seq_length) ])

# longer stretches

atom_1 = [[1]]*50 + [[0]]*50
atom_2 = [[0]]*50 + [[1]]*50

bin_1 = np.array(atom_1*100)
bin_2 = np.array(atom_2* 100)
seq_1 = uni_1
seq_2 = uni_1

metrics = {'w1': [], 'w2': [], 'err': [], 'err_avg': []}

net = network(1, 1, w1_scaling = 0.1, w2_scaling = 1)
metrics['w1'].append(net.w1)
metrics['w2'].append(net.w2)

for i in range(seq_length):
    
    net.infer_step(seq_1[i][0], seq_2[i][0])
    net.learn_step()
    metrics['w1'].append(net.w1)
    metrics['w2'].append(net.w2)
    metrics['err'].append(net.err)
    metrics['err_avg'].append(net.err_avg)
#print(f'Final weights are w1 = {net.w1:.2f} and w2 = {net.w2:.2f}')
print(net.err_avg)
plt.plot(range(seq_length+1), metrics['w1'], label = 'w1')
plt.plot(range(seq_length+1), metrics['w2'], label = 'w2')
#plt.plot(range(seq_length), metrics['err'], label = 'err')
plt.plot(range(seq_length), metrics['err_avg'], label = 'err_avg')
plt.legend()
plt.show()
