import matplotlib.pyplot as plt
import numpy as np

w_minus = np.linspace(start=-10, stop=10, num=1000)
l_rate = 0.01
e0 = 0.5
y_mtr = 1.0
alpha = 1.0
y_vis = alpha * y_mtr # assuming a coupled training paradigm
w_optim = alpha # optimal weight leading to net zero input to error neuron
e_minus = np.maximum(0, -(y_vis - w_minus * y_mtr) + e0) # Firing rate of error neuron
w_update = - l_rate * y_mtr * (e_minus - e0)
w_thresh = (y_vis - e0) / y_mtr # at fixed motor and visual state, the error neuron will fire if w_minus > w_thresh.

# Integrate w_update to get loss. 
# Strictly correct would be to divide by the learning rate, but this is not necessary for the plot.
loss = [-np.trapz(w_update[:idx], w_minus[:idx]) for idx in range(len(w_minus))] 
loss = np.array(loss) 

# Plotting
plt.figure(1)
plt.plot(w_minus, w_update, label='weight update')
plt.plot(w_minus, loss, label='loss')
plt.axvline(x=w_optim, color='k', linestyle='--', label='optimal weight')
plt.axvline(x=w_thresh, color='k', linestyle=':', label='linear/ quadratic regime')
plt.axhline(y=0, color='k')
plt.xlabel('$w_{-}$')
plt.ylabel('numeric value')
plt.legend(frameon=False)
plt.savefig('results/figures/figs_convergence.png')
plt.show()





