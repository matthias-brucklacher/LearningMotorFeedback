import numpy as np
import matplotlib.pyplot as plt

sim_time = 90 # total number of simulation steps
train_paradigm = 'CT' # Either coupled training (CT) or non-coupled (NT) following Attinger et al. 2017

class net:
    def __init__(self, train_paradigm):
        self.train_paradigm = train_paradigm # CT or NT
        if self.train_paradigm=='CT':
            self.wts = 1.
            self.wts2 = 1.
        elif self.train_paradigm=='NT':
            self.wts = 1.0
            self.wts2 = 0.
        self.l_rate = 0.1
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
        self.wts -= self.l_rate*self.err*self.motor

    def test(self, visflow_seq, motor_seq):
        assert len(visflow_seq) == len(motor_seq), "Optic flow sequence of different length than motor sequence"
        timesteps = len(visflow_seq)
        error_list = []
        for timeit in range(timesteps):
            self.infer_step(visflow_seq[timeit], motor_seq[timeit])
            error_list.append(self.err)
        return error_list

def preprocessing(data_list, baseline):
    """
    data_list: list of error neuron responses
    baseline: scalar value, so far calculated by hand
    """
    new_data_list = []
    for valueIt in data_list:
        new_data_list.append(np.absolute(valueIt)-data_list[-1])
    return new_data_list
    

# Prepare training data
ramp_down = np.array([1,0.9,0.8,0.7,0.6,0.6,0.4,0.3,0.2,0.1])
ramp_up = np.flip(ramp_down)
constant_on = np.ones(10)
constant_off = np.zeros(10)
train_data_visual = np.concatenate([constant_off, ramp_up, constant_on, ramp_down, constant_off, ramp_up, constant_on, constant_on, constant_off])
train_data_motor_ct = train_data_visual
train_data_motor_nt = np.concatenate([constant_on, ramp_down, constant_off, constant_off, constant_on, ramp_up, constant_on, constant_on, constant_on])

train_data_motor = {'CT': train_data_motor_ct, 'NT': train_data_motor_nt}

ct_net = net('CT')
nt_net = net('NT')

list = []

# # Training loop
# for timeit in range(sim_time):
#     nt_net.infer_step(train_data_visual[timeit], train_data_motor['NT'][timeit])
#     ct_net.infer_step(train_data_visual[timeit], train_data_motor['CT'][timeit])
#     ct_net.learn_step()
#     nt_net.learn_step()

print('ct_net final weights after initial training '+ str(ct_net.wts))

# Prepare test data
constant_off_short = np.zeros(5)
constant_on_short = np.ones(5)
test_data_visual = np.concatenate([constant_off, ramp_up, constant_on, constant_off_short, constant_on])

test_data_motor_mismatch = np.concatenate([constant_off, ramp_up, constant_on, constant_on_short, constant_on]) # Same as visual, but second-to-last block (mismatch) is 1
test_data_motor_mismatch_2 = np.concatenate([constant_off, ramp_up, constant_off, constant_on_short, constant_off])

test_data_motor_playback_halt = np.zeros(45)
test_data_motor_playback_halt_3 =  np.concatenate([constant_off, constant_off, constant_on, constant_off_short, constant_on])
test_data_motor = {'mismatch': test_data_motor_mismatch, 'playback_halt': test_data_motor_playback_halt}

# Run tests for two conditions per network 
mismatch_motor = test_data_motor_mismatch
playback_motor_standing = test_data_motor_playback_halt
playback_motor_stopping = test_data_motor_playback_halt_3

errors_ct_mismatch = ct_net.test(test_data_visual, mismatch_motor)
errors_nt_mismatch = nt_net.test(test_data_visual, mismatch_motor)
errors_ct_playback_halt_standing = ct_net.test(test_data_visual, playback_motor_standing)
errors_nt_playback_halt_standing = nt_net.test(test_data_visual, playback_motor_standing)
errors_ct_playback_halt_stopping = ct_net.test(test_data_visual, playback_motor_stopping)
errors_nt_playback_halt_stopping = nt_net.test(test_data_visual, playback_motor_stopping)

deltaf_errors_ct_mismatch = preprocessing(errors_ct_mismatch,0)
deltaf_errors_nt_mismatch = preprocessing(errors_nt_mismatch, 0.5)
deltaf_errors_ct_playback_halt_standing = preprocessing(errors_ct_playback_halt_standing, 0)
deltaf_errors_nt_playback_halt_standing = preprocessing(errors_nt_playback_halt_standing, 0.5)
deltaf_errors_ct_playback_halt_stopping = preprocessing(errors_ct_playback_halt_stopping, 0)
deltaf_errors_nt_playback_halt_stopping = preprocessing(errors_nt_playback_halt_stopping, 0.5)

print(errors_ct_mismatch[30:35])

# Plot error neuron response
plot_range_lower = 29
plot_range_upper = 40
time_range = [i/5 for i in range(-1,plot_range_upper-plot_range_lower-1)]

fig, axs = plt.subplots(4, 3, sharey = True)


axs[0,0].plot(time_range, errors_ct_mismatch[plot_range_lower:plot_range_upper], color = 'blue', label = 'CT')
axs[0,0].plot(time_range, errors_nt_mismatch[plot_range_lower:plot_range_upper], color = 'red', label = 'NT')
axs[0,0].set_title('Mismatch')
axs[0,0].legend(frameon = False)
axs[0,0].set_xticklabels([])
axs[0,0].set_ylabel('EN response')

axs[0,1].plot(time_range, errors_ct_playback_halt_standing[plot_range_lower:plot_range_upper], 'b--', label = 'CT')
axs[0,1].plot(time_range, errors_nt_playback_halt_standing[plot_range_lower:plot_range_upper], 'r--', label = 'NT')
axs[0,1].legend(frameon = False)
axs[0,1].set_xticklabels([])
axs[0,1].set_title('Playback halt 1')

axs[0,2].plot(time_range, errors_ct_playback_halt_stopping[plot_range_lower:plot_range_upper], 'b--', label = 'CT')
axs[0,2].plot(time_range, errors_nt_playback_halt_stopping[plot_range_lower:plot_range_upper], 'r--', label = 'NT')
axs[0,2].legend(frameon = False)
axs[0,2].set_xticklabels([])
axs[0,2].set_title('Playback halt 2')

axs[1,0].plot(time_range, deltaf_errors_ct_mismatch[plot_range_lower:plot_range_upper], color = 'blue', label = 'CT mismatch')
axs[1,0].plot(time_range, deltaf_errors_nt_mismatch[plot_range_lower:plot_range_upper], color = 'red', label = 'NT mismatch')
axs[1,0].set_xticklabels([])
axs[1,0].set_ylabel('$\Delta$ F/F')

axs[1,1].plot(time_range, deltaf_errors_ct_playback_halt_standing[plot_range_lower:plot_range_upper], 'b--', label = 'CT')
axs[1,1].plot(time_range, deltaf_errors_nt_playback_halt_standing[plot_range_lower:plot_range_upper], 'r--', label = 'NT')
axs[1,1].set_xticklabels([])

axs[1,2].plot(time_range, deltaf_errors_ct_playback_halt_stopping[plot_range_lower:plot_range_upper], 'b--', label = 'CT')
axs[1,2].plot(time_range, deltaf_errors_nt_playback_halt_stopping[plot_range_lower:plot_range_upper], 'r--', label = 'NT')
axs[1,2].set_xticklabels([])

axs[2,0].plot(time_range, test_data_visual[plot_range_lower:plot_range_upper], 'tab:green')
axs[2,0].set_ylabel('Optic Flow')
axs[2,0].set_xticklabels([])

axs[2,1].plot(time_range, test_data_visual[plot_range_lower:plot_range_upper], 'tab:green')
axs[2,1].set_xticklabels([])

axs[2,2].plot(time_range, test_data_visual[plot_range_lower:plot_range_upper], 'tab:green')
axs[2,2].set_xticklabels([])


axs[3,0].plot(time_range, mismatch_motor[plot_range_lower:plot_range_upper], 'tab:purple')
axs[3,0].set_ylabel('Running')
axs[3,0].set_xlabel('Time [s]')

axs[3,1].plot(time_range, playback_motor_standing[plot_range_lower:plot_range_upper], 'tab:purple')

axs[3,2].plot(time_range, playback_motor_stopping[plot_range_lower:plot_range_upper], 'tab:purple')

for axIt in axs.reshape(-1):
    axIt.spines["top"].set_visible(False)
    axIt.spines["right"].set_visible(False)

#plt.plot(time_range, errors_ct_mismatch[error_range_low:error_range_high], color = 'blue', label = 'CT mismatch')
#plt.plot(time_range, errors_nt_mismatch[error_range_low:error_range_high], color = 'red', label = 'NT mismatch')
#plt.plot(time_range, errors_ct_playback_halt[error_range_low:error_range_high], 'b--', label = 'CT playback halt')
#plt.plot(time_range, errors_nt_playback_halt[error_range_low:error_range_high], 'r--', label = 'NT playback halt')
#plt.xlabel('Time [s]')
#plt.ylabel('Error neuron response')
#plt.legend()
#plt.axvspan(0, 1, color='yellow', alpha=0.5)
plt.show()