from unittest import runner
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

data_type = 'test'
visual_data = np.load('in_data/data_visual_condition-'+data_type+'.npy')
running_data = np.load('in_data/data_running_ct_condition-'+data_type+'.npy')
seq_idx = 1 # which sequence to process

num_seqs, sequence_length, img_height, img_width = visual_data.shape[0], visual_data.shape[1], visual_data.shape[2], visual_data.shape[3]

flow = np.zeros((num_seqs, sequence_length, img_height, img_width,2))
for i in range(num_seqs):
    for j in range(sequence_length-1):
        flow[i,j] = cv2.calcOpticalFlowFarneback(visual_data[i,j], visual_data[i,j+1], None, 0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)#[:,:,1]

# Save horizontal optic flow
np.save('in_data/data_opticflow_condition-'+data_type+'.npy', flow)

# Display animated movie for selected sequence
flow_data_to_plot = flow[seq_idx,...,0]
fig, axs = plt.subplots(2, figsize=(4, 4))
frame = 0
im0 = axs[0].imshow(flow_data_to_plot[frame], origin='lower', vmin= 0, vmax = 1)
plt.colorbar(im0,ax=axs[0],shrink=0.5, label = 'OF magnitude')
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[0].set_ylabel('Optic Flow')
axs[1].set_xlim([0,running_data.shape[-1]])
axs[1].set_ylim([-1.1,1.1])
axs[1].set_ylabel('Running speed')
axs[1].set_xlabel('Time')
x = []
y = []

def update(*args):
    global frame
    im0.set_array(flow_data_to_plot[frame])
    x.append(frame)
    y.append(running_data[seq_idx,frame])
    axs[1].clear()
    axs[1].plot(x,y)
    axs[1].set_xlim([0,running_data.shape[-1]])
    axs[1].set_ylim([-1.1,1.1])
    axs[1].set_ylabel('Running speed')
    axs[1].set_xlabel('Time')
    frame += 1
    frame %= 100
    return 
plt.tight_layout()
ani = animation.FuncAnimation(fig, update, interval=100, repeat = False)
f = r"C:/Users/mbruckl/OneDrive/OneDrive - UvA/Desktop/animation.mp4" 
writervideo = animation.FFMpegWriter(fps=20) 
ani.save(f, writer=writervideo)
plt.show()