import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# control flags
data_type = 'test' # Create 'train' or 'test' data

# Define frame size, split into 16 tiles to facilitate location detection
img_tile_width = 20
img_tile_height = img_tile_width
img_width = 4*img_tile_width
img_height = 4*img_tile_height

sequence_length = 100 
num_seqs = 10 # number of different sequences

# Running signals consisting of random blocks of binary states: CT and NT
blocksize = 10
running_ct = np.zeros((num_seqs, sequence_length))
running_nt = np.zeros_like(running_ct)
assert sequence_length%blocksize == 0, 'Sequence length must be a multiple of 10'
for i in range(int(sequence_length/blocksize)):
    a = np.random.binomial(1,0.5,size = (10,1))
    b = np.random.binomial(1,0.5,size = (10,1))
    running_ct[:,i*blocksize:i*blocksize+blocksize] = a
    running_nt[:,i*blocksize:i*blocksize+blocksize] = b

# Create static random binary background pattern for each sequence
background_density = 0.2 # fraction of active pixels
#a = np.random.binomial(1, background_density, size = (num_seqs, img_height, img_width + sequence_length-1))
a = np.random.rand(num_seqs, img_height, img_width + sequence_length-1)*255

# Create frames by shifting background pattern
visual_input = np.zeros((num_seqs, sequence_length, img_height, img_width))

for i in range(num_seqs):
    lateral_shift = 0
    for j in range(sequence_length):
        visual_input[i,j] = np.roll(a[i], lateral_shift, axis=1)[:,0:img_width]
        if running_ct[i,j] == 1:
            lateral_shift+=1

# Create one object per sequence and place in a tile
if data_type == 'test': 
    #object_density = 0.95 # fraction of active pixels
    #objects = np.random.binomial(1,object_density, size = (num_seqs, img_tile_height, img_tile_width))
    objects = np.random.rand(num_seqs, img_tile_height, img_tile_width)*255
    objects = np.repeat(objects[:,np.newaxis,:,:], sequence_length, axis=1)
    object_location = np.random.randint(4, size = (num_seqs,2)) # for each sequence show object in random tile

    for i in range(num_seqs):
        x_offset = object_location[i,0]*img_tile_width
        y_offset = object_location[i,1]*img_tile_height
        visual_input[i, :, y_offset:y_offset+img_tile_height, x_offset:x_offset+img_tile_width] = np.where(objects[i] != 0, objects[i], visual_input[i, :, y_offset:y_offset+img_tile_height, x_offset:x_offset+img_tile_width])

# Save data
np.save('../in_data/data_visual_condition-'+data_type+'.npy', visual_input)
np.save('../in_data/data_running_ct_condition-'+data_type+'.npy', running_ct)
np.save('../in_data/data_running_nt_condition-'+data_type+'.npy', running_nt)

# Display animated movie for selected sequence
seq_idx = 1
visual_data_to_plot = visual_input[seq_idx]
running_data = running_ct
fig, axs = plt.subplots(2, figsize=(4, 4))
frame = 0
im0 = axs[0].imshow(visual_data_to_plot[frame], origin='lower', vmin= 0, vmax = 256)
plt.colorbar(im0,ax=axs[0],shrink=0.5, label = 'Neuronal activity')
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[0].set_ylabel('Visual input')
axs[1].set_xlim([0,running_data.shape[-1]])
axs[1].set_ylim([-1.1,1.1])
axs[1].set_ylabel('Running speed')
axs[1].set_xlabel('Time')
x = []
y = []

def update(*args):
    global frame
    im0.set_array(visual_data_to_plot[frame])
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
    return im0, 

plt.tight_layout()

ani = animation.FuncAnimation(fig, update, interval=200, repeat = False)
f = r"C:\Users\mbruckl\OneDrive - UvA\Documents\animation.mp4" 
writervideo = animation.PillowWriter(fps=20) 
ani.save('animation.gif', writer=writervideo)
plt.show()

#C:\Users\mbruckl\OneDrive - UvA\Documents\LearningMotorFeedback

