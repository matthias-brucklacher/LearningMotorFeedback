import matplotlib.pyplot as plt
from model2_retinotopic.network.network_hierarchical import HierarchicalNetwork
from model2_retinotopic.during_run_analysis.analysis_functions import Recording
from model2_retinotopic.during_run_analysis.segmentation import segmentation_performance
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as T

def train(n_runs, train_dataset, n_epochs, learning_rate, test_dataset=None, train_simultaneously=False, run_id='', load_path=None, save_path=None, eval_frame=9):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    recording = Recording('timestamp', 'error', 'IoU')

    # Multiple runs for statistical analysis
    for run_it in range(n_runs):
        torch.manual_seed(run_it)
        recording.add_run()

        # Training data
        train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

        # Load pretrained network if desired
        img_width = next(enumerate(train_loader))[1][0][0].shape[-2]
        movements_dim = next(enumerate(train_loader))[1][0][1].shape[-1]
        net = HierarchicalNetwork(img_width=img_width, movement_dim=movements_dim)

        if load_path is None and train_simultaneously:
            print('\nTraining simultaneously from scratch.')
            net.train_all()
        elif load_path is None and train_simultaneously == False:
            print('\nTraining motor-to-visual pathway from scratch.')
            net.train_stream_1()
        elif load_path is not None and train_simultaneously == False:
            print('\nPretrained network loaded. Training visual-to-visual pathway.')
            net.load_state_dict(torch.load(load_path + f'_run-{run_it}'))
            net.train_stream_2()
        elif load_path is not None and train_simultaneously == True:
            print('\nPretrained network loaded. Training both streams simultaneously.')
            net.load_state_dict(torch.load(load_path + f'_run-{run_it}'))
            net.train_all()

        net.to(device)   

        # Set up loss and optimizer
        optimizer = optim.SGD(net.parameters(), lr=learning_rate)

        # Training loop
        IoU = segmentation_performance(net, train_dataset, eval_frame=eval_frame, run_id=run_id) 
        if run_it == 0:
            run_id += f'_run-{run_it}'
        else:
            run_id = run_id[:-1] + f'{run_it}'
        
        #print(f'Before training: IoU {IoU:.2f}')
        recording.update(timestamp=0, error=0, IoU=IoU) # No error measured before training 
        if test_dataset is None:
            test_dataset = train_dataset
        recording = training_loop(run_id=run_id, n_epochs=n_epochs, train_loader=train_loader, test_dataset=test_dataset, recording=recording, train_simultaneously=train_simultaneously, net=net, optimizer=optimizer, device=device, eval_frame=eval_frame)
        
        # Save weights
        if save_path is not None:
            torch.save(net.state_dict(), save_path + f'_run-{run_it}')

    recording_mean_std = recording.compute_mean_std()
    return recording_mean_std

def training_loop(n_epochs, train_loader, test_dataset, recording, train_simultaneously, net, optimizer, device, run_id='', eval_frame=9):

    optimizer_stream_1 = optim.SGD([net.mtr_to_pPE0.weight, net.mtr_to_nPE0.weight], lr=4000) # Separate optimizer for motor-to-visual pathway in joint training
    assert net.inference_mode is not None, 'inference_mode must be set before training loop'
    assert net.plasticity_mode is not None, 'plasticity_mode must be set before training loop'
        
    for epoch_it in range(n_epochs):
        running_loss = 0
        for i, data in enumerate(train_loader, 0):
            data = data[0] # Remove labels
            net.reset_activity(batch_size = len(data[0]))
            optic_flow, motor_state = data[0].to(device), data[1].to(device)
            sequence_length = optic_flow.shape[1]
            for frame_it in range(sequence_length):
                _, _, training_loss, monitoring_loss = net(optic_flow[:, frame_it], motor_state[:, frame_it])
                optimizer.zero_grad()
                optimizer_stream_1.zero_grad()
                training_loss.backward()
                optimizer.step()
                if train_simultaneously: 
                    optimizer_stream_1.step()
                running_loss += monitoring_loss.item()
        if net.inference_mode != 'stream_1': # Stream 2 needs to be active for segmentation 
            IoU = segmentation_performance(net, test_dataset, eval_frame=eval_frame, run_id=run_id)
            #print(f'Epoch {epoch_it} IoU {IoU:.2f}')
        else:
            IoU = 0
        recording.update(error=running_loss, timestamp=epoch_it + 1, IoU=IoU)
        print(f'Epoch {epoch_it} running loss {running_loss:.3f}')
    return recording