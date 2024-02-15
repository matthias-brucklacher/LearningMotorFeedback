from copy import deepcopy
from model2_retinotopic.network.network_hierarchical import HierarchicalNetwork
from model2_retinotopic.network.training_configurations import pretrain_fashionmnist, train_fashionmnist
from model2_retinotopic.network.network_paths import network_paths
from model2_retinotopic.data_handling.fashionmnist_datasets import fashionmnist_train, fashionmnist_test
from model2_retinotopic.during_run_analysis.analysis_functions import Recording
import numpy as np
import pandas as pd
from tabulate import tabulate
import torch
from torch.utils.data import Subset
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

@torch.no_grad()
def prepare_data_features(model, dataset):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Prepare model
    network = deepcopy(model)
    network.to(device)
    network.infer_all()

    # Encode all images
    NUM_WORKERS = 0
    data_loader = data.DataLoader(dataset, batch_size=64, num_workers=NUM_WORKERS, shuffle=False)
    labels = []
    feat_dict = {
        'MT': [],
        'MST': [],
        'V1-DS': [],
        'Full vis.': [],
    }

    for batch_input, batch_labels in tqdm(data_loader):
        optic_flow, movement, visual_sequence, segmentation_labels = batch_input
        optic_flow = optic_flow.to(device)
        network.reset_activity(batch_size=optic_flow.shape[0])
        movement = movement.to(device)
        eval_frame = 30
        for frame_it in range(eval_frame):
            _, _, _, _ = network(optic_flow[:, frame_it], movement[:, frame_it])

        # Get features from populations
        batch_feats = dict.fromkeys(feat_dict)

        batch_feats['MT'] = torch.sigmoid(network.MT_pre.flatten(start_dim=1)) # Flatten number of feature maps
        batch_feats['MT'] = torch.swapaxes(batch_feats['MT'], 1, 0) # Batch dim trailing.

        batch_feats['MST'] = torch.sigmoid(network.MST_pre.flatten(start_dim=1)) # Flatten number of feature maps
        batch_feats['MST'] = torch.swapaxes(batch_feats['MST'], 1, 0) # Batch dim trailing.

        batch_feats['V1-DS'] = optic_flow[:, -1, :, :, 0].flatten(start_dim=1) # Should be (1600, batch_size)
        batch_feats['V1-DS'] = torch.swapaxes(batch_feats['V1-DS'], 1, 0) # Batch dim trailing.

        batch_feats['Full vis.'] = visual_sequence[:, -1, :, :].flatten(start_dim=1) # Should be (1600, batch_size)
        batch_feats['Full vis.'] = torch.swapaxes(batch_feats['Full vis.'], 1, 0) # Batch dim trailing.

        for key in feat_dict.keys():
            feat_dict[key].append(batch_feats[key].detach().cpu())
        labels.append(batch_labels)

    # Concatenate along batch dimension
    labels = torch.cat(labels, dim=0)
    labels, idxs = labels.sort()
    for key in feat_dict.keys():
        feat_dict[key] = torch.cat(feat_dict[key], dim=-1) 
        feat_dict[key]= feat_dict[key][:, idxs] # Sort images by labels
        feat_dict[key] = torch.swapaxes(feat_dict[key], 1, 0) # Batch dim leading

        # Normalize
        mean = torch.mean(feat_dict[key], dim=0) 
        std = torch.std(feat_dict[key], dim=0)
        feat_dict[key] = (feat_dict[key] - mean) / (std + 1e-6)
    
        #feats = torch.where(feats==torch.inf, 0, feats)
        feat_dict[key] = torch.where(torch.abs(feat_dict[key]) > 100, 0, feat_dict[key])

    for key in feat_dict.keys():
        feat_dict[key] = data.TensorDataset(feat_dict[key], labels)
    return feat_dict

class Classifier(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.softmax(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features, 160)
        self.linear2 = nn.Linear(160, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x
    
def classification_accuracy(train_set, test_set):    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 64

    train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    dummy_representation, dummy_label = next(iter(train_loader))
    latent_dim = dummy_representation.shape[1]
    n_train_samples = len(train_set)
    n_test_samples = len(test_set)
    classifier = Classifier(in_features=latent_dim)
    classifier.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    for epoch_it in range(50):
        running_loss = 0
        for i_batch, sample_batch in enumerate(train_loader):

            feats, labels = sample_batch[0].to(device), sample_batch[1].to(device)
            x = classifier(feats)

            loss = criterion(x, labels) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'[{epoch_it + 1}] loss: {running_loss:.3f}')
        running_loss = 0.0
            
        # Evaluate accuracy on train set
        correct_preds = 0
        for i_batch, sample_batch in enumerate(train_loader):
            feats, labels = sample_batch[0].to(device), sample_batch[1].to(device)
            pred = classifier(feats)
            correct_preds += torch.sum(torch.argmax(pred, dim=1) == labels)
        train_acc = correct_preds / n_train_samples

        # Evaluate accuracy on test set
        correct_preds = 0
        for i_batch, sample_batch in enumerate(test_loader):
            feats, labels = sample_batch[0].to(device), sample_batch[1].to(device)
            pred = classifier(feats)
            correct_preds += torch.sum(torch.argmax(pred, dim=1) == labels)
        test_acc = correct_preds / n_test_samples
    return train_acc.cpu(), test_acc.cpu()

def to_table_cell(acc_mean_std):
    # Takes a list with a tuple of shape (1, 2) and returns a string of shape 'mean /pm std'
    acc_mean = acc_mean_std[0][0] * 100
    acc_std = acc_mean_std[0][1] * 100
    return f'{acc_mean:.1f} $\pm$ {acc_std:.1f}'

def tab_readout(n_runs=2):
    torch.manual_seed(0)

    # Train networks
    pretrain_fashionmnist(n_runs=n_runs)
    train_fashionmnist(n_runs=n_runs)

    populations = ['MT', 'MST', 'V1-DS', 'Full vis.']
    acc_train = Recording(*populations)
    acc_test = Recording(*populations)

    acc = pd.DataFrame(columns = ['Population', 'Train acc.', 'Test acc.'])

    for run_it in range(n_runs):
        acc_train.add_run()
        acc_test.add_run()

        # Load trained network
        net = HierarchicalNetwork(img_width=40)
        net.load_state_dict(torch.load(f'{network_paths["trained_fashionmnist"]}_run-{run_it}'))
        net.infer_all()

        # Prepare data from which to infer representations. Smaller subsets are sufficient, but not necessary
        num_train_samples = 4000 
        num_test_samples = 1000 
        train_set_network = fashionmnist_train()
        test_set_network = fashionmnist_test()
        train_set_network = Subset(train_set_network, np.arange(num_train_samples))
        test_set_network = Subset(test_set_network, np.arange(num_test_samples))

        # Infer representations with trained networks and get visual features.
        train_sets_readout = prepare_data_features(net, train_set_network)
        test_sets_readout = prepare_data_features(net, test_set_network)

        # Train classifiers on representations 
        for pop_it in populations:
            population_train_acc, population_test_acc = classification_accuracy(train_sets_readout[pop_it], test_sets_readout[pop_it])
            acc_train.update(**{pop_it: population_train_acc})
            acc_test.update(**{pop_it: population_test_acc})
            acc = pd.concat([acc, pd.DataFrame({'Population': pop_it, 'Train acc.': population_train_acc.numpy(), 'Test acc.': population_test_acc.numpy()}, index = [0])], ignore_index=True)

    acc_train_mean_std = acc_train.compute_mean_std()
    acc_test_mean_std = acc_test.compute_mean_std()
    # Save acc
    acc.to_csv('results/intermediate/readout.csv', index=False)

    table = []
    headers = ['Area', 'Acc. train', 'Acc. test']
    for row_it in iter(acc_train_mean_std):
        table.append([row_it, to_table_cell(acc_train_mean_std[row_it]), to_table_cell(acc_test_mean_std[row_it]) ])
    table.append(['Chance', '10.0', '10.0'])
    table_latex = tabulate(table, headers=headers, tablefmt="latex_raw", floatfmt=".2f")
    table_readable = tabulate(table, headers=headers, floatfmt=".2f")

    # Store both to text files
    with open('results/figures/table_latex.txt', 'w') as f:
        f.write(table_latex)
    with open('results/figures/table_readable.txt', 'w') as f:
        f.write(table_readable)
    return table_latex, table_readable

if __name__ == '__main__':
    table_latex, table_readable = tab_readout(n_runs=4)
    print(table_latex)
    print(table_readable)