import matplotlib.pyplot as plt
from model2_retinotopic.data_handling.animals_datasets import animals_training_coupled
from model2_retinotopic.network.training_configurations import train_animals_separately
from model2_retinotopic.network.network_paths import network_paths
from model2_retinotopic.create_figures.figure_helpers import snapshot
import torch

def fig_snapshot():
    torch.manual_seed(0)
    train_dataset = animals_training_coupled()
    train_animals_separately(n_runs=1)

    snapshot(load_path_trained=network_paths['trained_animals'], 
                      load_path_pretrained=network_paths['pretrained_animals'],
                      dataset=train_dataset, 
                      eval_frame=28)
    
if __name__ == '__main__':
    fig_snapshot()
    plt.show()


   