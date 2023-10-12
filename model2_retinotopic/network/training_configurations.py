import os
import torch   
from torch.utils.data import Subset
from model2_retinotopic.network.train import train
from model2_retinotopic.data_handling.fashionmnist_datasets import empty_like_fashionmnist_train, fashionmnist_train
from model2_retinotopic.data_handling.multimove_dataset import empty_multimove
from model2_retinotopic.data_handling.animals_datasets import animals_interleaved_with_empty, empty_like_animals_coupled, animals_training_coupled, animals_training_noncoupled
from model2_retinotopic.network.network_paths import network_paths
import numpy as np

# Training parameters
params_learning_curve = {
    'epochs_pretrain': 10,
    'epochs_train': 15,#15,
    'lr_pretrain': 200,
    'lr_train': 0.02, # 0.02
}

params_separate_training = {
    'epochs_pretrain': 10,
    'epochs_train': 10, 
    'lr_pretrain': 200,
    'lr_train': 0.02, # 0.02
}

# Wrapper functions for training and pretraining
def retraining_condition(network_path, n_runs=1, rec_path=None):

    SKIP_RETRAINING_QUERY = False  # Should be set to False unless you want to skip all optional retraining. This will easily lead to wrong results as 
    # even though retraining may be desired (due to change in initialization), but is automatically skipped with  enabled.

    exists_already = 1
    do_retrain = None
    for run_it in range(n_runs):
        exists_already *= int(os.path.isfile(network_path + f'_run-{run_it}'))
    rec_path_provided = (rec_path is not None)
    if rec_path_provided:
        exists_already *= int(os.path.isfile(rec_path))
    if exists_already == False:
        do_retrain = True
    else:
        while do_retrain is None:
            if SKIP_RETRAINING_QUERY:
                do_retrain = False
                break
            text = input(f"Existing networks found under {network_path}. [Y] to use, [N] to retrain from scratch. ")  
            if text == 'Y':
                print("Using existing networks")
                do_retrain = False
            elif text == 'N':
                print("Retraining networks")
                do_retrain = True
            else:
                # Repeat until valid input
                print("Invalid input. Please try again.")
    return do_retrain

    
def pretrain_fashionmnist(n_samples=10, n_runs=1):
    # Pretrain network on empty dataset and save its weights
    net_path = network_paths['pretrained_fashionmnist']
    retrain = retraining_condition(net_path, n_runs)
    if retrain:
        train_dataset = Subset(empty_like_fashionmnist_train(), np.arange(n_samples))
        train(n_runs=n_runs,
            n_epochs=params_separate_training['epochs_pretrain'],
            learning_rate=params_separate_training['lr_pretrain'],
            save_path=net_path,
            train_dataset=train_dataset,
            )
    
def train_fashionmnist(n_samples=100, n_runs=1):
    # Train network on fashionmnist dataset and save its weights
    net_path = network_paths['trained_fashionmnist']
    retrain = retraining_condition(net_path, n_runs)
    if retrain:
        train_dataset = Subset(fashionmnist_train(), np.arange(n_samples))
        train(n_runs=n_runs,
            n_epochs=params_separate_training['epochs_train'],
            learning_rate=params_separate_training['lr_train'],
            load_path=network_paths['pretrained_fashionmnist'],
            save_path=net_path,
            train_dataset=train_dataset,
            )
        
def train_animals_jointly(n_runs=1):
    net_path = network_paths['trained_animals_jointly']
    retrain = retraining_condition(net_path, n_runs)
    if retrain:
        train(n_runs=1, 
            n_epochs=40,
            learning_rate=0.02,
            train_simultaneously=True,
            save_path=net_path,
            train_dataset=animals_interleaved_with_empty(),
            test_dataset=animals_training_coupled()
            )

def pretrain_animals_noncoupled(n_runs=1):
    net_path = network_paths['pretrained_animals_noncoupled']
    rec_path = 'results/intermediate/rec_animals_pretrained_noncoupled.npy'
    retrain = retraining_condition(net_path, n_runs, rec_path)
    if retrain:
        recording = train(n_runs=n_runs, 
                        n_epochs=params_learning_curve['epochs_pretrain'], 
                        train_dataset=animals_training_noncoupled(), 
                        learning_rate=params_learning_curve['lr_pretrain'],
                        save_path=net_path)
        np.save(rec_path, recording)
    else:
        recording = np.load(rec_path, allow_pickle=True).item()
    return recording

def train_animals_nonempty(n_runs=1):
    path_pretrained = network_paths['pretrained_animals_nonempty']
    rec_path_pretrained = 'results/intermediate/rec_animals_pretrained_nonempty.npy'
    redo_pretraining = retraining_condition(path_pretrained, n_runs, rec_path_pretrained)
    if redo_pretraining:
        recording_pretrain = train(n_runs=n_runs, 
                                n_epochs=params_learning_curve['epochs_pretrain'], 
                                train_dataset=animals_training_coupled(), 
                                learning_rate=params_learning_curve['lr_pretrain'],
                                save_path=path_pretrained)
        np.save(rec_path_pretrained, recording_pretrain)
    else:
        recording_pretrain = np.load(rec_path_pretrained, allow_pickle=True).item()
    
    path_trained = network_paths['trained_animals_after_nonempty_pretraining']
    rec_path_trained = 'results/intermediate/rec_animals_trained_after_nonempty_pretraining.npy'
    redo_training = retraining_condition(path_trained, n_runs, rec_path_trained)
    if redo_training:
        recording_train = train(n_runs=n_runs, 
                                n_epochs=params_learning_curve['epochs_train'], 
                                train_dataset=animals_training_coupled(), 
                                learning_rate=params_learning_curve['lr_train'],
                                load_path=path_pretrained,
                                save_path=path_trained)
        np.save(rec_path_trained, recording_train)
    else:
        recording_train = np.load(rec_path_trained, allow_pickle=True).item()
    return recording_pretrain, recording_train


def pretrain_animals(n_runs=1):
    net_path = network_paths['pretrained_animals']
    rec_path = 'results/intermediate/rec_animals_pretrained.npy'
    retrain = retraining_condition(net_path, n_runs, rec_path)
    if retrain:
        recording = train(n_runs=n_runs, 
                        n_epochs=params_separate_training['epochs_pretrain'], 
                        train_dataset=empty_like_animals_coupled(), 
                        learning_rate=params_separate_training['lr_pretrain'],
                        save_path=net_path)
        np.save(rec_path, recording)
    else:
        recording = np.load(rec_path, allow_pickle=True).item()
    return recording

def train_animals_separately(n_runs=1):
    # Pretraining
    path_pretrained = network_paths['pretrained_animals']
    recording_pretrain = pretrain_animals(n_runs=n_runs)
    rec_paths = {
        'pretrain': 'results/intermediate/rec_animals_pretrained.npy',
        'training': 'results/intermediate/rec_animals_trained.npy',
    }

    # Training
    path_trained = network_paths['trained_animals']
    repeat_training = retraining_condition(path_trained, n_runs, rec_paths['training'])
    if repeat_training:
        recording_training = train(n_runs=n_runs, 
                                    n_epochs=params_separate_training['epochs_train'],
                                    train_dataset=animals_training_coupled(), 
                                    learning_rate=params_separate_training['lr_train'],
                                    load_path=path_pretrained,
                                    save_path=path_trained)
        # Save recording
        np.save(rec_paths['training'], recording_training)

    else:
        recording_training = np.load(rec_paths['training'], allow_pickle=True).item()
    return recording_pretrain, recording_training

def pretrain_multimove(n_runs=1):
    net_path = network_paths['pretrained_multimove']
    retrain = retraining_condition(net_path, n_runs)
    if retrain:
        train(n_runs=n_runs,
            n_epochs=params_separate_training['epochs_pretrain'],
            learning_rate=params_separate_training['lr_pretrain'],
            save_path=net_path,
            train_dataset=empty_multimove(),
            )