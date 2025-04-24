import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import os
import pandas
from itertools import product
import pandas as pd
import dataframe_image as dfi
import shutil
import torch
from .models import TransformerClassifier, StepwiseMLPClassifier

from .filters.classifier_filter_lerobot import classifier_filter as classifier_filter_lerobot
from .filters.classifier_filter_robodiff import classifier_filter as classifier_filter_robodiff
from .filters.classifier_filter_aloha import classifier_filter as classifier_filter_aloha

#analyze classifier sweep and perform filtering
#This file should import from the general one in demo_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_tensorboard_scalars(logdir):
    """
    Reads all scalar series from TensorBoard logs in the given directory and returns a dictionary of NumPy arrays.
    
    Args:
        logdir (str): Directory containing the TensorBoard event files.

    Returns:
        dict: A dictionary where keys are scalar tags, and values are dictionaries with 'steps' and 'values' NumPy arrays.
    """
    if not os.path.exists(logdir):
        return None

    # Initialize EventAccumulator
    ea = event_accumulator.EventAccumulator(logdir, size_guidance={
        event_accumulator.SCALARS: 0,  # Load all scalar data
    })
    ea.Reload()  # Loads events from file

    # Get all scalar tags
    tags = ea.Tags()['scalars']

    scalar_dict = {}

    for tag in tags:
        if "bce_loss" not in tag and "recall" not in tag and "mean" not in str(tag).lower():
            continue
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        scalar_dict[tag] = {'steps': steps, 'values': values}

    return scalar_dict

def get_dir(name, early_ep=25, late_ep=800, train_ds_size=100, val_ds_size=100, arch="small_stepwise", root_dir='./data/example/classifier_logs', weight_decay=0.1, seed=1):
    if late_ep is None:
        ep_str = f"train_ep{early_ep}_same_ep_val"
    else:
        ep_str = f"train_ep{early_ep}_val_ep{late_ep}"  

    res = os.path.join(root_dir, name, f'weight_decay_{weight_decay}', 'lr_0.0001',
                        f"train_dataset_size_{train_ds_size}", f"val_dataset_size_{val_ds_size}",
                        ep_str,
                        arch, str(seed))
    return res


def select_ep_ckpt_val_loss_min(log, same_ep_val=False, metric_type='stepwise'):
    val_kwrd = f'CkptVal/{metric_type}_bce_loss' if not same_ep_val else f'Val/{metric_type}_bce_loss'
    if val_kwrd not in log:
        return -1, -1.
    index = np.argmin(log[val_kwrd]['values'])
    val = log[val_kwrd]['values'][index]
    return index, val



def get_model(classifier_dir, arch_name):

    arch_type = 'stepwise' if 'stepwise' in arch_name else 'transformer'

    with open(os.path.join(classifier_dir, 'best_ep.txt'), 'r') as file:
        # Read the line from the file
        line = file.readline()
        
        # Split the line into parts
        parts = line.split()
        
        # Convert the parts to int and float
        ep = int(parts[0])
        if arch_type == 'stepwise':
            thresh = float(parts[1])
        else:
            thresh = 0.5

    print(ep, thresh)
    #TODO: get rid of harcoding of model hyperparams
    if arch_name == 'small_stepwise':
        model = StepwiseMLPClassifier(7, [8, 8], 0.3).to(device)
    elif arch_name == 'small_deep_stepwise':
        model = StepwiseMLPClassifier(7, [8, 8, 8], 0.3).to(device)
    elif arch_name == 'med_stepwise':
        model = StepwiseMLPClassifier(7, [16, 16], 0.3).to(device)
    elif arch_name == 'med_deep_stepwise':
        model = StepwiseMLPClassifier(7, [16, 16, 16], 0.3).to(device)
    elif arch_name == 'large_stepwise':
        model = StepwiseMLPClassifier(7, [32, 32], 0.3).to(device)
    elif arch_name == 'large_deep_stepwise':
        model = StepwiseMLPClassifier(7, [32, 32, 32], 0.3).to(device)
    elif arch_name == 'small_transformer':
        model = TransformerClassifier(7, model_dim=8,
                                     num_heads=2,
                                     num_layers=2,
                                     dropout_prob=0.3,
                                     dim_feedforward=8).to(device)
    elif arch_name == 'med_transformer':
        model = TransformerClassifier(7, model_dim=16,
                                     num_heads=2,
                                     num_layers=2,
                                     dropout_prob=0.3,
                                     dim_feedforward=16).to(device)
    elif arch_name == 'big_transformer':
        model = TransformerClassifier(7, model_dim=32,
                                     num_heads=4,
                                     num_layers=3,
                                     dropout_prob=0.3,
                                     dim_feedforward=32).to(device)

    model_file = os.path.join(classifier_dir, f"model_{ep}.pth")
    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))


    return model, thresh


def analyzer_classifier_sweep_and_filter(expt_dir, run_name, eps_dict, dataset_type='lerobot', model_arch='small_stepwise', train_dataset_size=100, val_dataset_size=100, max_steps=300, root_dir = None, variant_str = ""):

    assert dataset_type == 'lerobot', "only lerobot data type implemented so far"

    eps_dict = eps_dict[run_name]

    if root_dir is None:
        root_dir = os.path.join(expt_dir, 'classifier_logs')

    train_eps = eps_dict['train_eps']
    val_eps = eps_dict['val_eps']

    metric_type = 'trajwise' if 'transformer' in model_arch else 'stepwise'

    min_val_loss = 9999999999.
    chosen_classifier_log = None
    for tr_ep, val_ep in product(eps_dict[run_name]['train'], eps_dict[run_name]['cross_val']):
        log_dir = get_dir(run_name, train_ds_size=train_dataset_size, val_ds_size=val_dataset_size, early_ep=tr_ep, late_ep=val_ep, arch=model_arch, root_dir=root_dir)
        log = read_tensorboard_scalars(log_dir)
        val_loss = select_ep_ckpt_val_loss_min(log, metric_type=metric_type)
        #TODO: add something that prints oracle metrics for each classifier along with val loss
        print(log_dir, val_loss)
        if val_loss < min_val_loss:
            chosen_classifier_log = log_dir

    model, thresh = get_model(chosen_classifier_log, model_arch)


    #TODO: currently hardcoded to lerobot datatypes
    orig_demo_file = os.path.join(expt_dir, 'demos', f'{run_name}.txt')
    demo_score_only_demos = os.path.join(expt_dir, f'demo_score_datasets/{run_name}/demo_score_demos_only{variant_str}.txt')

    rollout_files = []
    all_eps = []
    for ep in eps_dict['train'] + eps_dict['cross_val']:
        if ep in all_eps:
            continue
        all_eps.append(ep)
        rollout_file = os.path.join(expt_dir, f'policy_logs/{run_name}_init_policy/rollouts/ep{ep}')
        rollout_files.append(rollout_file)
    
    demo_score_demos_plus_rollouts = os.path.join(expt_dir, f'demo_score_datasets/{run_name}/demo_score_rollouts_plus_demos{variant_str}.txt')

    # NEW FILTERED DATASET, DEMOS ONLY
    classifier_filter_lerobot(orig_datasets=[orig_demo_file,],
                              new_datasets_file=demo_score_only_demos,
                              model=model,
                              classifier_thresh=thresh)
    
    # NEW FILTERED DATASET, DEMOS + ROLLOUTS
    classifier_filter_lerobot(orig_datasets=[orig_demo_file,] + rollout_files.copy(),
                              new_datasets_file=demo_score_demos_plus_rollouts,
                              model=model,
                              classifier_thresh=thresh)

    
    print("Done filtering")

    