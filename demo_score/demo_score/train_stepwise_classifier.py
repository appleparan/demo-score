import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import numpy as np
from tqdm import tqdm
import random

from .dataset import StepwiseClassifierDataset
from .models import StepwiseMLPClassifier



PIN_MEMORY=False
NUM_WORKERS=0
FAST_DATALOADER = False #Use Dataloader the circumvents Dataset class. Potentially faster but not fully tested



def eval_rollouts(dataset, model, prefix, criterion=None, thresholds=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preds = []
    labels = []
    inputs = []
    ep_idxs = []
    res = {f'{prefix}/stepwise_bce_loss': 0.,}

    datasets = dataset if isinstance(dataset, list) else [dataset,]

    for dataset in datasets:
        for ep in range(dataset.get_num_eps()):
            input, label = dataset.get_episode(ep)
            label = torch.full((input.shape[0],), float(label))
            input, label = input.to(device), label.to(device)
            frac = input.shape[0]/len(dataset)

            ep_idxs.append(ep)

            with torch.inference_mode():
                pred = model(input).squeeze(1)
                if criterion is not None:
                    bce_loss = F.binary_cross_entropy(pred, label)
                    res[f'{prefix}/stepwise_bce_loss'] += bce_loss.item()*frac

                preds.append(pred.mean().detach().view(1))
                labels.append(label[0].view(1))
                inputs.append(input)

    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    inputs = torch.cat(inputs, dim=0)

    res[f'{prefix}/mean'] = preds.mean().item()
    res[f'{prefix}/fail_mean'] = preds[labels < 0.5].mean().item()
    res[f'{prefix}/success_mean'] = preds[labels >= 0.5].mean().item()

    if thresholds is None:
        thresholds = res[f'{prefix}/mean']
        thresholds = {"": res[f'{prefix}/mean'], "ThreshFailMean": res[f'{prefix}/fail_mean']}
        res['thresholds'] = thresholds.copy()

    res[f'{prefix}/trajwise_bce_loss'] = F.binary_cross_entropy(preds, labels)
    
    return res



def train(data_dir, data_root, run_name, num_eps=3000, num_val=0, dropout_prob=0.3, hidden_sizes=[8, 8], format='robodiff', weight_ratio=4,
           ckpt_val_dataset=None, batch_size=1024, log_root='./data/classifier_logs/', dataset_size=100, max_steps=None, lr=1e-4, weight_decay=0.1, ckpt_val_dataset_size=None, input_spec = 'default'):

    print("input_spec", input_spec)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ckpt_val_dataset_size is None:
        ckpt_val_dataset_size = dataset_size
    print("Using ckpt_val_dataset_size", ckpt_val_dataset_size)
    

    dataset = StepwiseClassifierDataset(data_dir=data_dir,
                                        data_root=data_root,
                                        format=format,
                                        max_steps=max_steps,
                                        input_spec = input_spec)
    num_rollout_eps = dataset.get_num_eps()
    del dataset

    if num_val > 0:
        if format != 'robodiff':
            raise NotImplementedError("Can't handle same-checkpoint val episodes right now")

        ep_indices = list(range(dataset_size + num_val))
        val_ep_indices = random.sample(ep_indices, num_val)
        train_ep_indices = [ei for ei in ep_indices if ei not in val_ep_indices]
    else:
        train_ep_indices = list(range(dataset_size))
        val_ep_indices = None #HACK 


    


    train_dataset = StepwiseClassifierDataset(data_dir=data_dir,
                                        data_root=data_root,
                                        format=format,
                                        split_ep_idx = train_ep_indices,
                                        max_steps=max_steps,
                                        input_spec = input_spec)
    if num_val > 0:
        if format == 'lerobot':
            raise NotImplementedError("Dataset class will not handle this currently")
        val_dataset = StepwiseClassifierDataset(data_dir=data_dir,
                                            data_root=data_root,
                                            format=format,
                                            split_ep_idx = val_ep_indices,
                                            max_steps=max_steps,
                                            input_spec = input_spec)


    model = StepwiseMLPClassifier(train_dataset.get_shape(), hidden_sizes, dropout_prob).to(device)
    #print(model)
    model.train()

    log_dir = os.path.join(log_root, run_name)
    writer = SummaryWriter(log_dir=log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(os.path.join(log_dir, "train_val_eps.txt"), 'w') as f:
        f.write("train_eps " + " ".join([str(idx) for idx in train_ep_indices]) + '\n')
        if val_ep_indices is not None:
            f.write("val_eps " + " ".join([str(idx) for idx in val_ep_indices]) + '\n')

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCELoss()

    if FAST_DATALOADER:
        train_dataloader = train_dataset.get_fast_dataloader(batch_size=batch_size)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)

    if num_val > 0:
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)

    if ckpt_val_dataset is not None:
        ckpt_val_dataset = StepwiseClassifierDataset(ckpt_val_dataset, format=format, data_root=data_root, max_steps=max_steps, split_ep_idx =list(range(ckpt_val_dataset_size)), input_spec = input_spec)

    best_ep = 0
    min_val_loss = 99999999
    
    for ep in tqdm(range(num_eps)):
        
        model.train()

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            #Training
            opt.zero_grad()
            preds = model(inputs).squeeze(1)
            #bce_loss = criterion(preds, labels.float())
            bce_loss = F.binary_cross_entropy(preds, labels, weight=None if weight_ratio is None else labels.float() + 1/(weight_ratio - 1))
            bce_loss.backward()
            opt.step()

        model.eval()
        for k, v in eval_rollouts(train_dataset, model, "Train", criterion).items():
            if str(k) == 'thresholds':
                thresholds = v
            else:
                writer.add_scalar(str(k), v, ep)
        if num_val > 0:
            for k, v in eval_rollouts(val_dataset, model, "Val", criterion, thresholds=thresholds).items():
                writer.add_scalar(str(k), v, ep)
        if ckpt_val_dataset is not None:
            for k, v in eval_rollouts(ckpt_val_dataset, model, "CkptVal", criterion, thresholds=thresholds).items():
                writer.add_scalar(str(k), v, ep)
                if "stepwise_bce_loss" in str(k):
                    if v < min_val_loss:
                        min_val_loss = v
                        best_ep = ep
                        best_threshold = thresholds[""]
        #if (best_ep == ep) or (ep < 30) or (ep < 300 and ep % 10 == 0) or ep % 100 == 0:
        torch.save(model.state_dict(), os.path.join(log_dir, f"model_{ep}.pth"))

    with open(os.path.join(log_dir, "best_ep.txt"), 'w') as f:
        f.write(str(best_ep) + " " + str(best_threshold))