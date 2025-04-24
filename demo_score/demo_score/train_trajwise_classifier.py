import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import numpy as np
from tqdm import tqdm


from .dataset import TrajwiseClassifierDataset
from .models import MLPPoolMLP, TransformerClassifier, create_model


def eval_rollouts(dataloader, model, prefix, criterion=None,):

    preds = []
    labels = []
    inputs = []
    ep_idxs = []
    res = {f'{prefix}/trajwise_bce_loss': 0.}

    for tup in dataloader:

        input_batch, label_batch = tup[0].cuda(), tup[1].cuda()
        frac = float(label_batch.shape[0])/len(dataloader.dataset)
        if len(tup) > 2:
            ep_idxs.append(tup[2])

        with torch.inference_mode():
            pred_batch = model(input_batch)
            if criterion is not None:
                bce_loss = criterion(pred_batch, label_batch.float())
                res[f'{prefix}/trajwise_bce_loss'] += frac*bce_loss.item()

            preds.append(pred_batch.detach())
            labels.append(label_batch.detach())
            inputs.append(input_batch)
    
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    inputs = torch.cat(inputs, dim=0)
    labels = labels > 0.5
    if len(ep_idxs) > 0:
        ep_idxs = torch.cat(ep_idxs, dim=0)

    cl_success = (preds > 0.5)
    correct = cl_success == labels


    res[f'{prefix}/accuracy'] = correct.float().mean().item()
    res[f'{prefix}/precision'] = (torch.logical_and(cl_success, labels).sum()/cl_success.sum()).item()
    res[f'{prefix}/recall'] = (torch.logical_and(cl_success, labels).sum()/labels.sum()).item()


    return res



PIN_MEMORY=False
NUM_WORKERS=0

def train(data_dir, data_root, run_name, classifier_type='transformer', num_eps=3000, num_val=0, position_enc_dim=None, steps=300, pool='max', dropout_prob=0.3,
           enc_hidden_sizes=[8, 8], dec_hidden_sizes=[8, 8], model_dim=8, num_heads=2, num_layers=2, dim_feedforward=8, format='robodiff', weight_ratio=None,
           ckpt_val_dataset=None, batch_size=32, log_root='./data/classifier_logs/', dataset_size=100, **kwargs):

    train_ep_indices = list(range(dataset_size))

    dataset = TrajwiseClassifierDataset(data_dir=data_dir,
                                        steps=steps,
                                        data_root=data_root,
                                        format=format,
                                        split_ep_idx = train_ep_indices.copy(),
                                        include_ep_idxs = (format == 'aloha'))

    model = create_model(dataset.get_shape(), classifier_type, position_enc_dim, pool, dropout_prob, 
                         enc_hidden_sizes=enc_hidden_sizes, dec_hidden_sizes=dec_hidden_sizes,
                         model_dim=model_dim, num_layers=num_layers, num_heads=num_heads, dim_feedforward=dim_feedforward)
    print(model)
    model.train()

    log_dir = os.path.join(log_root, run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    criterion = torch.nn.BCELoss()

    if num_val > 0:
        raise NotImplementedError("Currently this is not working in some cases")
        train_dataset, val_dataset = random_split(dataset, [len(dataset) - num_val, num_val])
    else:
        train_dataset = dataset

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)
    if num_val > 0:
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)

    

    if ckpt_val_dataset is not None:
        ckpt_val_dataset = TrajwiseClassifierDataset(ckpt_val_dataset, steps=steps, format=format, data_root=data_root, include_ep_idxs = (format == 'aloha'), split_ep_idx = list(range(dataset_size)))
        ckpt_val_dataloader = DataLoader(ckpt_val_dataset, batch_size=batch_size, shuffle=True, pin_memory=PIN_MEMORY)

    min_val_loss = 99999999
    
    best_ep = 0
    for ep in tqdm(range(num_eps)):
        
        model.train()
        for batch in train_dataloader:
            inputs, labels = batch[0], batch[1]
            inputs, labels = inputs.cuda(), labels.cuda()

            #Training
            opt.zero_grad()
            preds = model(inputs)
            #bce_loss = criterion(preds, labels.float())
            bce_loss = F.binary_cross_entropy(preds, labels, weight=None if weight_ratio is None else labels.float() + 1/(weight_ratio - 1))
            bce_loss.backward()
            opt.step()

        model.eval()
        for k, v in eval_rollouts(train_dataloader, model, "Train", criterion,).items():
            writer.add_scalar(str(k), v, ep)
        if num_val > 0:
            for k, v in eval_rollouts(val_dataloader, model, "Val", criterion,).items():
                writer.add_scalar(str(k), v, ep)
        if ckpt_val_dataset is not None:
            for k, v in eval_rollouts(ckpt_val_dataloader, model, "CkptVal", criterion,).items():
                writer.add_scalar(str(k), v, ep)
                if "trajwise_bce_loss" in str(k):
                    if v < min_val_loss:
                        min_val_loss = v
                        best_ep = ep
        #if (best_ep == ep) or (ep < 30) or (ep < 300 and ep % 10 == 0) or ep % 100 == 0:
        torch.save(model.state_dict(), os.path.join(log_dir, f"model_{ep}.pth"))

    with open(os.path.join(log_dir, "best_ep.txt"), 'w') as f:
        f.write(str(best_ep))

    return best_ep

