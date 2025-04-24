import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset
import click
import h5py
import numpy as np
from scipy.stats import wasserstein_distance_nd


class ClassifierMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_prob=0.3):
        super(ClassifierMLP, self).__init__()
        
        # List to store the layers
        layers = []
        
        # Input layer to the first hidden layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_prob))
        
        # Loop through hidden layers and dynamically add them
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
        
        # Final output layer
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        
        # Combine layers into a Sequential model
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return torch.sigmoid(self.model(x))
    
    def embedding(self, x):
        for layer in list(self.model.children())[:-3]:
            x = layer(x)
        return x

def make_dataset(trajectory_data : str, device="cuda", num_val=40):

    actions = []
    labels = []

    with h5py.File(trajectory_data, 'r') as f:
        for i in range(len(f)):
            actions.append(f[f'rollout_{i}'][:])
            labels.append(np.full(actions[-1].shape[0], f[f'rollout_{i}'].attrs['succes']))
    
    if num_val > 0:
        val_actions = actions[-num_val:]
        val_labels = [True if l[0] else False for l in labels[-num_val:]]

        val_dataset = {'actions': [torch.tensor(val_actions[i], dtype=torch.float32).to(device) for i in range(num_val)],
                        "labels": val_labels,
                        "num_episodes": num_val}
        
        train_actions = actions[:-num_val]
        train_labels = [True if l[0] else False for l in labels[:-num_val]]
    else:
        val_dataset = None
        train_actions = actions[:]
        train_labels = [True if l[0] else False for l in labels[:]]

    train_dataset = {'actions': [torch.tensor(train_actions[i], dtype=torch.float32).to(device) for i in range(len(train_actions))],
                    "labels": train_labels,
                    "num_episodes": len(train_actions)}

    actions = np.vstack(train_actions)
    labels = np.concatenate(labels[:len(labels)-num_val], axis=0).astype(float)

    labels = torch.tensor(labels, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.float32).to(device)

    dataset = TensorDataset(actions, labels)
    
    return dataset, train_dataset, val_dataset

def load_demos(demo_data, device):
    loaded_demos = []
    with h5py.File(demo_data, 'r') as f:
        demos = f['data']
        for i in range(len(demos)):
            loaded_demo = {}
            demo = demos[f'demo_{i}']
            loaded_demo['actions'] = torch.tensor(demo['actions'][:], dtype=torch.float32).to(device)
            if True:
                loaded_demo['mode'] = demo.attrs['scripted_policy_type']
            loaded_demos.append(loaded_demo)

    return loaded_demos
        
def demo_val(demo_data, classifier, mean):

    cutoffs = list(np.arange(0.1, 0.7, 0.05))
    cutoffs.append("mean")

    res = {}
    for c in cutoffs:
        for sf in ['success', "fail"]:
            for mode in ['ph', 'side']:
            #for mode in ['corner', 'side']:
                res[f"Oracle{c}/{mode}_{sf}"] = 0

    for demo in demo_data:
        pred = classifier(demo['actions']).mean()

        for c in cutoffs:
            c_num = mean if c == "mean" else c
            if pred > c_num:
                res[f"Oracle{c}/{demo['mode']}_success"] += 1
            else:
                res[f"Oracle{c}/{demo['mode']}_fail"] += 1
            
    return res

def rollout_stats(dataset, classifier, mean, criterion=None):

    stats = {'num_accurate': 0,
             'num_classified_success': 0,
             'success_accurate': 0,
             'fail_accurate': 0,
             'stepwise_accurate': 0,
             "num_episodes": dataset['num_episodes'],
             "num_actual_success": sum(dataset['labels']),
             "num_actual_fail":dataset['num_episodes'] - sum(dataset['labels']),
             }
    
    stats['frac_actual_success'] = stats['num_actual_success']/stats['num_episodes']

    total_steps = 0
    for i in range(dataset['num_episodes']):
        preds = classifier(dataset['actions'][i])
        pred = preds.mean()
        if criterion is not None:
            labels = torch.full(preds.shape[0], float(dataset['labels'][i])).cuda()
            loss = criterion(preds.squeeze(1), labels)
        accurate = (pred > mean) == dataset['labels'][i]
        if accurate:
            stats['num_accurate'] += 1
            if dataset['labels'][i]:
                stats['success_accurate'] += 1
            else:
                stats['fail_accurate'] += 1
        if pred > mean:
            stats['num_classified_success'] += 1

        num_preds = (preds > mean).sum()
        num_steps = preds.shape[0]
        if dataset['labels'][i]:
            stats['stepwise_accurate'] += num_preds
        else:
            stats['stepwise_accurate'] += num_steps - num_preds
        total_steps += num_steps

    stats['frac_classified_success'] = stats['num_classified_success']/stats['num_episodes']
    stats['frac_accurrate'] = stats['num_accurate']/stats['num_episodes']
    stats['frac_of_successes_accurate'] = stats['success_accurate']/stats['num_actual_success']
    stats['frac_of_fails_accurate'] = stats['fail_accurate']/stats['num_actual_fail']
    stats['stepwise_accurate_frac'] = float(stats['stepwise_accurate'])/total_steps
    
    return stats

def wasserstein(embs1, embs2):
    embs1_np = [e.mean(axis=0).cpu().detach().numpy() for e in embs1]
    embs2_np = [e.mean(axis=0).cpu().detach().numpy() for e in embs2]

    return wasserstein_distance_nd(embs1_np, embs2_np)

def cross_match(embs1, embs2):
    embs1 = torch.stack([e.mean(axis=0) for e in embs1], dim=0)
    embs2 = torch.stack([e.mean(axis=0) for e in embs2], dim=0)

    dists = []

    for i in range(len(embs1)):
        ds = ((embs1[i] - embs2)**2).mean(dim=1)
        dists.append(torch.min(ds).item())

    return sum(dists)/len(dists)


def emb_divergence(demo_data, rollout_dataset, classifier, mean):

    res = {}
    
    actual_fail_rollout_embeddings = []
    classified_fail_rollout_embeddings = []
    for rollout_i in range(rollout_dataset['num_episodes']):
        emb = classifier.embedding(rollout_dataset['actions'][rollout_i])
        if rollout_dataset['labels'][rollout_i]:
            actual_fail_rollout_embeddings.append(emb.detach().clone())
        if classifier(rollout_dataset['actions'][rollout_i]).mean() < mean:
            classified_fail_rollout_embeddings.append(emb.detach().clone())

    demo_embeddings = []
    for demo in demo_data:
        demo_embeddings.append(classifier.embedding(demo['actions']))

    res['wasserstein_classified'] = wasserstein(classified_fail_rollout_embeddings, demo_embeddings)
    res['wasserstein_actual'] = wasserstein(actual_fail_rollout_embeddings, demo_embeddings)

    res['cross_match_classified_demo'] = cross_match(classified_fail_rollout_embeddings, demo_embeddings)
    res['cross_match_demo_classified'] = cross_match(demo_embeddings, classified_fail_rollout_embeddings)

    res['cross_match_actual_demo'] = cross_match(actual_fail_rollout_embeddings, demo_embeddings)
    res['cross_match_demo_actual'] = cross_match(demo_embeddings, actual_fail_rollout_embeddings)

    return res




@click.command()
@click.option('-t', '--trajectory_data', required=True)
@click.option('--demo_data', default=None)
@click.option('-l', '--log_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('-n', '--noise', type=float, default=0.)
@click.option('--ckpt_val_data', default=None)
def train(trajectory_data, demo_data, log_dir, device, noise, ckpt_val_data):

    #classifier = ClassifierMLP(7, [16,16]).to(device) #big
    classifier = ClassifierMLP(7, [8,8]).to(device)
    #classifier = ClassifierMLP(7, [4,4]).to(device) #small

    traj_dataset, train_dataset, val_dataset = make_dataset(trajectory_data, device=device)
    traj_dataloader = DataLoader(traj_dataset, batch_size=256, shuffle=True)

    demo_data = load_demos(demo_data, device)

    _, ckpt_val_dataset, _ = make_dataset(ckpt_val_data, num_val=0)

    opt = optim.AdamW(classifier.parameters(), lr=1e-4, weight_decay=0.1)
    criterion = torch.nn.BCELoss()

    writer = SummaryWriter(log_dir=log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for ep in range(2000):

        classifier.train()
        running_stepwise_correct = 0
        running_loss = 0
        for inputs, labels in traj_dataloader:

            inputs += noise*torch.randn_like(inputs)           

            #training forward and backward pass
            opt.zero_grad()
            preds = classifier(inputs)
            loss = criterion(preds.squeeze(1), labels)
            running_loss = loss.item()*preds.shape[0]
            loss.backward()
            opt.step()
            correct = ((preds.squeeze(1) > 0.5) == (labels > 0.5)).sum().item()
            running_stepwise_correct += correct

        writer.add_scalar("Train/Loss", running_loss/len(traj_dataset), ep)
        writer.add_scalar("Train/StepwiseAccuracy", float(running_stepwise_correct)/len(traj_dataset), ep)
         
        classifier.eval()

        #compute mean and std
        all_preds = []
        eval_running_stepwise_correct = 0
        for inputs, labels in traj_dataloader:
            with torch.inference_mode():
                preds = classifier(inputs)

            correct = ((preds.squeeze(1) > 0.5) == (labels > 0.5)).sum().item()
            eval_running_stepwise_correct += correct

            all_preds.append(preds.cpu().numpy())

        all_preds = np.concatenate([p.flatten() for p in all_preds])
        mean = np.mean(all_preds)
        std = np.std(all_preds)

        writer.add_scalar("Eval/Mean", mean, ep)
        writer.add_scalar("Eval/STD", std, ep)
        writer.add_scalar("Eval/StepwiseAccuracy", float(eval_running_stepwise_correct)/len(traj_dataset), ep)

        if True: #only if labelled modes
            demo_eval_results = demo_val(demo_data, classifier, mean)
            for k,v in demo_eval_results.items():
                writer.add_scalar(k, v, ep)

        train_stats = rollout_stats(train_dataset, classifier, mean)
        for k, v in train_stats.items():
            writer.add_scalar("TrainStats/" + str(k), v, ep)

        val_stats = rollout_stats(val_dataset, classifier, mean, criterion=None)
        for k, v in val_stats.items():
            writer.add_scalar("ValStats/" + str(k), v, ep)
        
        ckpt_val_stats = rollout_stats(ckpt_val_dataset, classifier, mean, criterion=None)
        for k, v in ckpt_val_stats.items():
            writer.add_scalar("CkptValStats/" + str(k), v, ep)

        emb_div_stats = emb_divergence(demo_data, train_dataset, classifier, mean)
        for k, v in emb_div_stats.items():
            writer.add_scalar("EmbDiv/" + str(k), v, ep)

        ckpt_val_emb_div_stats = emb_divergence(demo_data, ckpt_val_dataset, classifier, mean)
        for k, v in ckpt_val_emb_div_stats.items():
            writer.add_scalar("CkptValEmbDiv/" + str(k), v, ep)

        if ep > 0:
            torch.save(classifier.state_dict(), log_dir + f"/model{ep}.pth")

if __name__ == '__main__':
    train()