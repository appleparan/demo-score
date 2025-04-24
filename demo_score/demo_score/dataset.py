import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import h5py
import csv
from torch.utils.data import RandomSampler
from torch.utils.data import BatchSampler

class ClassifierDataset(Dataset):

    def __init__(self, data_dir, data_root='./data', format='lerobot', include_ep_idxs=False,
                 split_ep_idx = None, max_steps=None, input_spec = None):
        """
        split_ep_idx -- manually include which epochs to include (for a manual train/val split)
        """
        self.input_spec = input_spec
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if max_steps is not None:
            if format != 'robodiff':
                raise NotImplementedError("max_steps only compatible with robodiff")

        self.include_ep_idxs = include_ep_idxs
        self.format = format
        if self.format == 'lerobot':
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            
            self.lrd = LeRobotDataset(data_dir, data_root, split="train")

            if split_ep_idx is not None:
                for e in range(self.lrd.num_episodes):
                    if e not in split_ep_idx:
                        raise NotImplementedError("Right now using partial LeRobotDataset for classifier is not supported")

            self.obs_state = torch.stack(self.lrd.hf_dataset['observation.state'][:], dim=0).to(self.device)
            if 'next.reward' in self.lrd.hf_dataset.features:
                self.rew = torch.tensor(self.lrd.hf_dataset['next.reward'][:]).to(self.device)
            self.ep_data_index = {k:v.clone() for k, v in self.lrd.episode_data_index.items()}
            self.ep_index = self.lrd.hf_dataset['episode_index'][:]
            self.labels = []
            for ep in range(self.ep_data_index['to'].shape[0]):
                from_index = self.ep_data_index['from'][ep]
                to_index = self.ep_data_index['to'][ep]
                if hasattr(self, "rew"):
                    success = self.rew[to_index-1] > 3.99
                else:
                    success = True # assume if no rewards then all episodes successes
                self.labels.append(success)
            self.labels = torch.tensor(self.labels).float().to(self.device)
        elif self.format == 'robodiff':
            self.actions = []
            self.labels = []
            with h5py.File(os.path.join(data_root, data_dir), 'r') as f:
                if 'data' in f:
                    self.modes = []
                    episode_type = 'rollout' if 'rollout_0' in f['data'].keys() else 'demo'
                    for i in range(len(f['data'])):
                        if split_ep_idx is not None:
                            if i not in split_ep_idx:
                                continue
                        actions = torch.tensor(f[f'data/{episode_type}_{i}/actions'][:]).float().to(self.device)
                        if 'object' in self.input_spec:
                            object_obs = torch.tensor(f[f'data/{episode_type}_{i}/obs/object'][:]).float().to(self.device)
                            actions = torch.cat((actions, object_obs), dim=1) #TODO HACK since this is no longer just actions consider renaming
                        if max_steps is not None:
                            if actions.shape[0] > max_steps:
                                actions = actions[:max_steps]
                        self.actions.append(actions)
                        if episode_type == 'demo':
                            self.labels.append(True)
                        else:
                            self.labels.append(f[f'data/{episode_type}_{i}'].attrs['success'])
                        if 'scripted_policy_type' in f[f'data/{episode_type}_{i}'].attrs:
                            self.modes.append(f[f'data/{episode_type}_{i}'].attrs['scripted_policy_type'])
                    if len(self.modes) == len(self.actions):
                        self._mode_arrays = self._create_mode_arrays(self.modes)
                        self.has_mode_arrays = True
                else:
                    for i in range(len(f)):
                        if split_ep_idx is not None:
                            if i not in split_ep_idx:
                                continue
                        actions = torch.tensor(f[f'rollout_{i}'][:])
                        self.actions.append(actions)
                        self.labels.append(f[f'rollout_{i}'].attrs['succes'])
            self.labels = torch.tensor(self.labels).to(self.device)
            self.actions = [a.to(self.device) for a in self.actions]

            self.ep_index = []
            self.cumulative_traj_lens = []
            cumul_len = 0
            for i, actions in enumerate(self.actions):
                self.cumulative_traj_lens.append(cumul_len)
                self.ep_index.extend([i]*actions.shape[0])
                cumul_len += actions.shape[0]
            self.ep_index = np.array(self.ep_index, dtype=np.int32)
            assert self.ep_index.shape[0] == sum([actions.shape[0] for actions in self.actions]) #TODO remove uneccesary assert
        elif self.format == 'aloha':
            full_dir = os.path.join(data_root, data_dir)
            self.state_vecs = []
            
            ep_idx = -1
            while ep_idx < 999999:
                ep_idx += 1
                fname = os.path.join(full_dir, f'episode_{ep_idx}.hdf5')
                if not os.path.exists(fname):
                    break
                print(fname)
                with h5py.File(fname, 'r') as f:
                    qpos = f['observations']['qpos'][:, :]
                    qvel = f['observations']['qvel'][:, :]
                    self.state_vecs.append(torch.tensor(np.hstack((qpos, qvel))).float().to(self.device))
                
            if os.path.exists(os.path.join(full_dir, 'labels.csv')):
                with open(os.path.join(full_dir, 'labels.csv'), 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    label_arr = list(reader)
                    self.labels = torch.tensor([float(l[0]) for l in label_arr]).float().to(self.device)
            else:
                self.labels = torch.ones((len(self.state_vecs,))).float().to(self.device) ## All demos are successes


            #HACK duplicate logic with robodiff -- merge into helper func
            self.ep_index = []
            self.cumulative_traj_lens = []
            cumul_len = 0
            for i, svs in enumerate(self.state_vecs):
                self.cumulative_traj_lens.append(cumul_len)
                self.ep_index.extend([i]*svs.shape[0])
                cumul_len += svs.shape[0]
            self.ep_index = np.array(self.ep_index, dtype=np.int32)
            assert self.ep_index.shape[0] == sum([sv.shape[0] for sv in self.state_vecs]) #TODO remove uneccesary assert
  


    def _create_mode_arrays(self, modes, include_names=[]):
        modes = [m if " " not in m else "neither" for m in modes]
        mode_names = list(set(modes))
        for inc_name in include_names:
            if inc_name not in mode_names:
                mode_names.append(inc_name)
        mode_arrays = {}
        for mode_name in mode_names:
            mode_arrays[mode_name] = []
            for m in modes:
                mode_arrays[mode_name].append(m == mode_name)
            mode_arrays[mode_name] = torch.tensor(mode_arrays[mode_name]).to(self.device)
        return mode_arrays

    def get_mode_array_for_episodes(self, ep_idxs):
        res = {}
        for k in self._mode_arrays:
            res[k] = self._mode_arrays[k][ep_idxs]
        return res

    def get_shape(self):
        if self.format == 'lerobot':
            return 14
        elif self.format == 'robodiff':
            if 'object' in self.input_spec and self.input_spec is not None:
                return 21
            else:
                return 7
        elif self.format == 'aloha':
            return 14 #HACK for now hardcode one-sided aloha (7 pos + 7 vel)
            
    

class TrajwiseClassifierDataset(ClassifierDataset):

    def __init__(self, data_dir, steps=None, data_root='./data', format='lerobot', **kwargs):
        super().__init__(data_dir, data_root, format=format, **kwargs)
        self.steps = steps
        print(self.steps, self.min_traj_len())
        self.steps = min(self.steps, self.min_traj_len(), kwargs['max_steps'] if 'max_steps' in kwargs else 999999999)
        if self.steps != steps:
            print(f"Shortening Traj Length to {self.steps}")

    def __len__(self):
        if self.format == 'lerobot':
            return self.lrd.num_episodes
        elif self.format == 'robodiff':
            return len(self.actions)
        elif self.format == 'aloha':
            return len(self.state_vecs)
  
    def min_traj_len(self):
        if self.format == 'lerobot':
            raise NotImplementedError
        elif self.format == 'robodiff':
            return min([a.shape[0] for a in self.actions])
        elif self.format == 'aloha':
            return min([o.shape[0] for o in self.state_vecs])
    
    def __getitem__(self, ep_index):
        if self.format == 'lerobot':
            from_index = self.ep_data_index['from'][ep_index]
            to_index = self.ep_data_index['to'][ep_index]
            if hasattr(self, "rew"):
                success = (self.rew[to_index-1] > 3.99).float() #final reward = 4 is success on aloha task
            else:
                success = torch.tensor(1.) # assume if no rewards then all episodes successes
            label =  success

            obs = self.obs_state[from_index:to_index]
            if self.steps is not None:
                obs = obs[:self.steps]

            return obs, label
        elif self.format == 'robodiff':
            actions = self.actions[ep_index][:self.steps]
            label = self.labels[ep_index].float()
            if self.include_ep_idxs:
                return actions, label, torch.tensor(ep_index)
            else:
                return actions, label
        elif self.format == 'aloha':
            state_vec = self.state_vecs[ep_index][:self.steps]
            label = self.labels[ep_index]
            if self.include_ep_idxs:
                return state_vec, label, torch.tensor(ep_index)
            else:
                return state_vec, label
        

class TensorDatasetWrapper(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return idx


    
class StepwiseClassifierDataset(ClassifierDataset):

    def __init__(self, data_dir, data_root='./data', format='lerobot', **kwargs):
        super().__init__(data_dir, data_root, format=format, **kwargs)

    def __len__(self):
        if self.format == 'lerobot':
            return self.lrd.num_samples
        elif self.format == 'robodiff':
            return sum([a.shape[0] for a in self.actions])
        elif self.format == 'aloha':
            return sum([o.shape[0] for o in self.state_vecs])
        
    def get_num_eps(self):
        if self.format == 'lerobot':
            return int(self.lrd.num_episodes) #TODO: this will cause error when dataset is split 
        elif self.format == 'robodiff':
            return len(self.actions)
        elif self.format == 'aloha':
            return len(self.state_vecs)
        
    
    def __getitem__(self, index):
        ep_index = self.ep_index[index]
        if self.format == 'lerobot':
            """
            from_index = self.ep_data_index['from'][ep_index]
            to_index = self.ep_data_index['to'][ep_index]
            if hasattr(self, "rew"):
                success = self.rew[to_index-1] > 3.99
            else:
                success = True # assume if no rewards then all episodes successes
            label = torch.tensor(1., device=device) if success else torch.tensor(0., device=device)

            assert label == self.labels[ep_index]
            """
            label = self.labels[ep_index]

            return self.obs_state[index], label
        elif self.format == 'robodiff':
            step_index = index - self.cumulative_traj_lens[ep_index]
            action = self.actions[ep_index][step_index]
            label = self.labels[ep_index].float()
            #label = torch.tensor(1., device='cuda') if label else torch.tensor(0., device='cuda')
            return action, label
        elif self.format == 'aloha':
            step_index = index - self.cumulative_traj_lens[ep_index]
            state_vec = self.state_vecs[ep_index][step_index]
            label = self.labels[ep_index]
            return state_vec, label

    
    def get_episode(self, ep_index):
        if self.format == 'lerobot':
            from_index = self.ep_data_index['from'][ep_index]
            to_index = self.ep_data_index['to'][ep_index]
            traj = self.obs_state[from_index:to_index]
            if isinstance(traj, list):
                traj = torch.stack(traj)

            if hasattr(self, "rew"):
                success = self.rew[to_index-1] > 3.99
            else:
                success = True # assume if no rewards then all episodes successes
            return traj, success
        elif self.format == 'robodiff':
            return self.actions[ep_index], self.labels[ep_index]
        elif self.format == 'aloha':
            return self.state_vecs[ep_index], self.labels[ep_index]
        
    def get_fast_dataloader(self, batch_size):
        features = []
        labels = []
        for idx in range(len(self)):
            feat, lab = self[idx]
            features.append(feat)
            labels.append(lab)
            
        features = torch.stack(features, dim=0).to(self.device)
        labels = torch.tensor(labels).to(self.device)

        sampler = RandomSampler(range(len(self)))

        batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)

        def collate_func(indices):
            return features[indices], labels[indices]
        
        dataloader = DataLoader(dataset=TensorDatasetWrapper(size=len(self)),
                                batch_sampler=batch_sampler,
                                collate_fn=collate_func)
        
        return dataloader


            
    