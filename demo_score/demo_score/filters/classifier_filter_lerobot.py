from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset
from datasets import concatenate_datasets, Features, Image
import torch
from lerobot.scripts.collect_data import save_dataset, _compile_episode_data
from lerobot.scripts.push_dataset_to_hub import save_meta_data
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.compute_stats import compute_stats
from pathlib import Path
from lerobot.common.datasets.utils import hf_transform_to_torch
from safetensors.torch import load_file
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def classifier_filter(orig_datasets, new_datasets_file, model, rew_thresh=3.99, classifier_type='stepwise', classifier_thresh=None):

    if classifier_type != 'stepwise':
        raise NotImplementedError("Only stepwise classifier implemented right now")
    else:
        if classifier_thresh is None:
            raise ValueError("Must specify classifier threshold")
    
    model = model.to(device)

    orig_datasets_lines = []

    for orig_datasets_file in orig_datasets:
        if '.txt' in orig_datasets_file:
            with open(orig_datasets_file, 'r') as orig_datasets:
                for line in orig_datasets:
                    line = line.strip()
                    orig_datasets_lines.append(line)
        else:
            path = Path('/') / orig_datasets_file / "meta_data" / "episode_data_index.safetensors"
            ep_data_index = load_file(path)
            num_episodes = int(ep_data_index['to'].shape[0])
            line = orig_datasets_file + " " + ",".join([str(el) for el in list(range(num_episodes))])
            orig_datasets_lines.append(line)
    
    new_lines = []
    for line in orig_datasets_lines:
        fname = line.split(' ')[0]

        path = Path('/') / fname / "meta_data" / "episode_data_index.safetensors"

        ep_data_index = load_file(path)
        take_eps = []
        for episode_idx in line.split(' ')[1].split(','):
            episode_idx = int(episode_idx)
            dataset_repo_root = '/'
            dataset_repo_id = fname
            split = f"train[{int(ep_data_index['from'][episode_idx])}:{int(ep_data_index['to'][episode_idx])}]"

            old_ep_dataset = LeRobotDataset(dataset_repo_id, dataset_repo_root, split=split)

            success = False
            if 'next.reward' in old_ep_dataset.hf_dataset.features:
                final_rew = old_ep_dataset.hf_dataset['next.reward'][-1]
                if final_rew > rew_thresh:
                    success = True
            else: #if no reward, assume it is a demo and assume success
                success = True

            if success:
                inputs = old_ep_dataset.hf_dataset['observation.state'][:]
                inputs = torch.stack(inputs).to(device)
                with torch.inference_mode():
                    pred = model(inputs).squeeze(0)
                if pred.shape[0] > 1:
                    pred = pred.mean()
                if pred.item() > classifier_thresh:
                    take_eps.append(episode_idx)
            
        if len(take_eps) > 0:
            new_line = fname + " " + ",".join([str(el) for el in take_eps])
            new_lines.append(new_line)


    os.makedirs(os.path.dirname(new_datasets_file), exist_ok=True)
    with open(new_datasets_file, 'w') as new_datasets_file:
        for nline in new_lines:
            new_datasets_file.write(nline + "\n")
