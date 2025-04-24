from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset
from datasets import concatenate_datasets, Features, Image
import torch
from lerobot.scripts.collect_data import save_dataset, _compile_episode_data
from lerobot.scripts.push_dataset_to_hub import save_meta_data
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.compute_stats import compute_stats
from pathlib import Path
from lerobot.common.datasets.utils import hf_transform_to_torch


def success_filter(orig_dataset : LeRobotDataset, save=None, rew_thresh=0.95, max_episodes=9999999, match_max_episodes=False):
    print(f"Using reward threshold {rew_thresh}")
    num_eps = orig_dataset.num_episodes

    old_hf = orig_dataset.hf_dataset

    new_hf = None
    new_ep_data_index = None

    num_taken = 0

    ep_num = -1
    new_to_index = torch.tensor(0)
    for i in range(num_eps):
        from_index = orig_dataset.episode_data_index['from'][i]
        to_index = orig_dataset.episode_data_index['to'][i]

        if num_taken >= max_episodes:
            break
        
        if old_hf['next.reward'][to_index-1] > rew_thresh:
            ep_num += 1
            new_from_index = new_to_index
            new_to_index = new_to_index + (to_index - from_index)
            print(new_from_index, new_to_index)
            num_taken += 1
            subdataset = old_hf.select(range(from_index, to_index)).with_format(None)
            def replace_ep_index(example):
                example['episode_index'] = torch.tensor(ep_num)
                example['index'] = example['index'] - from_index + new_from_index
                return example
            new_subdataset = subdataset.map(replace_ep_index, num_proc=8, features=subdataset.features)
            for col_name in new_subdataset.column_names:
                if "image" in col_name:
                    new_subdataset = new_subdataset.cast_column(col_name, Image())
            if new_hf is None:
                new_hf = new_subdataset
                new_ep_data_index = {"from": new_from_index.reshape((1,)), "to": new_to_index.reshape((1,))}
            else:
                new_hf = concatenate_datasets([new_hf, new_subdataset])
                new_ep_data_index['from'] = torch.cat([new_ep_data_index['from'], new_from_index.reshape((1,))])
                new_ep_data_index['to'] = torch.cat([new_ep_data_index['to'], new_to_index.reshape((1,))])
    print(f"Took {num_taken} out of {num_eps} episodes")

    if match_max_episodes:
        if num_taken != max_episodes:
            print("Not enough succeses")
            return
    
    new_dataset = LeRobotDataset.from_preloaded(hf_dataset=new_hf, episode_data_index=new_ep_data_index)
    
    if save is not None:
        directory = Path(save)
        if directory.exists():
            pass#raise ValueError("directory already exists.")
        
        new_hf.with_format(None).save_to_disk(str(directory / "train"))
        new_hf.set_transform(hf_transform_to_torch)
        stats = compute_stats(new_dataset)

        info = {
            "codebase_version": CODEBASE_VERSION,
            "fps": orig_dataset.fps,
            "video": orig_dataset.video,
    }



        meta_data_dir = directory / "meta_data"
        
        new_hf.set_transform(hf_transform_to_torch)

        save_meta_data(info, stats, new_dataset.episode_data_index, meta_data_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default="data/aloha_act_awk_rollout_suc_filter")
    parser.add_argument('--load', default="aloha_act_awk_rollout")
    parser.add_argument('--rew_thresh', type=float, default=3.99)
    parser.add_argument('--max_episodes', type=int, default=999999)
    parser.add_argument('--match_max_episodes', action='store_true')
    args = parser.parse_args()

    data = LeRobotDataset(args.load, "data/")

    success_filter(data, save=args.save, rew_thresh=args.rew_thresh, max_episodes=args.max_episodes, match_max_episodes=args.match_max_episodes)