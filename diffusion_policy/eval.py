"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import numpy as np
import h5py

class SteeredPolicy:

    def __init__(self, policy, classifier, num_samples, pred_theshold=None):
        from diffusion_policy.model.common.rotation_transformer import RotationTransformer
        
        self.classifier = classifier
        self.policy = policy
        self.dtype = policy.dtype
        self.device = policy.device
        self.num_samples = num_samples
        self.pred_threshold = pred_theshold

        self.rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

    def reset(self):
        self.policy.reset()


    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = torch.cat([pos, rot, gripper], dim=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction

    def predict_action(self, obs_dict):
        num_envs = obs_dict['obs'].shape[0]
        best_scores = None
        best_actions = None
        for _ in range(self.num_samples):
            od = obs_dict.copy()
            actions = self.policy.predict_action(od)
            actions = actions['action']
            with torch.inference_mode():
                env_actions = self.undo_transform_action(actions)
                shape = env_actions.shape
                preds = self.classifier(env_actions.view(-1, shape[-1]))
                preds = preds.view(shape[0], shape[1])
                if self.pred_threshold is not None:
                    preds = (preds > self.pred_threshold).float()
                preds = preds.sum(dim=1)

            if best_actions is None:
                best_actions = actions
                best_scores = preds
            else:
                mask = preds > best_scores
                best_actions[mask] = actions[mask]
                with torch.inference_mode():
                    best_scores[mask] = preds[mask].clone()

        return {'action':best_actions}

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', default='data/policy_vid')
@click.option('-d', '--device', default='cuda:0')
@click.option('-s', '--save', is_flag=True)
@click.option('--save_obs', type=bool, default=True)
@click.option('--seed', type=int, default=None)
@click.option('--classifier_ckpt', default=None)
@click.option('--num_rollouts', type=int, default=100)
@click.option('--action_samples', type=int, default=4)
@click.option('--pred_threshold', type=float, default=None)
def main(checkpoint, output_dir, device, save, seed, classifier_ckpt, num_rollouts, action_samples, pred_threshold, save_obs):
    #if os.path.exists(output_dir):
        #click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cfg.task.env_runner.n_test = num_rollouts
    cfg.task.env_runner.render_hw = [640, 640]
    cfg.task.env_runner.n_test_vis = 4
    cfg.task.env_runner.n_train = 0
    cfg.task.env_runner.n_train_vis = 0
    if seed is not None:
        cfg.task.env_runner.test_start_seed = seed
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    if classifier_ckpt is not None:
        from classifier import ClassifierMLP
        classifier = ClassifierMLP(7, [8,8]).to(device)
        classifier.load_state_dict(torch.load(classifier_ckpt))
        classifier.eval()

        policy = SteeredPolicy(policy, classifier, action_samples, pred_threshold)
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy, save_trajectory=save)

    if save:
        trajectories=runner_log.pop('trajectories')
        success_labels=runner_log.pop('success_array')
        traj_obs = runner_log.pop('traj_obs')
        with h5py.File(os.path.join(output_dir, 'rollouts.hdf5'), 'w') as f:
            if not save_obs:
                for i in range(len(trajectories)):
                    traj_data = f.create_dataset(f'rollout_{i}', data=trajectories[i])
                    traj_data.attrs['succes'] = success_labels[i]
            else:
                group = f.create_group('data')
                for i in range(len(trajectories)):
                    rollout_group = group.create_group(f'rollout_{i}')
                    rollout_group.attrs['success'] = success_labels[i]
                    traj_len = len(traj_obs[i]) - 1
                    rollout_group.create_dataset('actions', data=trajectories[i][:traj_len])
                    obs_group = rollout_group.create_group('obs')
                    for k in traj_obs[i][0]:
                        obs_group.create_dataset(str(k), data=[traj_obs[i][j][str(k)] for j in range(traj_len)])
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

    

if __name__ == '__main__':
    main()
