#!/bin/bash

#This file trains a new policy on the filtered dataset
#Run from lerobot directory using demoscore environment


# ONLY DEMOS (only demos from the original demonstration set in the new, filtered dataset)
MUJOCO_GL=egl python lerobot/scripts/train.py policy=act env=aloha wandb.enable=True hydra.job.name=ex1_seed10000_demo_score_only_demos training.offline_steps=300000 training.eval_freq=10000 training.save_freq=10000 seed=14000 training.log_freq=100 eval.n_episodes=256 eval.batch_size=64 wandb.project=demoscore hydra.run.dir=data/example/policy_logs/ex1_seed10000_demo_score_only_demos +datasets_file=data/example/demo_score_datasets/ex1_seed10000/demo_score_demos_only.txt


# DEMOS + ROLLOUTS (both demos from the original demonstration set AND rollouts collected from the base policy in the new, filtered dataset)
MUJOCO_GL=egl python lerobot/scripts/train.py policy=act env=aloha wandb.enable=True hydra.job.name=ex1_seed10000_demo_score_rollouts_plus_demos training.offline_steps=300000 training.eval_freq=10000 training.save_freq=10000 seed=14000 training.log_freq=100 eval.n_episodes=256 eval.batch_size=64 wandb.project=demoscore hydra.run.dir=data/example/policy_logs/ex1_seed10000_demo_score_rollouts_plus_demos +datasets_file=data/example/demo_score_datasets/ex1_seed10000/demo_score_rollouts_plus_demos.txt