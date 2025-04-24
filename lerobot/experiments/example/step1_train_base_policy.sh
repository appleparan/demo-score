#!/bin/bash

#Train base policy with entire demonstration set

#Run from lerobot directory using demoscore environment

#Put your or symbolicly link your dataset to data/example/demos/ex1

MUJOCO_GL=egl python lerobot/scripts/train.py policy=act env=aloha wandb.enable=True hydra.job.name=ex1_seed10000_base_policy training.offline_steps=300000 training.eval_freq=10000 training.save_freq=10000 seed=10000 training.log_freq=100 eval.n_episodes=256 eval.batch_size=64 wandb.project=demoscore hydra.run.dir=data/example/policy_logs/ex1_seed10000_base_policy +datasets_file=data/example/demos/ex1