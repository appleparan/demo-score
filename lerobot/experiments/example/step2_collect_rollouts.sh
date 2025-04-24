#!/bin/bash

#Collect and save rollouts from several checkpoints of the base policy

#Run from lerobot directory using demoscore environment


#NOTE: using seed=12000 so that env is initialized differently than during policy training (which used seed=10000)

eps=("070000" "150000" "220000" "300000")
for ep in "${eps[@]}"; do
    MUJOCO_GL=egl python lerobot/scripts/collect_data.py -p data/example/policy_logs/ex1_seed10000_base_policy/checkpoints/$ep/pretrained_model/ --out-dir data/example/policy_logs/ex1_seed10000_base_policy/rollouts/ep$ep eval.n_episodes=100 eval.batch_size=100 seed=12000
done