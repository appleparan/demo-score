#!/bin/bash

# Note that using seed 12000 (instead of seed 10000 as for initial policy to avoid using overlapping seeds for environment initialization to ensure the domain randomization differs between the new policy and the episodes from the old policy used to train this policy.)




# ONLY DEMOS (only demos from the original demonstration set in the new, filtered dataset)
dataset="data/example/demos/ex1_seed10000/demo_score_datasets/ex1_seed10000/demo_score_demos_only.hdf5"

HYDRA_FULL_ERROR=1 python train.py --config-dir=experiments/example --config-name=lowdim_square_diffpolicy_cnn_square_example.yaml logging.name=ex1_seed10000_demo_score_only_demos training.device=cuda:0 task.dataset.dataset_path=$dataset task.dataset_path=$dataset task.env_runner.dataset_path=$dataset hydra.run.dir='data/example/policy_logs/${logging.name}' training.seed=12000 task.dataset.seed=12000 task.env_runner.test_start_seed=12000 task.env_runner.n_envs=32





# DEMOS + ROLLOUTS (both demos from the original demonstration set AND rollouts collected from the base policy in the new, filtered dataset)
dataset="data/example/demos/ex1_seed10000/demo_score_datasets/ex1_seed10000/demo_score_rollouts_plus_demos.hdf5"

HYDRA_FULL_ERROR=1 python train.py --config-dir=experiments/example --config-name=lowdim_square_diffpolicy_cnn_square_example.yaml logging.name=ex1_seed10000_demo_score_rollouts_plus_demos training.device=cuda:0 task.dataset.dataset_path=$dataset task.dataset_path=$dataset task.env_runner.dataset_path=$dataset hydra.run.dir='data/example/policy_logs/${logging.name}' training.seed=12000 task.dataset.seed=12000 task.env_runner.test_start_seed=12000 task.env_runner.n_envs=32
