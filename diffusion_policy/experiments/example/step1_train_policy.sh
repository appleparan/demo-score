#!/bin/bash

name="ex1"
seed=10000

dataset="data/example/demos/${name}_seed${seed}/demos.hdf5"

HYDRA_FULL_ERROR=1 python train.py --config-dir=experiments/example --config-name=lowdim_square_diffpolicy_cnn_square_example.yaml logging.name=${name}_seed${seed}_base_policy training.device=cuda:0 task.dataset.dataset_path=$dataset task.dataset_path=$dataset task.env_runner.dataset_path=$dataset hydra.run.dir='data/example/policy_logs/${logging.name}' training.seed=$seed task.dataset.seed=$seed task.env_runner.test_start_seed=$seed task.env_runner.n_envs=32
