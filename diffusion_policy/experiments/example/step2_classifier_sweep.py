#This file performs a sweep of classifier training runs
#This file should be run using the demoscore environment from the lerobot folder


from itertools import product
import random


## MODEL CONFIGS
#Uncommenting model arch config adds it to the classifier sweep

model_dicts = {}
#model_dicts["small_transformer"] = {"model_dim": 8, "dim_feedforward": 8, "num_heads": 2, "num_layers": 2, "classifier_type": "transformer"}
#model_dicts["med_transformer"] = {"model_dim": 16, "dim_feedforward": 16, "num_heads": 2, "num_layers": 2, "classifier_type": "transformer"}
#model_dicts["big_transformer"] = {"model_dim": 32, "dim_feedforward": 32, "num_heads": 4, "num_layers": 3, "classifier_type": "transformer"}

# MLPs
model_dicts['small_stepwise'] = {'hidden_sizes': [8, 8]}
#model_dicts['small_deep_stepwise'] = {'hidden_sizes': [8, 8, 8]}
#model_dicts['med_stepwise'] = {'hidden_sizes': [16, 16]}
#model_dicts['med_deep_stepwise'] = {'hidden_sizes': [16, 16, 16]}
#model_dicts['large_stepwise'] = {'hidden_sizes': [32, 32]}

####


## ROLLOUT EPOCH CHOICES ##
#Rollouts should have been collected from multiple policy checkpoints at this point.
#Here we specify which sets of rollouts from which policy training epochs should be used for training and cross-validation of the classifier

eps_dict = {}

eps_dict['ex1_seed10000'] = {
    "train": [250, 500, 750, 1000],
    "cross_val": [1000,]
}

## ## ##

def example_sweep():
    runs = []

    tr_dataset_size = 100
    cross_val_dataset_size = 100
    self_val_dataset_size = 0 # Refers to drawing validation examples from same epoch as train
    seed = 1
    weight_decay = 0.1
    lr = 1e-4

    for model_arch in model_dicts.keys():
        model_arch = str(model_arch)
        for name in eps_dict.keys():
            name = str(name)
            for tr_ep, val_ep in product(eps_dict[name]['train'], eps_dict[name]['cross_val']):
                run = {'steps': 100, "num_val":self_val_dataset_size,
                        'format': 'robodiff', 'data_root': '.', 
                        'dataset_size': tr_dataset_size, 'ckpt_val_dataset_size': cross_val_dataset_size,
                        "log_root": "data/example/classifier_logs", 'lr' : lr}
                run['run_name'] = f"{name}/train_dataset_size_{tr_dataset_size}/val_dataset_size_{cross_val_dataset_size}/train_ep{tr_ep}_val_ep{val_ep}/{model_arch}/weight_decay_{weight_decay}/lr_{lr}/{seed}"
                run['data_dir'] = f"example/policy_logs/{name}_base_policy/rollouts/ep{tr_ep}"
                run['ckpt_val_dataset'] = f"example/policy_logs/{name}_base_policy/rollouts/ep{val_ep}"
                run.update(model_dicts[model_arch])
                runs.append(run)
                if 'stepwise' in model_arch:
                    run.pop('steps')
    random.shuffle(runs)
    return runs

if __name__ == '__main__':
    from demo_score.demo_score.run_sweep import sweep

    runs = example_sweep()
    sweep(runs)