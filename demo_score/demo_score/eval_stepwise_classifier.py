from train_stepwise_classifier import *
import os
if False:
    from .train_stepwise_classifier import *



def eval_single_classifier_epoch(data_dir, data_root, eval_dataset, classifier_dir, format='robodiff'):

    with open(os.path.join(classifier_dir, 'best_ep.txt'), 'r') as file:
        # Read the line from the file
        line = file.readline()
        
        # Split the line into parts
        parts = line.split()
        
        # Convert the parts to int and float
        ep = int(parts[0])
        thresh = float(parts[1])

    print(ep, thresh)
    model = StepwiseMLPClassifier(7, [8, 8], 0.3)
    model_file = os.path.join(classifier_dir, f"model_{ep}.pth")
    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))

    train_dataset = StepwiseClassifierDataset(data_dir=data_dir,
                                        data_root=data_root,
                                        format=format,)
    
    eval_dataset = StepwiseClassifierDataset(eval_dataset, data_root=data_root, format=format)

    model.eval()

    tr_res = eval_rollouts(train_dataset, model, "Train", use_mode_oracle=False)
    thresholds = tr_res.pop("thresholds")

    eval_res = eval_rollouts(eval_dataset, model, "Eval", use_mode_oracle=False, tresholds=thresholds)

    print(thresholds, tr_res, eval_res)


if __name__ == "__main__":

    eval_single_classifier_epoch('sweep2/policy_logs/ph100_side100/rollouts/ep200.hdf5',
                                data_root='/diffusion_policy/data/',
                                 eval_dataset='sweep2/demos/ph100_side100_seed10000/demos.hdf5',
                                 classifier_dir='diffusion_policy/data/sweep2.2/classifier_logs/ph100_side100_seed10000/dataset_size_256/train_ep200_val_ep250/small_stepwise/weight_ratio_4/3/')