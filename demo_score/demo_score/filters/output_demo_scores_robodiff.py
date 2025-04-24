import h5py
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def output_demo_scores(demo_file, model, output, steps=None, threshold= 0.5, max_steps=None, num_eps_per_file=None):

    model.eval()

    assert isinstance(demo_file, str)

    demo_scores = {}

    with h5py.File(demo_file, 'r') as f_old:
        
        old_demos = f_old['data']
        episode_key = 'demo' if 'demo_0' in old_demos else 'rollout'
        if steps is not None:
            steps = min(steps, min([old_demos[f'{episode_key}_{i}/actions'][:].shape[0] for i in range(len(old_demos))]))

        for i in range(len(old_demos)):
            if num_eps_per_file is not None:
                if i > num_eps_per_file - 1:
                    continue

            inputs = old_demos[f'{episode_key}_{i}/actions'][:]
            if max_steps is not None:
                if inputs.shape[0] > max_steps:
                    print("hit max_steps crit")
                    inputs = inputs[:max_steps]
            inputs = torch.tensor(inputs, dtype=torch.float).to(device)[:steps]
            inputs = inputs.unsqueeze(0)
            with torch.inference_mode():
                pred = model(inputs)

            demo_scores[f'demo_{i}'] = pred.squeeze(2).squeeze(0).detach().cpu().numpy()

        print(demo_scores)

        np.savez(output, threshold=float(threshold), **demo_scores)