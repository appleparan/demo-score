import h5py
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def filter_demos(demo_file, model, output, steps=None, threshold=0.5, metadata={}, remove_failed_rollouts=True, max_steps=None, num_eps_per_file=None, arch_type = 'stepwise'):

    model.eval()

    metadata['meta.demo_file'] = demo_file
    metadata['meta.threshold'] = threshold
    metadata['meta.output'] = output

    print('max_steps', max_steps)

    if not isinstance(demo_file, list):
        demo_file = [demo_file,]

    if not isinstance(num_eps_per_file, list):
        num_eps_per_file = [num_eps_per_file,]*len(demo_file)

    new_idx = 0
    with h5py.File(output, 'w') as f_new:
        new_demos = f_new.create_group('data')
        for df_idx, iter_demo_file in enumerate(demo_file):
            with h5py.File(iter_demo_file, 'r') as f_old:
                
                old_demos = f_old['data']
                episode_key = 'demo' if 'demo_0' in old_demos else 'rollout'
                if steps is not None:
                    steps = min(steps, min([old_demos[f'{episode_key}_{i}/actions'][:].shape[0] for i in range(len(old_demos))]))

                
                for k, v in metadata.items():
                    new_demos.attrs[str(k)] = str(v)

                for i in range(len(old_demos)):

                    if num_eps_per_file[df_idx] is not None:
                        if i > num_eps_per_file[df_idx] - 1:
                            continue

                    print(iter_demo_file, i)
                    if 'success' in old_demos[f'{episode_key}_{i}'].attrs:
                        if (not old_demos[f'{episode_key}_{i}'].attrs['success']) and remove_failed_rollouts:
                            print("Skipping failed rollout", f'{episode_key}_{i}')
                            continue #remove failed rollouts

                    inputs = old_demos[f'{episode_key}_{i}/actions'][:]
                    if max_steps is not None:
                        if inputs.shape[0] > max_steps:
                            print("hit max_steps crit")
                            inputs = inputs[:max_steps]
                    inputs = torch.tensor(inputs, dtype=torch.float).to(device)[:steps]
                    inputs = inputs.unsqueeze(0)
                    with torch.inference_mode():
                        pred = model(inputs).squeeze(0)
                    #print(pred)
                    if len(pred.shape) > 0:
                        if pred.shape[0] > 1:
                            pred = pred.mean()
                    if pred > threshold:
                        old_demos.copy(old_demos[f'{episode_key}_{i}'], new_demos, name=f'demo_{new_idx}')
                        new_idx += 1
                        print("Taking with classifier:", f'{episode_key}_{i}', iter_demo_file)
                    else:
                        print("Rejecting with classifier:", f'{episode_key}_{i}', iter_demo_file)
            
                for k in f_old['data'].attrs.keys():
                    f_new['data'].attrs[k] = f_old['data'].attrs[k]

