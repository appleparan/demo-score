from classifier import *

@click.command()
@click.option('--demo_data', required=True)
@click.option('--output', required=True)
@click.option('-c', '--checkpoint', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('-t', '--threshold', type=float, required=True)
def main(demo_data, output, checkpoint, device, threshold):

    classifier = ClassifierMLP(7, [8,8]).to(device)
    classifier.load_state_dict(torch.load(checkpoint))
    classifier.eval()

    new_idx = 0
    with h5py.File(output, 'w') as f_new:
        with h5py.File(demo_data, 'r') as f_old:
            old_demos = f_old['data']
            new_demos = f_new.create_group('data')

            for i in range(len(old_demos)):
                inputs = old_demos[f'demo_{i}/actions'][:]
                inputs = torch.tensor(inputs, dtype=torch.float).to(device)
                with torch.inference_mode():
                    pred = classifier(inputs).mean()
                if pred > threshold:
                    #new_demos.create_group(f'demo_{new_idx}')
                    old_demos.copy(old_demos[f'demo_{i}'], new_demos, name=f'demo_{new_idx}')
                    new_idx += 1
        
            for k in f_old['data'].attrs.keys():
                f_new['data'].attrs[k] = f_old['data'].attrs[k]

if __name__ == '__main__':
    main()