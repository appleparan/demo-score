import os
import time
import h5py
import torch
import numpy as np
from dataset import StepwiseClassifierDataset
from models import StepwiseMLPClassifier



def filter_demos(demo_folder, model, output_folder, threshold):
    """
        For each timestep:
        observations
        - images
            - cam_high          (480, 640, 3) 'uint8'
            - cam_low           (480, 640, 3) 'uint8'
            - cam_left_wrist    (480, 640, 3) 'uint8'
            - cam_right_wrist   (480, 640, 3) 'uint8'
        - qpos                  (14,)         'float64'
        - qvel                  (14,)         'float64'
        
        action                  (14,)         'float64'
        """
    model.eval()
    os.makedirs(output_folder, exist_ok=True)
    print("demo_folder", demo_folder)

    dataset = StepwiseClassifierDataset(data_dir=demo_folder,
                                        data_root='/',
                                        format='aloha')

    new_ep = 0
    for ep in range(dataset.get_num_eps()):
        input, label = dataset.get_episode(ep)
        input = input.cuda()
        with torch.inference_mode():
            pred = model(input).mean()
        if pred > threshold:
            print(f"Taking epoch {ep}")
            os.system(f"cp {demo_folder}/episode_{ep}.hdf5 {output_folder}/episode_{new_ep}.hdf5")

            new_ep += 1
    
    return True

