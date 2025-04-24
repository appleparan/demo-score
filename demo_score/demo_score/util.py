import torch
import random
import numpy as np


    
def set_global_seed(seed):
    # Set the seed for Python's random module
    random.seed(seed)
    
    # Set the seed for NumPy
    np.random.seed(seed)
    
    # Set the seed for PyTorch (CPU)
    torch.manual_seed(seed)
    
    # Set the seed for PyTorch (GPU, if using CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    
    # Ensure deterministic behavior (useful in debugging but can reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False