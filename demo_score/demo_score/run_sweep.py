import multiprocessing
import time
import os
import sys

from .train_trajwise_classifier import train as train_trajwise
from .util import set_global_seed
from .train_stepwise_classifier import train as train_stepwise



def run_one(kwargs):
    set_global_seed(int(kwargs['run_name'].split('/')[-1]))
    print("Now running", kwargs['run_name'])
    if 'stepwise' in kwargs['run_name']:
        train_stepwise(**kwargs)
    else:
        train_trajwise(**kwargs)


def is_folder_empty(folder_path):
    # List the contents of the folder
    # os.listdir() returns an empty list if the folder is empty
    return len(os.listdir(folder_path)) == 0

def target_func(kwargs):
    print('here')
    path = os.path.join(kwargs['log_root'], kwargs['run_name'])
    print(path)
    if not os.path.isdir(path):
        print("path", path)
        os.makedirs(path)
    else:    
        print(f"Skipping {kwargs['run_name']} because it already exists")
        time.sleep(3)
        return 
    run_one(kwargs)        
    time.sleep(3)

def sweep(sweep_func, num_proc, filter_keywords=[], debug=False, max_to_run=None):

    args_list = sweep_func()
    filtered_args = []

    for arg_list in args_list:
        # Check for filter keywords
        if filter_keywords is not None:
            do_run = all(fk in arg_list['run_name'] for fk in filter_keywords)
            if not do_run:
                continue
        filtered_args.append(arg_list)

    if max_to_run is not None:
        filtered_args = filtered_args[:max_to_run]

    if debug or num_proc == 1:
        for arg in filtered_args:
            target_func(arg)
            if debug:
                break
    else:
        multiprocessing.set_start_method('spawn')
        with multiprocessing.Pool(processes=num_proc) as pool:
            pool.map(target_func, filtered_args)
    
