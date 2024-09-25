import itertools
from inspect import signature
import os
import pickle

import random
import torch
import numpy as np

def generate_combinations_loader(param_pool):
    """
    Generates all possible combinations of parameters given a dictionary 
    where the keys are parameter names and the values are lists of possible 
    values (or a single value).

    Args:
        param_pool (dict): A dictionary where each key is a parameter name and 
        each value is either a list of possible values for that parameter or a 
        single value.

    Yields:
        dict: A dictionary representing one combination of parameters.
    """
    keys, values = zip(*[
        (k, v if isinstance(v, list) else [v])
        for k, v in param_pool.items()
    ])
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))

def filter_param(param: dict, obj) -> dict:
    """
    Filters a dictionary of parameters to include only those that are valid 
    for the given object, based on its signature.

    Args:
        param (dict): A dictionary of parameters to be filtered.
        obj (callable): The object (e.g., a function or class) whose 
        signature will be used to filter the parameters.

    Returns:
        dict: A filtered dictionary containing only the parameters that are 
        valid for the given object.
    """
    init_params = signature(obj).parameters
    return {k: v for k, v in param.items() if k in init_params}

def generate_data_name(n_nodes: int, env_type: str, trajectory_length: int, num_desired_trajectories: int, args = None, seed = 70):
    name = f"data_n_nodes_{n_nodes}_env_{env_type}_traj_len_{trajectory_length}_n_traj_{num_desired_trajectories}_args_{args}_seed_{seed}.pickle"
    return name

# Function to set the random seed for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dataset_loader(directory):
    """
    A generator function that yields the contents of each .pickle file in the given directory.

    Parameters:
    - directory (str): The path to the directory containing the .pickle files.

    Yields:
    - data: The content of each .pickle file, one at a time.
    """
    for filename in os.listdir(directory):
        if filename.endswith('.pickle'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)  # Load the .pickle file
                yield data  # Yield the loaded data one by one

def matches_filter(allowed_values_dict, input_values_dict):
    # Iterate through the input values dictionary
    for key, value in allowed_values_dict.items():
        # If the key exists in allowed_values_dict and the value does not match
        if key in input_values_dict:
            if input_values_dict[key] not in value:
                return False
        else:
            return False
    # If all checks pass, return True
    return True