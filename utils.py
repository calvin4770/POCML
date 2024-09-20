import itertools
from inspect import signature

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

def generate_data_name(n_nodes: int, env: str, trajectory_length: int, num_desired_trajectories: int, args = None, seed = 70):
    name = f"data_n_nodes_{n_nodes}_env_{env}_traj_len_{trajectory_length}_n_traj_{num_desired_trajectories}_args_{args}_seed_{seed}.pickle"
    return name

def generate_run_name(params):
    # TODO cutomize run name
    name = f"tree_sdim_{params['state_dim']}_rfdim_{params['random_feature_dim']}_a_{params['alpha']}_bypass_{params['memory_bypass']}_usgo_{params['update_state_given_obs']}"
    return name

# Function to set the random seed for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
