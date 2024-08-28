import itertools
from inspect import signature

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