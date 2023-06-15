import torch
import random
import numpy as np
from types import FunctionType


def undeco(func):
    if func.__closure__:
        for cell in func.__closure__:
            if isinstance(cell.cell_contents, FunctionType):
                return cell.cell_contents
    return func


def check_data(name):
    """
    Check whether the dataset is available.
    """
    if name not in ['Web30K', 'Yahoo', 'Istella']:
        raise ValueError("Unknown dataset name. Please choose in ['Web30K', 'Yahoo', 'Istella']")


def check_device():
    """
    Check available accelerator device.
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    return device


def set_seed(random_seed):
    """
    Set random seed.
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
