import torch


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
