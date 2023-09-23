import torch

def select_device(cuda_use: bool = True) -> torch.device:
    if cuda_use:
        return torch.device('cuda' if cuda_use and torch.cuda.is_available() else 'cpu')
    else:
        return torch.device('cpu')