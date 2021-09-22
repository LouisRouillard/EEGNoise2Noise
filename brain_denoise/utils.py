import numpy as np

import torch


def split_idx(n, splits=(0.6, 0.8, 1), shuffle=False):
    idx = torch.arange(n)
    if shuffle:
        idx = np.random.permutation(idx)

    train = idx[: int(splits[0] * n)]
    valid = idx[int(splits[0] * n) : int(splits[1] * n)]
    test = idx[int(splits[1] * n) : int(splits[2] * n)]
    return train, valid, test


def set_torch_seed(seed):
    """Set torch's seed."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)