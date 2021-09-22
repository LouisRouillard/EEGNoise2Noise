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
