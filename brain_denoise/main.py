from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch import nn
from brain_denoise.linear_net import LinearNet
from brain_denoise.data.simluator import simulate_data
import torch
import numpy as np
from torch.nn import MSELoss
from .train import train


def split_idx(n, splits=(0.6, 0.8, 1), shuffle=False):
    idx = torch.arange(n)
    if shuffle:
        idx = np.random.permutation(idx)

    train = idx[: int(splits[0] * n)]
    valid = idx[int(splits[0] * n) : int(splits[1] * n)]
    test = idx[int(splits[1] * n) : int(splits[2] * n)]
    return train, valid, test


if __name__ == "__main__":
    # Params
    ns, nc, nt = 1000, 50, 10
    bs = 10
    device = "cpu"

    # Data
    data_in, data_out, signal = simulate_data(
        ns, nc, nt, noise_types=["gaussian", "dirac"]
    )

    # Build loader
    dataset = TensorDataset(data_in, data_out)
    train, valid, test = split_idx(len(dataset))
    train_loader = DataLoader(dataset[train], batch_size=bs)
    valid_loader = DataLoader(dataset[valid], batch_size=bs)
    test_loader = DataLoader(dataset[test], batch_size=bs)

    # Initiate Model
    model = LinearNet(nc, nt)
    model.to(device)

    # Initiate Loss
    loss = MSELoss()

    # Intiate Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train
    train(train_loader, model, loss, optimizer)