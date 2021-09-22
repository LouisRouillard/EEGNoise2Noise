from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch import nn
import torch
import numpy as np
from torch.nn import MSELoss
from train import train_eval_model, run_epoch
from utils import split_idx

from brain_denoise.models.linear_net import LinearNet
from brain_denoise.data.simulator import simulate_data


if __name__ == "__main__":
    # Params
    ns, nc, nt = 1000, 50, 16
    bs = 10
    device = "cpu"
    n_epochs = 20

    # Data
    data_in, data_out, signal = simulate_data(
        ns, nc, nt, noise_types=["gaussian", "dirac"]
    )

    # Build loader
    dataset = TensorDataset(data_in, data_out)
    testset = TensorDataset(data_in, signal)
    train, valid, test = split_idx(len(dataset), splits=(0.6, 0.8, 1))
    train_loader = DataLoader(dataset[train], batch_size=bs)
    valid_loader = DataLoader(dataset[valid], batch_size=bs)
    test_loader = DataLoader(testset[test], batch_size=bs)

    # Instantiate Model
    model = LinearNet(nc, nt)
    model.to(device)

    # Initiate Loss
    loss = MSELoss(reduce=None)

    # Instantiate Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train
    train_eval_model(
        train_loader=train_loader,
        model=model,
        loss_fn=loss,
        optimizer=optimizer,
        valid_loader=valid_loader,
        test_loader=test_loader,
        device=device,
        n_epochs=10,
    )

    final_loss = run_epoch(
        dataloader=test_loader, 
        model=model, 
        loss_fn=loss, 
        device="cpu", 
        train=False,
        optimizer=None,
        n_epochs=10,
    )
    print(f"Final test loss : {final_loss:>3f}")
