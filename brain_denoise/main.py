from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch import nn
from models.linear_net import LinearNet
from data.simulator import simulate_data
import torch
import numpy as np
from torch.nn import MSELoss
from train import train as train_model
from run_eval import run_eval
from utils import split_idx

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

    # Initiate Model
    model = LinearNet(nc, nt)
    model.to(device)

    # Initiate Loss
    loss = MSELoss(reduce=None)

    # Intiate Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train
    train_model(
        train_loader,
        model,
        loss,
        optimizer,
        validloader=valid_loader,
        testloader=test_loader,
        n_epochs=n_epochs,
    )
    final_loss = run_eval(test_loader, model, loss)
    print(f"Final test loss : {final_loss:>3f}")