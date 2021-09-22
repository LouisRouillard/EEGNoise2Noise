# %%
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch import nn
from brain_denoise.models.linear_net import LinearNet
from brain_denoise.models.modules import UNet1D
from brain_denoise.data.simulator import simulate_data
from brain_denoise.visualizers.time_series import plot_signals
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import MSELoss
from brain_denoise.train import train_eval_model, run_epoch
from brain_denoise.utils import split_idx

# %%
# Params
ns, nc, nt = 1000, 3, 128
bs = 10
device = "cpu"

# Data
data_in, data_out, signal = simulate_data(
    ns, nc, nt, noise_type=["gaussian", "step"]
)

# Build loader
dataset = TensorDataset(data_in, data_out)
testset = TensorDataset(data_in, signal)
train, valid, test = split_idx(len(dataset))
train_loader = DataLoader(dataset[train], batch_size=bs)
valid_loader = DataLoader(dataset[valid], batch_size=bs)
test_loader = DataLoader(testset[test], batch_size=bs)

# Initiate Model
model = UNet1D(
    time_length=nt,
    in_channels=nc,
    hidden_channels=[16, 32]
)
model.to(device)

# Initiate Loss
loss = MSELoss()

# Intiate Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-2,
    weight_decay=1e-3
)

# Train
# %%
train_eval_model(
    train_loader=train_loader,
    model=model,
    loss_fn=loss,
    optimizer=optimizer,
    n_epochs=100,
    valid_loader=valid_loader,
    test_loader=test_loader,
    device="cpu"
)

final_loss = run_epoch(
    dataloader=test_loader, 
    model=model, 
    loss_fn=loss, 
    device="cpu", 
    train=False,
    optimizer=None,
    n_epochs=100,
)
print(f"Final test loss : {final_loss:>3f}")
# %%

data_pred = model(data_in)

# %%

viz_idx = 2
plot_signals(
    true_signal=signal[test][:viz_idx].detach().numpy(),
    noisy_signal=data_in[test][:viz_idx].detach().numpy(),
    pred_signal=data_pred[test][:viz_idx].detach().numpy()
)

