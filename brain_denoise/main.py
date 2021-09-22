# %%
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch import nn
from models.linear_net import LinearNet
from models.modules import UNet1D
from data.simulator import simulate_data
from visualizers.time_series import plot_signals
import torch
import numpy as np
from torch.nn import MSELoss
from train import train as train_model
from run_eval import run_eval

# %%
def split_idx(n, splits=(0.6, 0.8, 1), shuffle=False):
    idx = torch.arange(n)
    if shuffle:
        idx = np.random.permutation(idx)

    train = idx[: int(splits[0] * n)]
    valid = idx[int(splits[0] * n) : int(splits[1] * n)]
    test = idx[int(splits[1] * n) : int(splits[2] * n)]
    return train, valid, test

# %%
# Params
ns, nc, nt = 1000, 3, 128
bs = 10
device = "cpu"

# Data
data_in, data_out, signal = simulate_data(
    ns, nc, nt, noise_types=["gaussian", "dirac"]
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
train_model(
    train_loader,
    model,
    loss,
    optimizer,
    validloader=valid_loader,
    testloader=test_loader,
    n_epochs=500
)
final_loss = run_eval(test_loader, model, loss)
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

# %%
