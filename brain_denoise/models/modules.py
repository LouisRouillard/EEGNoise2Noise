# %%
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
class UNet1D(nn.Module):

    def __init__(
        self,
        time_length: int = 32,
        in_channels: int = 1,
        hidden_channels: List[int] = [4, 8, 16],
        **kwargs
    ):
        """Unet architecture

        Parameters
        ----------
        time_length : int, optional
            by default 32
        channels : int, optional
            by default 1
        """
        super().__init__(**kwargs)

        self.n_conv_steps = len(hidden_channels)
        effective_channels = [in_channels] + hidden_channels
        reduced_time_length = int(time_length / (2**self.n_conv_steps))
        hidden_features = reduced_time_length * hidden_channels[-1]
        self.hidden_shape = (-1, hidden_channels[-1], reduced_time_length)

        self.convs = []
        self.pools = []
        self.deconvs = []
        self.depools = []

        for idx in range(self.n_conv_steps):
            self.convs.append(
                nn.Conv1d(
                    in_channels=effective_channels[idx],
                    out_channels=effective_channels[idx + 1],
                    kernel_size=3,
                    padding=1
                )
            )
            self.pools.append(
                nn.MaxPool1d(
                    kernel_size=2
                )
            )
        
        self.linear = nn.Linear(
            in_features=hidden_features,
            out_features=hidden_features
        )

        for idx in range(self.n_conv_steps):
            self.deconvs.append(
                nn.Conv1d(
                    in_channels=effective_channels[-1 - idx],
                    out_channels=effective_channels[-2 - idx],
                    kernel_size=3,
                    padding=1
                )
            )
            self.depools.append(
                nn.Upsample(
                    scale_factor=2
                )
            )
        
        self.output = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Outputs datum of same shape

        Parameters
        ----------
        x : torch.Tensor
            (B, C, T)

        Returns
        -------
        torch.Tensor
            (B, C, T)
        """

        z = x

        for idx in range(self.n_conv_steps):
            z = F.relu(self.convs[idx](z))
            z = self.pools[idx](z)


        z = torch.flatten(
            z,
            start_dim=1,
            end_dim=-1
        )

        z = F.relu(self.linear(z))
        z = z.reshape(
            self.hidden_shape
        )

        for idx in range(self.n_conv_steps):
            z = self.depools[idx](z)
            z = F.relu(self.deconvs[idx](z))
  
        z = self.output(z)

        return z

# %%
nt, nc = 10, 4
model = UNet1D(
    time_length=nt,
    in_channels=nc,
    hidden_channels=[4, 8]
)
# %%
model(torch.zeros((1, nc, nt)))
# %%
