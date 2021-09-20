# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
class UNet1D(nn.Module):

    def __init__(
        self,
        time_length: int = 32,
        channels: int = 1,
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

        n_channels_1 = 4
        n_channels_2 = 8
        n_channels_3 = 16
        self.reduced_time_length = time_length / 8
        n_channels_4 = self.reduced_time_length * n_channels_2

        self.conv_1 = nn.Conv1d(
            in_channels=channels,
            out_channels=n_channels_1,
            kernel_size=3
        )
        self.pool_1 = nn.MaxPool1d(
            kernel_size=2
        )
        self.conv_2 = nn.Conv1d(
            in_channels=n_channels_1,
            out_channels=n_channels_2,
            kernel_size=3
        )
        self.pool_2 = nn.MaxPool1d(
            kernel_size=2    
        )
        self.conv_3 = nn.Conv1d(
            in_channels=n_channels_2,
            out_channels=n_channels_3,
            kernel_size=3
        )
        self.pool_3 = nn.MaxPool1d(
            kernel_size=2  
        )

        self.conv_4 = nn.Conv1d(
            in_channels=n_channels_3,
            out_channels=n_channels_3,
            kernel_size=1
        )

        self.up_sample_1 = nn.Upsample(
            scale_factor=2
        )
        self.deconv_1 = nn.ConvTranspose1d(
            in_channels=n_channels_3,
            out_channels=n_channels_2,
            kernel_size=3
        )        
        self.up_sample_2 = nn.Upsample(
            scale_factor=2
        )
        self.deconv_2 = nn.ConvTranspose1d(
            in_channels=n_channels_2,
            out_channels=n_channels_1,
            kernel_size=3
        )
        self.up_sample_3 = nn.Upsample(
            scale_factor=2
        )
        self.deconv_3 = nn.ConvTranspose1d(
            in_channels=n_channels_1,
            out_channels=channels,
            kernel_size=3
        )

        self.up_sample_4 = nn.Upsample(
            size=time_length
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

        z = F.relu(self.conv_1(x))
        z = self.pool_1(z)

        z = F.relu(self.conv_2(z))
        z = self.pool_2(z)

        z = F.relu(self.conv_3(z))
        z = self.pool_3(z)

        z = F.relu(self.conv_4(z))

        z = self.up_sample_1(z)
        z = F.relu(self.deconv_1(z))

        z = self.up_sample_2(z)
        z = F.relu(self.deconv_2(z))

        z = self.up_sample_3(z)
        z = F.relu(self.deconv_3(z))

        y = self.up_sample_4(z)

        return y

# %%

batch_size = 1
time_length = 32
channels = 4

# %%
model = UNet1D(
    time_length=time_length,
    channels=channels
)

# %%

x = torch.zeros(
    (batch_size, channels, time_length)
)

y = model(x)

y.shape
# %%

# %%
