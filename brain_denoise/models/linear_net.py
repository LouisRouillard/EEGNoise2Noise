from torch import nn


class LinearNet(nn.Module):
    def __init__(self, nc, nt):
        super().__init__()
        self.linear = nn.Linear(nc * nt, nc * nt)
        self.relu = nn.ReLU()

    def forward(self, X):
        bs, _, _ = X.shape
        out = X.reshape(bs, -1)
        out = self.linear(out)
        out = self.relu(out)
        out = out.reshape(X.shape)
        return out