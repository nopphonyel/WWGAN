import torch
from torch import nn


class MoGLayer(nn.Module):

    def __init__(self, noise_dim: tuple):
        super(MoGLayer, self).__init__()

        pre_std = torch.zeros(noise_dim)
        pre_std = torch.nn.init.uniform_(pre_std, -0.2, 0.2)
        self.std = nn.Parameter(pre_std, requires_grad=True)

        pre_mean = torch.zeros(noise_dim)
        pre_mean = torch.nn.init.uniform_(pre_mean, -1.0, 1.0)
        self.mean = nn.Parameter(pre_mean, requires_grad=True)

    def forward(self, noise, DEV):
        return self.mean.to(DEV) + (self.std.to(DEV) * noise.to(DEV))
