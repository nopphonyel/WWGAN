import torch
from torch import nn


class MoGLayer(nn.Module):

    def __init__(self, noise_dim: tuple):
        """

        :param noise_dim: The noise dimension
        """
        super(MoGLayer, self).__init__()

        pre_std = torch.zeros(noise_dim)
        pre_std = torch.nn.init.uniform_(pre_std, -0.2, 0.2)
        self.std = nn.Parameter(pre_std, requires_grad=True)

        pre_mean = torch.zeros(noise_dim)
        pre_mean = torch.nn.init.uniform_(pre_mean, -1.0, 1.0)
        self.mean = nn.Parameter(pre_mean, requires_grad=True)

    def to(self, *args):
        """
        Just override a bit to move the parameters
        :param args: Expected to be a device name
        :return: Nothing
        """
        super(MoGLayer, self).to(args[0])
        self.mean = self.mean.to(args[0])
        self.std = self.std.to(args[0])

    def forward(self, noise):
        return self.mean + (self.std * noise)
