"""
Based on https://github.com/allenai/hidden-networks/blob/master/simple_mnist_example.py
"""
import torch
from torch import autograd, nn
from torch.nn import functional as F
import math


def init_weights(data, init='normal'):
    """
    Layer weight initializer for relu activation and fan_in mode
    """
    gain = nn.init.calculate_gain('relu')
    fan = nn.init._calculate_correct_fan(data, 'fan_in')
    std = gain / fan ** 0.5
    data_normal = torch.randn_like(data) * std
    if init == 'normal':
        return data_normal
    elif init == 'signed':
        return data_normal.sign() * std
    else:
        raise ValueError


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class SupermaskConv(nn.Conv2d):
    def __init__(self, sparsity, init_type, **kwargs):
        super().__init__(**kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        self.sparsity = sparsity
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        self.weight.data = init_weights(self.weight.data, init_type)
        self.weight.requires_grad = False

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), self.sparsity)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class SupermaskConvTranspose(nn.ConvTranspose2d):
    def __init__(self, sparsity, init_type, **kwargs):
        super().__init__(**kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        self.sparsity = sparsity
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")
        self.weight.data = init_weights(self.weight.data, init_type)
        self.weight.requires_grad = False

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), self.sparsity)
        w = self.weight * subnet
        x = F.conv_transpose2d(
            x, w, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation
        )
        return x


class SupermaskLinear(nn.Linear):
    def __init__(self, sparsity, init_type, **kwargs):
        super().__init__(**kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        self.sparsity = sparsity
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")
        self.weight.data = init_weights(self.weight.data, init_type)
        self.weight.requires_grad = False

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), self.sparsity)
        w = self.weight * subnet
        return F.linear(x, w, self.bias)
