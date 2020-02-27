import torch
from torch import nn
from torch.nn import functional as F
from .layers import SupermaskConv, SupermaskLinear


class SimpleNet(nn.Module):
    def __init__(self, num_classes, sparsity):
        super(SimpleNet, self).__init__()
        self.conv1 = SupermaskConv(sparsity=sparsity, in_channels=3, out_channels=32,
                                   kernel_size=3, stride=1, bias=False)
        self.conv2 = SupermaskConv(sparsity=sparsity, in_channels=32, out_channels=64,
                                   kernel_size=3, stride=1, bias=False)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = SupermaskLinear(sparsity=sparsity, in_features=12544, out_features=128, bias=False)
        self.fc2 = SupermaskLinear(sparsity=sparsity, in_features=128, out_features=num_classes, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
