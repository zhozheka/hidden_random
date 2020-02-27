"""
Based on https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""
import torch
import torch.nn as nn
from .layers import SupermaskLinear, SupermaskConv


class VGG(nn.Module):
    def __init__(self, num_classes, features, sparsity=None, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.num_classes = num_classes
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.sparsity = sparsity
        self.classifier = nn.Sequential(
            #nn.Linear(512 * 7 * 7, 4096),
            SupermaskLinear(in_features=512 * 7 * 7, out_features=4096, bias=False, sparsity=self.sparsity),
            nn.ReLU(True),
            nn.Dropout(),
            #nn.Linear(4096, 4096),
            SupermaskLinear(in_features=4096, out_features=4096, bias=False, sparsity=self.sparsity),
            nn.ReLU(True),
            nn.Dropout(),
            #nn.Linear(4096, num_classes),
            SupermaskLinear(in_features=4096, out_features=num_classes, bias=False, sparsity=self.sparsity)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                #nn.init.constant_(m.bias, 0)


def make_layers(cfg, sparsity, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            #conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            conv2d = SupermaskConv(sparsity=sparsity, in_channels=in_channels, out_channels=v,
                                   kernel_size=3, padding=1, bias=False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v, affine=False), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(num_classes, arch, cfg, batch_norm, sparsity, **kwargs):
    model = VGG(num_classes, make_layers(cfgs[cfg], sparsity, batch_norm=batch_norm), sparsity, **kwargs)
    return model


def vgg11(num_classes, sparsity, **kwargs):
    return _vgg(num_classes, 'vgg11', 'A', False, sparsity, **kwargs)


def vgg11_bn(num_classes, sparsity, **kwargs):
    return _vgg(num_classes, 'vgg11', 'A', True, sparsity, **kwargs)
