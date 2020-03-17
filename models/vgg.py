"""
Based on https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""
import torch
import torch.nn as nn
from .layers import SupermaskLinear, SupermaskConv

sparsity_glb = -1
init_type_glb = ''


class VGG(nn.Module):
    def __init__(self, num_classes, features):
        super(VGG, self).__init__()
        self.features = features
        self.num_classes = num_classes
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.sparsity = sparsity_glb
        self.init_type = init_type_glb

        self.classifier = nn.Sequential(
            SupermaskLinear(in_features=512 * 7 * 7, out_features=4096, bias=False,
                            sparsity=self.sparsity, init_type=self.init_type),
            nn.ReLU(True),
            nn.Dropout(),
            SupermaskLinear(in_features=4096, out_features=4096, bias=False,
                            sparsity=self.sparsity, init_type=self.init_type),
            nn.ReLU(True),
            nn.Dropout(),
            SupermaskLinear(in_features=4096, out_features=num_classes, bias=False,
                            sparsity=self.sparsity, init_type=self.init_type)
        )
        if True:
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


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = SupermaskConv(sparsity=sparsity_glb, init_type=init_type_glb, in_channels=in_channels,
                                   out_channels=v, kernel_size=3, padding=1, bias=False)
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


def _vgg(num_classes, arch, cfg, batch_norm, sparsity, init_type, **kwargs):
    global sparsity_glb, init_type_glb
    sparsity_glb = sparsity
    init_type_glb = init_type

    features = make_layers(cfgs[cfg], batch_norm=batch_norm)
    model = VGG(num_classes, features, **kwargs)
    return model


def vgg11(num_classes, sparsity, init_type, **kwargs):
    return _vgg(num_classes, 'vgg11', 'A', False, sparsity, init_type, **kwargs)


def vgg11_bn(num_classes, sparsity, init_type,  **kwargs):
    return _vgg(num_classes, 'vgg11', 'A', True, sparsity, init_type, **kwargs)


def vgg13(num_classes, sparsity, init_type, **kwargs):
      return _vgg(num_classes, 'vgg13', 'B', False, sparsity, init_type, **kwargs)


def vgg13_bn(num_classes, sparsity, init_type, **kwargs):
    return _vgg(num_classes, 'vgg13_bn', 'B', True, sparsity, init_type, **kwargs)


def vgg19(num_classes, sparsity, init_type, **kwargs):
    return _vgg(num_classes, 'vgg19', 'E', False, sparsity, init_type, **kwargs)


def vgg19_bn(num_classes, sparsity, init_type, **kwargs):
    return _vgg(num_classes, 'vgg19', 'E', True, sparsity, init_type, **kwargs)

