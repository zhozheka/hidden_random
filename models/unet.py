"""
Based on https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import SupermaskConv, SupermaskConvTranspose


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, sparsity, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            #nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            SupermaskConv(sparsity=sparsity, in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                          padding=1, bias=False),
            nn.BatchNorm2d(out_channels, affine=False),
            nn.ReLU(inplace=True),
            #nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            SupermaskConv(sparsity=sparsity, in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                          padding=1, bias=False),
            nn.BatchNorm2d(out_channels, affine=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, sparsity, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(sparsity, in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, sparsity, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            #self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.up = SupermaskConvTranspose(sparsity=sparsity, in_channels=in_channels // 2,
                                             out_channels=in_channels // 2, kernel_size=2, stride=2, bias=False)

        self.conv = DoubleConv(sparsity, in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, sparsity, in_channels, out_channels):
        super(OutConv, self).__init__()
        #self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = SupermaskConv(sparsity=sparsity, in_channels=in_channels,
                                  out_channels=out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, sparsity, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(sparsity=sparsity, in_channels=n_channels, out_channels=64)
        self.down1 = Down(sparsity, 64, 128)
        self.down2 = Down(sparsity, 128, 256)
        self.down3 = Down(sparsity, 256, 512)
        self.down4 = Down(sparsity, 512, 512)
        self.up1 = Up(sparsity, 1024, 256, bilinear)
        self.up2 = Up(sparsity, 512, 128, bilinear)
        self.up3 = Up(sparsity, 256, 64, bilinear)
        self.up4 = Up(sparsity, 128, 64, bilinear)
        self.outc = OutConv(sparsity=sparsity, in_channels=64, out_channels=n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    batch = torch.rand(16, 3, 256, 256)
    model = UNet(n_channels=3, n_classes=2, sparsity=0.5, bilinear=False)
    out = model(batch)
    print('result.shape: {}'.format(out.shape))
