import torchvision
from .vgg import *
from .simplenet import SimpleNet
from .unet import UNet


def get_model(args, num_classes, **kwargs):
    if args.model == 'vgg11bn':
        model = vgg11_bn(num_classes, **kwargs)

    elif args.model == 'simplenet':
        model = SimpleNet(num_classes, **kwargs)

    elif args.model == 'unet':
        model = UNet(3, num_classes, **kwargs)

    else:
        raise ValueError

    return model
