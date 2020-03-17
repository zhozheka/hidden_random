import torchvision
from .vgg import *
from .simplenet import SimpleNet
from .unet import UNet
from .resnet import *


def get_model(args, num_classes, sparsity, init_type, **kwargs):
    model = vgg11_bn(num_classes, sparsity, init_type, **kwargs)
    if args.model == 'vgg11':
        model = vgg11(num_classes, sparsity, init_type, **kwargs)

    elif args.model == 'vgg11bn':
        model = vgg11_bn(num_classes, sparsity, init_type, **kwargs)

    elif args.model == 'vgg19':
        model = vgg19(num_classes, sparsity, init_type, **kwargs)

    elif args.model == 'vgg19bn':
        model = vgg19_bn(num_classes, sparsity, init_type, **kwargs)

    elif args.model == 'simplenet':
        model = SimpleNet(num_classes, sparsity, init_type, **kwargs)

    elif args.model == 'resnet18':
        model = resnet18(num_classes, sparsity, init_type, **kwargs)

    elif args.model == 'resnet34':
        model = resnet34(num_classes, sparsity, init_type, **kwargs)

    elif args.model == 'unet':
        model = UNet(3, num_classes, sparsity, init_type, args.bilinear, **kwargs)

    else:
        raise ValueError

    return model
