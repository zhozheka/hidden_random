from torchvision import transforms as T

cifar10_normalization = T.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
cifar10 = {
    'train': T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        cifar10_normalization
    ]),
    'test': T.Compose([
        T.ToTensor(),
        cifar10_normalization
    ])
}

imagenet_normalization = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
imagenet = {
    'train': T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        imagenet_normalization
    ]),
    'test': T.Compose([
        T.ToTensor(),
        imagenet_normalization
    ])
}

#bowl_normalization = T.Normalize([0.164, 0.149, 0.182, 0.], [0.258, 0.238, 0.290, 1.])
bowl_normalization = T.Normalize([0.053, 0.046, 0.061, 0.], [0.109, 0.087, 0.143, 1.])
bowl = {
    'train': T.Compose([
        T.Resize((256, 256)),
        #T.RandomVerticalFlip(),
        #T.RandomHorizontalFlip(),
        T.ToTensor(),
        bowl_normalization
    ]),
    'test': T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        bowl_normalization
    ]),
}
