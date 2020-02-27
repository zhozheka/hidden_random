import os
import torch
from torch.utils.data import Dataset
from glob import glob
from PIL import Image


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform

        wnids = []
        with open(os.path.join(root, 'wnids.txt')) as f:
            for line in f:
                wnids.append(line.strip())
        wnids2id = dict((wnid, j) for j, wnid in enumerate(wnids))

        wnid2class = {}
        with open(os.path.join(self.root, 'words.txt')) as f:
            for line in f:
                wnid, cl = line.strip().split('\t')
                wnid2class[wnid] = cl
        self.classes = [wnid2class[wnid] for wnid in wnids]

        if train:
            path_template = os.path.join(self.root, 'train/*/images/*')
            self.paths = glob(path_template)
            self.labels = [wnids2id[path.split('/')[-1].split('_')[0]] for path in self.paths]
        else:
            img2wnid = {}
            with open(os.path.join(root, 'val/val_annotations.txt')) as f:
                for line in f:
                    spl = line.strip().split('\t')
                    img_name = spl[0]
                    wnid = spl[1]
                    img2wnid[img_name] = wnid

            path_template = os.path.join(self.root, 'val/images/*')
            self.paths = glob(path_template)
            self.labels = [wnids2id[img2wnid[path.split('/')[-1]]] for path in self.paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


class Bowl(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train
        self.classes = [0, 1]

        if train:
            self.ids = os.listdir(os.path.join(self.root, 'train'))
        else:
            self.ids = os.listdir(os.path.join(self.root, 'val'))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.root,
            'train' if self.train else 'val',
            self.ids[idx],
            'images',
            '{}.png'.format(self.ids[idx])
        )
        mask_path = os.path.join(
            self.root,
            'train' if self.train else 'val',
            self.ids[idx],
            'mask.png'
        )
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        if self.transform:
            img.putalpha(mask)
            res = self.transform(img)
            img = res[:3]
            mask = res[3].type(torch.long)
        return img, mask
