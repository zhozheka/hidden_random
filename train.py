import sys
import os
import json
import torch
from torch import nn
import torchvision
from datetime import datetime
import config
import argparse
import transforms
from torch.utils.data import DataLoader
from utils import AverageMeter, get_logger, intersection_over_union
from tqdm import tqdm
from datasets import TinyImageNet, Bowl
from models import get_model


class Trainer:
    def __init__(self, args):
        # set logging and model paths
        self.model_name = '{}_{}_{}_{}_{}'.format(
            args.model,
            args.sparsity,
            args.init_type,
            args.dataset,
            datetime.now().strftime('%Y_%b_%d_%H:%M:%S')
        )
        if args.bilinear:
            self.model_name = self.model_name + '_bilinear'

        self.model_dir = os.path.join(args.models_dir, self.model_name)
        os.mkdir(self.model_dir)

        self.logger = get_logger(os.path.join(self.model_dir, 'training.log'))
        self.logger.info(args)
        self.logger.info('Model dir: {}'.format(self.model_dir))
        self.args = args
        json.dump(args.__dict__,
                  fp=open(os.path.join(self.model_dir, 'args.json'), 'w'))

        self.save_chp = not args.not_save
        self.device = torch.device('cuda:{}'.format(args.cuda))
        self.loaders = self.get_loaders(args)
        self.num_classes = len(self.loaders['train'].dataset.classes)
        self.dataset_name = args.dataset
        self.sparsity = args.sparsity / 100
        self.epochs = args.epochs
        self.init_type = args.init_type
        self.model = get_model(args, self.num_classes, sparsity=self.sparsity, init_type=self.init_type).to(self.device)

        self.optimizer = torch.optim.SGD([p for p in self.model.parameters() if p.requires_grad],
                                         lr=args.lr,  momentum=args.momentum, weight_decay=args.wd)

        self.logger.info([name for name, p in self.model.named_parameters() if p.requires_grad])

        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=args.patience)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs)
        self.loss = nn.CrossEntropyLoss()

        self.save_interval = args.save_interval
        self.verbose_interval = args.verbose_interval
        self.best_score = 0.

    def load_checkpoint(self, chp_path):
        chp = torch.load(chp_path)
        assert self.model_name
        self.model.load_state_dict(chp['state_dict'])
        self.optimizer = chp['optimizer']

    def process_epoch(self, epoch, stage):
        assert stage in ['train', 'val']

        if stage == 'train':
            self.model.train()
        else:
            self.model.eval()

        meter_accuracy = AverageMeter()
        meter_loss = AverageMeter()
        meter_iou = AverageMeter()

        for i, (inp, target) in tqdm(enumerate(self.loaders[stage]), total=len(self.loaders[stage])):

            inp = inp.to(self.device)
            target = target.to(self.device)
            out = self.model(inp)
            loss_batch = self.loss(out, target)

            if stage == 'train':
                self.optimizer.zero_grad()
                loss_batch.backward()
                self.optimizer.step()

            pred = out.argmax(1)
            acc = (pred == target).cpu().numpy().mean()
            iou = intersection_over_union(pred, target)

            meter_accuracy.update(acc, len(inp))
            meter_loss.update(loss_batch.item(), len(inp))
            meter_iou.update(iou, len(inp))

            if (i + 1) % self.verbose_interval == 0 or i + 1 == len(self.loaders[stage]):
                if self.dataset_name == 'bowl':
                    self.logger.info('[{}] {} | Loss: {:.4}, IOU: {:.4}'.format(
                        epoch,
                        stage.capitalize(),
                        meter_loss.avg,
                        meter_iou.avg
                    ))
                else:
                    self.logger.info('[{}] {} | Loss: {:.4} / {:.4}, Accuracy: {:.4} / {:.4}'.format(
                        epoch,
                        stage.capitalize(),
                        meter_loss.val, meter_loss.avg,
                        meter_accuracy.val, meter_accuracy.avg
                    ))
        score = meter_iou.avg if self.dataset_name == 'bowl' else meter_accuracy.avg
        return meter_loss.avg, score

    def save_model(self, name, val_score=0, epoch=None):
        chp = {
            'state_dict': self.model.state_dict(),
            'epoch': epoch,
            'model': self.args.model,
            'num_classes': self.num_classes,
            'dataset': self.dataset_name,
            'sparsity': self.sparsity,
            'optimizer': self.optimizer,
            'val_score': val_score,
            'model_name': self.model_name
        }

        chp_path = os.path.join(self.model_dir, '{}.pth'.format(name))
        torch.save(chp, chp_path)
        # self.logger.info('Checkpoint {} saved'.format(chp_path))
        return chp_path

    def train(self):
        for epoch in range(1, self.epochs+1):

            self.logger.info('\nTraining: learning rate: {:.4}'.format(
                self.optimizer.param_groups[0]['lr']
            ))
            self.process_epoch(epoch, 'train')

            self.logger.info('\nValidation:')
            with torch.no_grad():
                val_loss, val_score = self.process_epoch(epoch, 'val')

            if val_score > self.best_score:  # trying to maximize
                self.best_score = val_score
                self.logger.info('New best checkpoint')
                self.save_model('best', val_score, epoch)

            # self.scheduler.step(val_loss)
            self.scheduler.step(epoch)

            if epoch % self.save_interval == 0 and self.save_interval > 0 or epoch == self.epochs:
                if self.save_chp:
                    chp_path = self.save_model('epoch_{}'.format(epoch), epoch)
                    self.logger.info('Checkpoint {} saved'.format(chp_path))

    @staticmethod
    def get_loaders(args):
        if args.dataset == 'cifar10':
            train_dataset = torchvision.datasets.CIFAR10(config.cifar10_root, train=True,
                                                         transform=transforms.cifar10['train'])
            val_dataset = torchvision.datasets.CIFAR10(config.cifar10_root, train=False,
                                                       transform=transforms.cifar10['test'])

        elif args.dataset == 'tiny-imagenet':
            train_dataset = TinyImageNet(config.tiny_imagenet_root, train=True,
                                         transform=transforms.imagenet['train'])
            val_dataset = TinyImageNet(config.tiny_imagenet_root, train=False,
                                       transform=transforms.imagenet['test'])

        elif args.dataset == 'bowl':
            train_dataset = Bowl(config.bowl_root, train=True,
                                 transform=transforms.bowl['train'])
            val_dataset = Bowl(config.bowl_root, train=False,
                               transform=transforms.bowl['test'])

        else:
            raise ValueError

        loaders = {
            'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8),
            'val': DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        }
        return loaders


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cuda', default=0, type=int)
    parser.add_argument('-models_dir', default=config.models_dir)
    parser.add_argument('-model', default='vgg11bn')
    parser.add_argument('-epochs', default=20, type=int)
    parser.add_argument('-lr', default=0.1, type=float)
    parser.add_argument('-save_interval', default=-1, type=int)
    parser.add_argument('-not_save', action='store_true')
    parser.add_argument('-verbose_interval', default=50, type=int)
    parser.add_argument('-batch_size', default=256, type=int)
    parser.add_argument('-dataset', default='cifar10')
    parser.add_argument('-momentum', default=0.9, type=float)
    parser.add_argument('-wd', default=0.0005,   type=float)
    parser.add_argument('-sparsity', default=50, type=int)
    parser.add_argument('-patience', default=5, type=int),
    parser.add_argument('-bilinear', action='store_true', help='Upsampling for U-net')
    parser.add_argument('-init_type', default='normal')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    sys.exit(main())
