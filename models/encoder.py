import torch
from torch import nn
import models
from collections import OrderedDict
from argparse import Namespace
import yaml
import os
import torch.nn.functional as F
import numpy as np


class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False


class EncodeProject(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        if hparams.arch == 'ResNet50':
            cifar_head = (hparams.data == 'cifar')
            self.convnet = models.resnet.ResNet50(cifar_head=cifar_head, hparams=hparams)
            self.encoder_dim = 2048
        elif hparams.arch == 'ResNet18':
            self.convnet = models.resnet.ResNet18(cifar_head=(hparams.data == 'cifar'))
            self.encoder_dim = 512
        else:
            raise NotImplementedError
        self.hparams = hparams

        num_params = sum(p.numel() for p in self.convnet.parameters() if p.requires_grad)

        print(f'======> Encoder: output dim {self.encoder_dim} | {num_params/1e6:.3f}M parameters')

        self.proj_dim = 128
        projection_layers = [
            ('fc1', nn.Linear(self.encoder_dim, self.encoder_dim, bias=False)),
            ('bn1', nn.BatchNorm1d(self.encoder_dim)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(self.encoder_dim, 128, bias=False)),
            ('bn2', BatchNorm1dNoBias(128)),
        ]

        self.projection = nn.Sequential(OrderedDict(projection_layers))

        self.rot_head = torch.nn.Linear(128, 4).cuda()

    def forward(self, x, out='z'):
        h = self.convnet(x)
        if out == 'h':
            return h
        elif out == 'rot':
            z = self.projection(h)
            ## divide tau for simclr but not for r2d2
            z = F.normalize(z, p=2, dim=1) / np.sqrt(self.hparams.temperature)
            return self.rot_head(z)
        return self.projection(h)
