import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ResNet import resnet18
from .resnet20_cifar import resnet20
from .ResNet import resnet50

import numpy as np


class MyModel(nn.Module):
    def __init__(self, args):
        super(MyModel, self).__init__()
        self.args = args
        if self.args.dataset == 'miniImageNet' or self.args.dataset == 'cifar_fs':
            self.backbone = resnet18()
            self.num_features = 512
            self.num_cls = 100
            self.base_cls_num = 60
            self.sessions = 9
            # self.inc_cls_num = 5
            self.inc_cls_num = (self.num_cls-self.base_cls_num) // (self.sessions-1)
        elif self.args.dataset == 'cub_200' or self.args.dataset == 'ImageNet_R':
            if self.args.pretrain:
                if self.args.network == 'ResNet50':
                    self.backbone = resnet50(True)
                    self.num_features = 2048
                elif self.args.network == 'ResNet18':
                    self.backbone = resnet18(True)
                    self.num_features = 512
            else:
                if self.args.pretrained_model == 'cec':
                    self.encoder = resnet18()
                else:
                    self.backbone = resnet18()
                self.num_features = 512
            self.num_cls = 200
            self.base_cls_num = 100
            self.inc_cls_num = 10
            self.sessions = 11
        else:
            raise Exception("Invalid dataset name {}".format(self.args.dataset))

        self.fc = nn.Linear(self.num_features, self.num_cls, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x, norm=False):
        x = self.encode(x)
        x = self.fc(x)
        if norm:
            x = x / x.norm(p=2, dim=-1, keepdim=True)
        return x


    def encode(self, x, avg=True, return_mid=False):
        if return_mid:
            x1, x2, x3, x4 = self.backbone(x, return_mid=True)
            x = self.avgpool(x4).squeeze(-1).squeeze(-1)
            return x1, x2, x3, x4, x
        else:

            if self.args.pretrained_model == 'cec':
                x = self.encoder(x)
            else:
                x = self.backbone(x)
            if avg:
                x = self.avgpool(x).squeeze(-1).squeeze(-1)
            return x

    
    def encode_mls(self, x, mode='concat'):
        x = self.backbone(x, return_layers='layer34')
        if mode == 'concat':
            feat3, feat4 = x
            feat3 = self.avgpool(feat3).squeeze(-1).squeeze(-1) # [-1, 256]
            feat4 = self.avgpool(feat4).squeeze(-1).squeeze(-1) # [-1, 512]
            x = torch.cat((feat3, feat4), dim=-1) # [-1, 768]
        return x
