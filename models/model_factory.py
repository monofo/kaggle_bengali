import math
from typing import List

import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F


class se_resnext101_32x4d(nn.Module):
    def __init__(self, pretrained):
        super(se_resnext101_32x4d, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["se_resnext101_32x4d"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["se_resnext101_32x4d"](pretrained=None)

        self.l0 = nn.Linear(2048, 168)
        self.l1 = nn.Linear(2048, 11)
        self.l2 = nn.Linear(2048, 7)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        return [l0, l1, l2]


class ResNet34(nn.Module):
    def __init__(self, pretrained):
        super(ResNet34, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained=None)
        
        self.l0 = nn.Linear(512, 168)
        self.l1 = nn.Linear(512, 11)
        self.l2 = nn.Linear(512, 7)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        return [l0, l1, l2]

class se_resnext50_32x4d(nn.Module):
    def __init__(self, pretrained):
        super(se_resnext50_32x4d, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["se_resnext50_32x4d"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["se_resnext50_32x4d"](pretrained=None)

        self.l0 = nn.Linear(2048, 168)
        self.l1 = nn.Linear(2048, 11)
        self.l2 = nn.Linear(2048, 7)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        return [l0, l1, l2]


class densenet121(nn.Module):
    def __init__(self, pretrained):
        super(densenet121, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["densenet121"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["densenet121"](pretrained=None)

        self.l0 = nn.Linear(1024, 168)
        self.l1 = nn.Linear(1024, 11)
        self.l2 = nn.Linear(1024, 7)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        return [l0, l1, l2] 


MODEL_LIST = {
    'se_resnext101_32x4d': se_resnext101_32x4d,
    'resnet34': ResNet34,
    'se_resnext50_32x4d': se_resnext50_32x4d,
    'densenet121': densenet121
}
