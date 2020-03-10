import math
from typing import List

import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F
from models.effcientNet import EfficientNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
from tqdm import tqdm_notebook as tqdm
from torch.optim import lr_scheduler

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
from torch.nn.parameter import Parameter
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)       
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'
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
        x = F.dropout(x, 0.4, self.training)
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

class ResNet34_bn(nn.Module):
    def __init__(self, pretrained):
        super(ResNet34_bn, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained=None)
        
        self.basemodel = nn.Sequential(*list(self.model.children())[:-2])
        self.num_ch = 512
        self.conv1 = double_conv(num_ch, 512)
        # vowel_diacritic       
        self.fc1 = nn.Conv2d(num_ch, 11, 1)
        # grapheme_root
        self.fc2 = nn.Conv2d(512, 168, 1)
        # consonant_diacritic
        self.fc3 = nn.Conv2d(num_ch, 7, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.gem =GeM()
        self.gem2 =GeM()


    def forward(self, x):
        # extract features
        x = self.features(x)
        x2 = self.conv1(x)
        # squeeze         to 1x1
        x = self.avgpool(x)
        x2 = self.avgpool(x2)

        x1 = self.fc1(x).squeeze(2).squeeze(2)
        x2o = self.fc2(x2).squeeze(2).squeeze(2)       
        x3 = self.fc3(x).squeeze(2).squeeze(2)
        
        return [x1, x2o, x3]

class ResNet18(nn.Module):
    def __init__(self, pretrained):
        super(ResNet18, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["resnet18"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["resnet18"](pretrained=None)
        
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
        # x = F.dropout(x, 0.4, self.training)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        return [l0, l1, l2]


class se_resnext50_32x4d_bn(nn.Module):
    def __init__(self, pretrained):
        super(se_resnext50_32x4d_bn, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["se_resnext50_32x4d"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["se_resnext50_32x4d"](pretrained=None)
        
        self.basemodel = nn.Sequential(*list(self.model.children())[:-2])
        self.num_ch = 2048
        self.conv1 = double_conv(self.num_ch, 512)
        # vowel_diacritic       
        self.fc1 = nn.Conv2d(self.num_ch, 11, 1)
        # grapheme_root
        self.fc2 = nn.Conv2d(512, 168, 1)
        # consonant_diacritic
        self.fc3 = nn.Conv2d(self.num_ch, 7, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.gem =GeM()
        self.gem2 =GeM()


    def forward(self, x):
        # extract features
        x = self.basemodel(x)
        x2 = self.conv1(x)
        # squeeze         to 1x1
        x = self.avgpool(x)
        x2 = self.avgpool(x2)

        x1 = self.fc1(x).squeeze(2).squeeze(2)
        x2o = self.fc2(x2).squeeze(2).squeeze(2)       
        x3 = self.fc3(x).squeeze(2).squeeze(2)
        
        return [x2o, x1, x3]
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
        x = F.dropout(x, 0.4, self.training)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        return [l0, l1, l2] 





class effcientNet(nn.Module):
    def __init__(self, pretrained):
        super(effcientNet, self).__init__()
        if pretrained is True:
            self.model = EfficientNet.from_pretrained('efficientnet-b2') 
        else:
            self.model = EfficientNet.from_name('efficientnet-b2') 
        

    def forward(self, x):
        l0, l1, l2 = self.model(x)
        return [l0, l1, l2] 


MODEL_LIST = {
    'se_resnext101_32x4d': se_resnext101_32x4d,
    'resnet34': ResNet34,
    'resnet18': ResNet18,
    'se_resnext50_32x4d': se_resnext50_32x4d,
    'densenet121': densenet121,
    'effcientNet': effcientNet,
    'ResNet34_bn': ResNet34_bn,
    'se_resnext50_32x4d_bn': se_resnext50_32x4d_bn,
}
