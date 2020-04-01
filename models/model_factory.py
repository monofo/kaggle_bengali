import math
from typing import List

import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F
from models.effcientNet import EfficientNet
from models.metric import ArcMarginProduct
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
from tqdm import tqdm_notebook as tqdm
from torch.optim import lr_scheduler
from torch.nn.parameter import Parameter


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
        # x = F.dropout(x, 0.4, self.training)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        return [l0, l1, l2]


class effcientNet(nn.Module):
    def __init__(self, pretrained):
        super(effcientNet, self).__init__()
        if pretrained is True:
            self.model = EfficientNet.from_pretrained('efficientnet-b4') 
        else:
            self.model = EfficientNet.from_name('efficientnet-b4') 
        

    def forward(self, x):
        l0, l1, l2 = self.model(x)
        return [l0, l1, l2] 


class ResNet34_(nn.Module):
    def __init__(self, pretrained, out_dim):
        super(ResNet34_, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained=None)
        
        self.l0 = nn.Linear(512, out_dim)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0 = self.l0(x)
        return l0


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine


class arcModel(nn.Module):
    def __init__(self, back_bone, out_dim ):
        super(arcModel, self).__init__()
        self.model = pretrainedmodels.__dict__["resnet34"](pretrained='imagenet')
        self.metric_classify = ArcMarginProduct(512, out_dim)
        self.fc = nn.Linear(512, out_dim)
        
    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        metric_output = self.metric_classify(x)
        graphme = self.fc(x)
        return graphme, metric_output
    
class seenModel(nn.Module):
    def __init__(self, back_bone):
        super(seenModel, self).__init__()
        self.model = pretrainedmodels.__dict__[back_bone](pretrained="imagenet")

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

class unseenModel(nn.Module):
    def __init__(self, back_bone, out_dim1, out_dim2):
        super(unseenModel, self).__init__()
        self.model = pretrainedmodels.__dict__[back_bone](pretrained="imagenet")      
        self.l0 = nn.Linear(512, out_dim)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0 = self.l0(x)
        return l0

MODEL_LIST = {
    'resnet34': ResNet34,
    'se_resnext50_32x4d': se_resnext50_32x4d,
    'effcientNet': effcientNet,
    'arc_model': arcModel,
    'seenModel': seenModel,
    'unseenModel': unseenModel,
}
