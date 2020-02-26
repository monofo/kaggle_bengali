import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import ohem_loss, topk_crossEntropy


class Loss_combine(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input, target, reduction='mean'):
        x1, x2, x3 = input
        x1, x2, x3 = x1.float(), x2.float(), x3.float()
        y = target.long()
        return 0.7*F.cross_entropy(x1,y[:,0],reduction=reduction) + 0.1*F.cross_entropy(x2,y[:,1],reduction=reduction) + \
          0.2*F.cross_entropy(x3,y[:,2],reduction=reduction)


def get_criterion(config):
    if config.loss.name == "combo":
        return Loss_combine()
    if config.loss.name == "CrossEntropy":
        return nn.CrossEntropyLoss()
    if config.loss.name == 'OHEM':
        return topk_crossEntropy(rate=config.loss.rate)
        
