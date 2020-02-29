import argparse
import gc
import os
import warnings
import itertools
from functools import partial
from typing import List, Optional, Union


import albumentations
import cv2
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm
import ttach as tta

from datasets.dataset_factory import KaggleDataset, make_loader
from losses.loss_factory import get_criterion
from models.model_factory import MODEL_LIST
from optimizers.optimizer_factory import get_optimizer
from schedulers.scheduler_factory import get_scheduler
# from transformars.transform_factory import get_transforms
from transformars.transform_factory import GridMask, Transform, apply_aug
from utils.config import load_config, save_config
from utils.metrics import macro_recall_multi
from utils.utils import (EarlyStopping, cutmix, cutmix_criterion, get_logger,
                         mixup, mixup_criterion, ohem_loss, rand_bbox)
from statistics import mean

os.environ["CUDA_VISIBLE_DEVICES"]='3'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Merger:

    def __init__(
            self,
            type: str = 'mean',
            n: int = 1,
    ):

        if type not in ['mean', 'gmean', 'sum', 'max', 'min', 'tsharpen']:
            raise ValueError('Not correct merge type `{}`.'.format(type))

        self.output = None
        self.type = type
        self.n = n

    def append(self, x):

        if self.type == 'tsharpen':
            x = x ** 0.5

        if self.output is None:
            self.output = x
        elif self.type in ['mean', 'sum', 'tsharpen']:
            self.output = self.output + x
        elif self.type == 'gmean':
            self.output = self.output * x
        elif self.type == 'max':
            self.output = F.max(self.output, x)
        elif self.type == 'min':
            self.output = F.min(self.output, x)

    @property
    def result(self):
        if self.type in ['sum', 'max', 'min']:
            result = self.output
        elif self.type in ['mean', 'tsharpen']:
            result = self.output / self.n
        elif self.type in ['gmean']:
            result = self.output ** (1 / self.n)
        else:
            raise ValueError('Not correct merge type `{}`.'.format(self.type))
        return result

def do_eval(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.
    valid_recall = 0.

    transforms = tta.Compose(
        [
            # tta.Rotate90(angles=[0])
            # tta.Rotate90(angles=[0, 90, 180, 270]),
            tta.Scale(scales=[1, 2, 4]),
            # tta.Multiply(factors=[0.9, 1, 1.1]),        
        ]
    )

    with torch.no_grad():
        total_loss = 0.
        for inputs in tqdm(data_loader, total=len(data_loader)):
            x = inputs["images"].to(device, dtype=torch.float)
            grapheme_root = inputs["grapheme_roots"].to(device, dtype=torch.long)
            vowel_diacritic = inputs["vowel_diacritics"].to(device, dtype=torch.long)
            consonant_diacritic = inputs["consonant_diacritics"].to(device, dtype=torch.long)

            a = Merger(type="mean", n=len(transforms))
            b = Merger(type="mean", n=len(transforms))
            c = Merger(type="mean", n=len(transforms))

            for transformer in transforms:
                augmented_image = transformer.augment_image(x)
                model_output = model(augmented_image)
                # print(transformer.deaugment_label(model_output[0]))
                a.append(transformer.deaugment_label(model_output[0]))
                b.append(transformer.deaugment_label(model_output[1]))
                c.append(transformer.deaugment_label(model_output[2]))

            logit_grapheme_root = a.result
            logit_vowel_diacritic = b.result
            logit_consonant_diacritic = c.result
                
            # logit_grapheme_root, logit_vowel_diacritic, logit_consonant_diacritic = model(x)


            valid_recall += macro_recall_multi(logit_grapheme_root, grapheme_root, logit_vowel_diacritic, vowel_diacritic, logit_consonant_diacritic, consonant_diacritic)


        valid_recall /= len(data_loader)
        metrics = {'valid_loss': total_loss, "valid_recall": valid_recall}
    return metrics


def run(config_file):
    config = load_config(config_file)  
    config.work_dir = '/home/kazuki/workspace/kaggle_bengali/result/'+config.work_dir
    os.makedirs(config.work_dir, exist_ok=True)
    os.makedirs(config.work_dir + "/checkpoints", exist_ok=True)
    print('working directory:', config.work_dir)
    logger = get_logger(config.work_dir+"log.txt")
    
    all_transforms = {}
    all_transforms['valid'] = Transform(size=config.data.image_size)

    dataloaders = {
        phase: make_loader(
            phase=phase,
            # batch_size=config.train.batch_size,
            batch_size=64,
            num_workers=config.num_workers,
            idx_fold=config.data.params.idx,
            fold_csv=config.data.params.fold_csv,
            transforms=all_transforms[phase],
            # debug=config.debug
            crop=config.transforms.crop
        )
        for phase in ['valid']
    }
    model = MODEL_LIST[config.model.version](pretrained=True)
    model = model.to(device)

    # model = torch.nn.DataParallel(model) # make parallel

    criterion = get_criterion(config)
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(optimizer, config)

    best_valid_recall = 0.0
    if 1:
        print('resume checkpoints')
        checkpoint = torch.load(config.work_dir + "/checkpoints/" + "121.pth")
        model.load_state_dict(checkpoint['checkpoint'])
        best_valid_recall = checkpoint['best_valid_recall']


    metrics_eval = do_eval(model, dataloaders["valid"], criterion, device)
    valid_recall = metrics_eval["valid_recall"]
    print(metrics_eval)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_file',
                        help='configuration file path',
                        default=None, type=str)

    return parser.parse_args()


def main():
    args = parse_args()
    run(args.config_file)


if __name__ == '__main__':
    main()
