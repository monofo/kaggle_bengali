import argparse
import gc
import os
import warnings

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

from datasets.dataset_factory import KaggleDataset, make_loader
from losses.loss_factory import get_loss
from models.model_factory import get_model
from optimizers.optimizer_factory import get_optimizer
from schedulers.scheduler_factory import get_scheduler
# from transformars.transform_factory import get_transforms
from transformars.transform_factory import Transform
from utils.config import load_config, save_config
from utils.metrics import macro_recall_multi

os.environ["CUDA_VISIBLE_DEVICES"]='1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def do_train(model, data_loader, criterion, optimizer, device, grad_acc=1):
        model.train()
        train_loss = 0.0
        train_recall = 0.0

        for idx, (inputs) in tqdm(enumerate(data_loader), total=len(data_loader)):
            x = inputs["images"].to(device)
            grapheme_root = inputs["grapheme_roots"].to(device)
            vowel_diacritic = inputs["vowel_diacritics"].to(device)
            consonant_diacritic = inputs["consonant_diacritics"].to(device)

            optimizer.zero_grad()
            
            logit_grapheme_root, logit_vowel_diacritic, logit_consonant_diacritic = model(x) 

            loss_gr = 2.0 * criterion(logit_grapheme_root, grapheme_root)
            loss_vd = criterion(logit_vowel_diacritic, vowel_diacritic )
            loss_cd = criterion(logit_consonant_diacritic, consonant_diacritic)

            train_loss += loss_gr.item() + loss_vd.item() + loss_cd.item()
            train_recall += macro_recall_multi(logit_grapheme_root, grapheme_root, logit_vowel_diacritic, vowel_diacritic, logit_consonant_diacritic, consonant_diacritic)

            (loss_gr + loss_cd + loss_vd).backward()

            if (idx % grad_acc) == 0:
                optimizer.step()
                optimizer.zero_grad()
            train_loss /= len(data_loader)

            return {'train_loss': train_loss}


def do_eval(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.
    valid_recall = 0.

    with torch.no_grad():
        total_loss = 0.
        for inputs in tqdm(data_loader, total=len(data_loader)):
            x = inputs["images"].to(device)
            grapheme_root = inputs["grapheme_roots"].to(device)
            vowel_diacritic = inputs["vowel_diacritics"].to(device)
            consonant_diacritic = inputs["consonant_diacritics"].to(device)

            logit_grapheme_root, logit_vowel_diacritic, logit_consonant_diacritic = model(x)

            loss_gr = 2.0 * criterion(logit_grapheme_root, grapheme_root)
            loss_vd = criterion(logit_vowel_diacritic, vowel_diacritic )
            loss_cd = criterion(logit_consonant_diacritic, consonant_diacritic)

            valid_recall += macro_recall_multi(logit_grapheme_root, grapheme_root, logit_vowel_diacritic, vowel_diacritic, logit_consonant_diacritic, consonant_diacritic)

            total_loss += loss_gr.item() + loss_vd.item() + loss_cd.item()

        total_loss /= len(data_loader)
        valid_recall /= len(data_loader)
        metrics = {'valid_loss': total_loss, "valid_recall": valid_recall}
    return metrics


def run(config_file):
    config = load_config(config_file)

    config.work_dir = '/home/kazuki/workspace/kaggle_bengali/result/' + config.work_dir
    os.makedirs(config.work_dir, exist_ok=True)
    os.makedirs(config.work_dir + "checkpoints", exist_ok=True)
    print('working directory:', config.work_dir)
    
    all_transforms = {}
    all_transforms['train'] = train_transform = Transform(
            size=config.data.image_size,
            threshold=20.,
            sigma=-1., 
            blur_ratio=0.2, 
            noise_ratio=0.2, 
            cutout_ratio=0.2,
            grid_distortion_ratio=0.2, 
            random_brightness_ratio=0.2,
            piece_affine_ratio=0.2, 
            ssr_ratio=0.2
    )
    all_transforms['valid'] = Transform(size=config.data.image_size)

    dataloaders = {
        phase: make_loader(
            phase=phase,
            batch_size=config.train.batch_size,
            num_workers=config.num_workers,
            idx_fold=config.data.params.idx,
            fold_csv=config.data.params.fold_csv,
            transforms=all_transforms[phase],
            # debug=config.debug
        )
        for phase in ['train', 'valid']
    }
    model = get_model(config)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(optimizer, config)

    accumlate_step = 1
    if config.train.accumulation_size > 0:
        accumlate_step = config.train.accumulation_size // config.train.batch_size

    valid_recall = 0.0
    best_valid_recall = 0.0
    for epoch in range(1, config.train.num_epochs+1):
        print(f'epoch {epoch} start ')
        
        # train code
        metric_train = do_train(model, dataloaders["train"], criterion, optimizer, device, accumlate_step)
        metrics_eval = do_eval(model, dataloaders["valid"], criterion, device)
        valid_recall = metrics_eval["valid_recall"]

        scheduler.step()

        print(f'epoch: {epoch} ', metric_train, metrics_eval)
        if valid_recall > best_valid_recall:
            print(f"save checkpoint: best_recall:{valid_recall}")
            torch.save(model.state_dict(), config.work_dir + "/checkpoints/" + "best.pth")
            best_valid_recall = valid_recall


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_file',
                        help='configuration file path',
                        default=None, type=str)
    # parser.add_argument('--out_dir', type=str, default="result/baseline")
    return parser.parse_args()


def main():
    args = parse_args()
    run(args.config_file)


if __name__ == '__main__':
    main()
