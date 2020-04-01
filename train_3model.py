import argparse
import gc
import os
import warnings

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
from collections import OrderedDict

os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SCALE = 4
def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict
    
    
def do_train(model, data_loader, criterion, optimizer, device, config, epoch, grad_acc=1):
        model[0].train()
        model[1].train()
        model[2].train()

        train_loss = 0.0
        train_recall = 0.0
        optimizer[0].zero_grad()
        optimizer[1].zero_grad()
        optimizer[2].zero_grad()
        

        for idx, (inputs) in tqdm(enumerate(data_loader), total=len(data_loader)):
            choice = np.random.rand(1)
            x = inputs["images"].to(device, dtype=torch.float)
            grapheme_root = inputs["grapheme_roots"].to(device, dtype=torch.long)
            vowel_diacritic = inputs["vowel_diacritics"].to(device, dtype=torch.long)
            consonant_diacritic = inputs["consonant_diacritics"].to(device, dtype=torch.long)
            
            if choice <= config.train.cutmix:
                data, targets = cutmix(x, grapheme_root, vowel_diacritic, consonant_diacritic, 1.)
                logit_grapheme_root = model[0](data)
                logit_vowel_diacritic = model[1](data)
                logit_consonant_diacritic = model[2](data)
                loss_gr, loss_vd, loss_cd = cutmix_criterion(logit_grapheme_root, logit_vowel_diacritic, logit_consonant_diacritic, targets, criterion)
            else:
                logit_grapheme_root = model[0](x)
                logit_vowel_diacritic = model[1](x)
                logit_consonant_diacritic = model[2](x)
                loss_gr = criterion(logit_grapheme_root, grapheme_root)
                loss_vd = criterion(logit_vowel_diacritic, vowel_diacritic )
                loss_cd = criterion(logit_consonant_diacritic, consonant_diacritic)

            train_recall += macro_recall_multi(logit_grapheme_root, grapheme_root, logit_vowel_diacritic, vowel_diacritic, logit_consonant_diacritic, consonant_diacritic)

            ((SCALE * loss_gr + loss_cd + loss_vd) / grad_acc).backward()

            if (idx % grad_acc) == 0:
                optimizer[0].step()
                optimizer[0].zero_grad()
                optimizer[1].step()
                optimizer[1].zero_grad()
                optimizer[2].step()
                optimizer[2].zero_grad()
            train_loss += SCALE * loss_gr.item() + loss_vd.item() + loss_cd.item()

        train_loss /= len(data_loader)
        train_recall /= len(data_loader)

        return {'train_loss': train_loss, 'train_recall': train_recall}


def do_eval(model, data_loader, criterion, device):
    model[0].eval()
    model[1].eval()
    model[2].eval()
    total_loss = 0.
    valid_recall = 0.

    with torch.no_grad():
        total_loss = 0.
        for inputs in tqdm(data_loader, total=len(data_loader)):
            x = inputs["images"].to(device, dtype=torch.float)
            grapheme_root = inputs["grapheme_roots"].to(device, dtype=torch.long)
            vowel_diacritic = inputs["vowel_diacritics"].to(device, dtype=torch.long)
            consonant_diacritic = inputs["consonant_diacritics"].to(device, dtype=torch.long)

            logit_grapheme_root = model[0](x)
            logit_vowel_diacritic = model[1](x)
            logit_consonant_diacritic = model[2](x)

            loss_gr = criterion(logit_grapheme_root, grapheme_root)
            loss_vd = criterion(logit_vowel_diacritic, vowel_diacritic )
            loss_cd = criterion(logit_consonant_diacritic, consonant_diacritic)

            valid_recall += macro_recall_multi(logit_grapheme_root, grapheme_root, logit_vowel_diacritic, vowel_diacritic, logit_consonant_diacritic, consonant_diacritic)

            total_loss += SCALE * loss_gr.item() + loss_vd.item() +  loss_cd.item()

        total_loss /= len(data_loader)
        valid_recall /= len(data_loader)
        metrics = {'valid_loss': total_loss, "valid_recall": valid_recall}
    return metrics


def run(config_file):
    config = load_config(config_file)  
    config.work_dir = '/home/koga/workspace/kaggle_bengali/result/'+config.work_dir
    os.makedirs(config.work_dir, exist_ok=True)
    os.makedirs(config.work_dir + "/checkpoints", exist_ok=True)
    print('working directory:', config.work_dir)
    logger = get_logger(config.work_dir+"log.txt")
    
    all_transforms = {}
    all_transforms['train'] = Transform(
            size=config.data.image_size,
            affine=config.transforms.affine,
            autoaugment_ratio=config.transforms.autoaugment_ratio,
            threshold=config.transforms.threshold,
            sigma=config.transforms.sigma,
            blur_ratio=config.transforms.blur_ratio,
            noise_ratio=config.transforms.noise_ratio,
            cutout_ratio=config.transforms.cutout_ratio,
            grid_distortion_ratio=config.transforms.grid_distortion_ratio,
            random_brightness_ratio=config.transforms.random_brightness_ratio,
            piece_affine_ratio=config.transforms.piece_affine_ratio,
            ssr_ratio=config.transforms.ssr_ratio,
            grid_mask_ratio=config.transforms.grid_mask_ratio,
            augmix_ratio=config.transforms.augmix_ratio,
    )
    all_transforms['valid'] = Transform(size=config.data.image_size)

    dataloaders = {
        phase: make_loader(
            phase=phase,
            df_path=config.train.dfpath,
            batch_size=config.train.batch_size,
            num_workers=config.num_workers,
            idx_fold=config.data.params.idx,
            fold_csv=config.data.params.fold_csv,
            transforms=all_transforms[phase],
            # debug=config.debug
            crop=config.transforms.crop
        )
        for phase in ['train', 'valid']
    }
    model_root = MODEL_LIST['Resnet34_3model'](pretrained=config.model.pretrained, out_dim=168)
    model_vowel = MODEL_LIST['Resnet34_3model'](pretrained=config.model.pretrained, out_dim=11)
    model_const = MODEL_LIST['Resnet34_3model'](pretrained=config.model.pretrained, out_dim=7)

    model_root = model_root.to(device)
    model_vowel = model_vowel.to(device)
    model_const = model_const.to(device)

    model_list = [model_root, model_vowel, model_const]

    criterion = get_criterion(config)
    optimizer_root = get_optimizer(config, model_root)
    optimizer_vowel = get_optimizer(config, model_vowel)
    optimizer_const = get_optimizer(config, model_const)

    optimizer_list = [optimizer_root, optimizer_vowel, optimizer_const]

    scheduler_root = get_scheduler(optimizer_root, config)
    scheduler_vowel = get_scheduler(optimizer_vowel, config)
    scheduler_const = get_scheduler(optimizer_const, config)

    scheduler_list = [scheduler_root, scheduler_vowel, scheduler_const]

    accumlate_step = 1
    if config.train.accumulation_size > 0:
        accumlate_step = config.train.accumulation_size // config.train.batch_size

    best_valid_recall = 0.0
    if config.train.resume:
        print('resume checkpoints')
        checkpoint = torch.load("/home/koga/workspace/kaggle_bengali/result/" + config.train.path)
        model.load_state_dict(fix_model_state_dict(checkpoint['checkpoint']))



    valid_recall = 0.0

    for epoch in range(1, config.train.num_epochs+1):
        print(f'epoch {epoch} start')
        logger.info(f'epoch {epoch} start ')

        metric_train = do_train(model_list, dataloaders["train"], criterion, optimizer_list, device, config, epoch, accumlate_step)
        torch.cuda.empty_cache()
        metrics_eval = do_eval(model_list, dataloaders["valid"], criterion, device)
        torch.cuda.empty_cache()
        valid_recall = metrics_eval["valid_recall"]

        scheduler_list[0].step(metrics_eval["valid_recall"])
        scheduler_list[1].step(metrics_eval["valid_recall"])
        scheduler_list[2].step(metrics_eval["valid_recall"])


        print(f'epoch: {epoch} ', metric_train, metrics_eval)
        logger.info(f'epoch: {epoch} {metric_train} {metrics_eval}')
        if valid_recall > best_valid_recall:
            print(f"save checkpoint: best_recall:{valid_recall}")
            logger.info(f"save checkpoint: best_recall:{valid_recall}")
            torch.save({
                'checkpoint_root': model_list[0].state_dict(),
                'checkpoint_vowel': model_list[1].state_dict(),
                'checkpoint_const': model_list[2].state_dict(),
                'epoch': epoch,
                'best_valid_recall': valid_recall,
                }, config.work_dir + "/checkpoints/" + f"{epoch}.pth")
            best_valid_recall = valid_recall

        torch.cuda.empty_cache()
        gc.collect()


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
