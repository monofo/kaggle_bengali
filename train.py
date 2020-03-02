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
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

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

os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def prepare_image(datadir, featherdir, data_type='train',

                  submission=False, indices=[0, 1, 2, 3]):
    assert data_type in ['train', 'test']
    if submission:
        image_df_list = [pd.read_parquet(datadir + f'{data_type}_image_data_{i}.parquet')
                         for i in indices]
    else:
        image_df_list = [pd.read_feather(featherdir + f'{data_type}_image_data_{i}.feather')
                         for i in indices]
    HEIGHT = 137
    WIDTH = 236
    images = [df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH) for df in image_df_list]
    del image_df_list
    gc.collect()
    images = np.concatenate(images, axis=0)
    return images


class KaggleDataset(Dataset):
    def __init__(self, images, df_label, train=True):
        self.df_label = df_label
        self.images = images
        self.train = train
  
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        grapheme_root = self.df_label.grapheme_root.values[index]
        vowel_diacritic = self.df_label.vowel_diacritic.values[index]
        consonant_diacritic = self.df_label.consonant_diacritic.values[index]

        image = self.images[index]
        image = 255 - image
        image = np.array(Image.fromarray(image).convert("RGB"))
        if self.train:
            image1 = Transform(size=128, autoaugment_ratio=1.,)(image)
            image2 = Transform(
                                affine=True,
                                size=128,
                                cutout_ratio=0.3,
                                ssr_ratio=0.3,
                                random_size_crop_ratio=0.3,
                            )(image)

            image1 = (image1).astype(np.float32) / 255.
            image2 = (image2).astype(np.float32) / 255.
            return {
                'image1': torch.tensor(image1, dtype=torch.float),
                'image2': torch.tensor(image2, dtype=torch.float),
                'grapheme_roots': torch.tensor(grapheme_root, dtype=torch.long),
                'vowel_diacritics': torch.tensor(vowel_diacritic, dtype=torch.long),
                'consonant_diacritics': torch.tensor(consonant_diacritic, dtype=torch.long)
            }
        else:
            image = Transform(size=128)(image)
            image = (image).astype(np.float32) / 255.
            return {
                'images': torch.tensor(image, dtype=torch.float),
                'grapheme_roots': torch.tensor(grapheme_root, dtype=torch.long),
                'vowel_diacritics': torch.tensor(vowel_diacritic, dtype=torch.long),
                'consonant_diacritics': torch.tensor(consonant_diacritic, dtype=torch.long)
            }


DATA_PATH = "/home/kazuki/workspace/kaggle_bengali/data/input/"
HEIGHT = 137
WIDTH = 236
SIZE = 128
df_path = '/home/kazuki/workspace/kaggle_bengali/data/input/'


def do_train(model, data_loader, criterion, optimizer, device, config, grad_acc=1):
        model.train()
        train_loss = 0.0
        train_recall = 0.0
        optimizer.zero_grad()
        for idx, (inputs) in tqdm(enumerate(data_loader), total=len(data_loader)):
            x1 = inputs["image1"].to(device, dtype=torch.float)
            x2 = inputs["image2"].to(device, dtype=torch.float)
            grapheme_root = inputs["grapheme_roots"].to(device, dtype=torch.long)
            vowel_diacritic = inputs["vowel_diacritics"].to(device, dtype=torch.long)
            consonant_diacritic = inputs["consonant_diacritics"].to(device, dtype=torch.long)
            
            # if choice <= config.train.cutmix:
            data, targets = cutmix(x1, grapheme_root, vowel_diacritic, consonant_diacritic, 1.)
            logit_grapheme_root1, logit_vowel_diacritic1, logit_consonant_diacritic1 = model(data)
            loss_gr1, loss_vd1, loss_cd1 = cutmix_criterion(logit_grapheme_root1, logit_vowel_diacritic1, logit_consonant_diacritic1, targets, criterion)

            # elif choice <= config.train.cutmix + config.train.mixup:
            # data, targets = mixup(x, grapheme_root, vowel_diacritic, consonant_diacritic, 0.4)
            # logit_grapheme_root, logit_vowel_diacritic, logit_consonant_diacritic = model(data)
            # loss_gr, loss_vd, loss_cd = mixup_criterion(logit_grapheme_root, logit_vowel_diacritic, logit_consonant_diacritic, targets, criterion)
     
            logit_grapheme_root3, logit_vowel_diacritic3, logit_consonant_diacritic3 = model(x2)
            loss_gr3 = criterion(logit_grapheme_root3, grapheme_root)
            loss_vd3 = criterion(logit_vowel_diacritic3, vowel_diacritic )
            loss_cd3 = criterion(logit_consonant_diacritic3, consonant_diacritic)

            logit_grapheme_root = (logit_grapheme_root1 + logit_grapheme_root3) / 2.
            logit_vowel_diacritic = (logit_vowel_diacritic1 + logit_vowel_diacritic3) / 2.
            logit_consonant_diacritic = (logit_consonant_diacritic1 + logit_consonant_diacritic3) / 2.

            loss_gr = (loss_gr1 + loss_gr3) / 2.
            loss_cd = (loss_cd1 + loss_cd3) / 2.
            loss_vd = (loss_vd1 + loss_vd3) / 2.
            train_recall += macro_recall_multi(logit_grapheme_root, grapheme_root, logit_vowel_diacritic, vowel_diacritic, logit_consonant_diacritic, consonant_diacritic)

            ((8 * loss_gr + loss_cd + loss_vd) / grad_acc).backward()

            if (idx % grad_acc) == 0:
                optimizer.step()
                optimizer.zero_grad()
            train_loss += 8 * loss_gr.item() + loss_vd.item() + loss_cd.item()

        train_loss /= len(data_loader)
        train_recall /= len(data_loader)

        return {'train_loss': train_loss, 'train_recall': train_recall}



def do_eval(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.
    valid_recall = 0.

    with torch.no_grad():
        total_loss = 0.
        for inputs in tqdm(data_loader, total=len(data_loader)):
            x = inputs["images"].to(device, dtype=torch.float)
            grapheme_root = inputs["grapheme_roots"].to(device, dtype=torch.long)
            vowel_diacritic = inputs["vowel_diacritics"].to(device, dtype=torch.long)
            consonant_diacritic = inputs["consonant_diacritics"].to(device, dtype=torch.long)

            logit_grapheme_root, logit_vowel_diacritic, logit_consonant_diacritic = model(x)

            loss_gr = criterion(logit_grapheme_root, grapheme_root)
            loss_vd = criterion(logit_vowel_diacritic, vowel_diacritic )
            loss_cd = criterion(logit_consonant_diacritic, consonant_diacritic)

            valid_recall += macro_recall_multi(logit_grapheme_root, grapheme_root, logit_vowel_diacritic, vowel_diacritic, logit_consonant_diacritic, consonant_diacritic)

            total_loss += 8 * loss_gr.item() + loss_vd.item() + loss_cd.item()

        total_loss /= len(data_loader)
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

    train_images = prepare_image(
        df_path, df_path, data_type='train', submission=False)
    train = pd.read_csv(DATA_PATH + "train.csv")
    fold_csv = config.data.params.fold_csv
    folds = pd.read_csv(DATA_PATH + fold_csv)
    idx_fold = config.data.params.idx

    train_ids = folds[folds["fold"]!=idx_fold].index
    train_df = train.iloc[train_ids]
    data_train = train_images[train_ids]
    image_dataset = KaggleDataset(data_train, train_df, train=True)
    dataloaders = {}
    dataloaders['train'] = DataLoader(
        image_dataset,
        batch_size=config.train.batch_size,
        num_workers=16,
        pin_memory=True,
        shuffle=True,
    )

    valid_ids = folds[folds["fold"]==idx_fold].index
    valid_df = train.iloc[valid_ids]
    data_valid = train_images[valid_ids]
    image_dataset = KaggleDataset(data_valid, valid_df, train=False)

    dataloaders['valid'] = DataLoader(
        image_dataset,
        batch_size=config.train.batch_size,
        num_workers=16,
        pin_memory=True,
        shuffle=True,
    )

    model = MODEL_LIST[config.model.version](pretrained=True)
    

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        config.optimizer.params.lr *= torch.cuda.device_count()

    model = model.to(device)

    criterion = get_criterion(config)
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(optimizer, config)

    accumlate_step = 1
    if config.train.accumulation_size > 0:
        accumlate_step = config.train.accumulation_size // config.train.batch_size

    best_valid_recall = 0.0
    if config.train.resume:
        print('resume checkpoints')
        checkpoint = torch.load(config.work_dir + "/checkpoints/" + "best.pth")
        model.load_state_dict(checkpoint['checkpoint'])
        best_valid_recall = checkpoint['best_valid_recall']

    # if config.train.earlyStopping:
    #     early_stopping = EarlyStopping(patience=patience, verbose=True)

    valid_recall = 0.0

    for epoch in range(1, config.train.num_epochs+1):
        print(f'epoch {epoch} start ')
        logger.info(f'epoch {epoch} start ')

        # train code
        metric_train = do_train(model, dataloaders["train"], criterion, optimizer, device, config, accumlate_step)
        metrics_eval = do_eval(model, dataloaders["valid"], criterion, device)
        valid_recall = metrics_eval["valid_recall"]

        scheduler.step(metrics_eval["valid_loss"])

        # early_stopping(metrics_eval["valid_loss"], model)

        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

        print(f'epoch: {epoch} ', metric_train, metrics_eval)
        logger.info(f'epoch: {epoch} {metric_train} {metrics_eval}')
        if valid_recall > best_valid_recall:
            print(f"save checkpoint: best_recall:{valid_recall}")
            logger.info(f"save checkpoint: best_recall:{valid_recall}")
            torch.save({
                'checkpoint': model.state_dict(),
                'epoch': epoch,
                'best_valid_recall': valid_recall,
                }, config.work_dir + "/checkpoints/" + f"{epoch}.pth")
            best_valid_recall = valid_recall


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
