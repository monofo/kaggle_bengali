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
from sklearn.metrics.pairwise import cosine_similarity

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
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        optimizer.zero_grad()
        

        for idx, (inputs) in tqdm(enumerate(data_loader), total=len(data_loader)):
            choice = np.random.rand(1)
            x = inputs["images"].to(device, dtype=torch.float)
            grapheme_id = inputs["grapheme_id"].to(device, dtype=torch.long)
            
            outputs = model(x)
            loss_metric = criterion(outputs[1], F.one_hot(grapheme_id, 1295).float())
            loss_0 = nn.CrossEntropyLoss()(outputs[0], grapheme_id)
            pred = outputs[1].argmax(1).detach()
            train_acc += (grapheme_id == pred).type(torch.FloatTensor).mean().cpu().numpy()
            # train_metric = cosine_similarity(outputs[0], F.one_hot(grapheme_id, 1295).float()))
            loss = loss_metric * .5 + loss_0 * .5

            (loss / grad_acc).backward()

            if (idx % grad_acc) == 0:
                optimizer.step()
                optimizer.zero_grad()
            train_loss += loss.item()

        train_loss /= len(data_loader)
        train_acc /= len(data_loader)

        return {'train_loss': train_loss, 'train_acc': train_acc}

def do_eval(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.
    valid_recall = 0.
    val_acc = 0
    with torch.no_grad():
        total_loss = 0.
        for inputs in tqdm(data_loader, total=len(data_loader)):
            x = inputs["images"].to(device, dtype=torch.float)
            grapheme_id = inputs["grapheme_id"].to(device, dtype=torch.long)

            outputs = model(x)
            loss_metric = criterion(outputs[1], F.one_hot(grapheme_id, 1295).float())
            loss_0 = nn.CrossEntropyLoss()(outputs[0], grapheme_id)
            pred = outputs[1].argmax(1).detach()
            val_acc += (grapheme_id == pred).type(torch.FloatTensor).mean().cpu().numpy()
            # valid_metric = cosine_similarity(outputs[0], F.one_hot(grapheme_id, 1296).float()))
            loss = loss_metric * .5 + loss_0 * .5

            total_loss += loss
        total_loss /= len(data_loader)
        val_acc /= len(data_loader)
    
        metrics = {'valid_loss': total_loss, "valid_metric": val_acc}
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
    model = MODEL_LIST[config.model.version](back_bone=config.model.back_bone, out_dim = 1295)
    

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        config.optimizer.params.lr *= torch.cuda.device_count()
        torch.backends.cudnn.benchmark = True

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
        checkpoint = torch.load("/home/koga/workspace/kaggle_bengali/result/" + config.train.path)
        model.load_state_dict(fix_model_state_dict(checkpoint['checkpoint']))
        # model.load_state_dict(checkpoint['checkpoint'])
        # best_valid_recall = checkpoint['best_valid_recall']

    # if config.train.earlyStopping:
    #     early_stopping = EarlyStopping(patience=patience, verbose=True)

    valid_recall = 0.0

    for epoch in range(1, config.train.num_epochs+1):
        print(f'epoch {epoch} start')
        logger.info(f'epoch {epoch} start ')

        metric_train = do_train(model, dataloaders["train"], criterion, optimizer, device, config, epoch, accumlate_step)
        torch.cuda.empty_cache()
        metrics_eval = do_eval(model, dataloaders["valid"], criterion, device)
        torch.cuda.empty_cache()
        valid_recall = metrics_eval["valid_metric"]

        scheduler.step(metrics_eval["valid_loss"])


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
