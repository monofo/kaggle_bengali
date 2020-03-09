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
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import (CheckpointCallback,
                                   CriterionAggregatorCallback,
                                   CriterionCallback, DiceCallback,
                                   EarlyStoppingCallback, IouCallback,
                                   MixupCallback, OptimizerCallback)
from catalyst.utils import get_device
from sklearn.model_selection import train_test_split
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
from utils.callbacks import HMacroAveragedRecall

os.environ["CUDA_VISIBLE_DEVICES"]='1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run(config_file):
    config = load_config(config_file)

    config.work_dir = '/home/kazuki/workspace/kaggle_bengali/result/' + config.work_dir
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
    # we have multiple criterions
    criterion = {
        "ce": nn.CrossEntropyLoss(),
        # Define your awesome losses in here. Ex: Focal, lovasz, etc
    }
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(optimizer, config)

    # model runner
    runner = SupervisedRunner(
        device=device,
        input_key="images",
        output_key=("logit_grapheme_root", "logit_vowel_diacritic", "logit_consonant_diacritic"),
        input_target_key=("grapheme_roots", "vowel_diacritics", "consonant_diacritics"),
    )

    callbacks = []

    if config.train.early_stop_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            patience=config.train.early_stop_patience))

    if config.train.accumulation_size > 0:
        accumulation_steps = config.train.accumulation_size // config.train.batch_size
        callbacks.extend(
            [
             OptimizerCallback(accumulation_steps=accumulation_steps)]
        )

    # to resume from check points if exists
    if os.path.exists(config.work_dir + '/checkpoints/best.pth'):
        callbacks.append(CheckpointCallback(
            resume=config.work_dir + '/checkpoints/last_full.pth'))

    callbacks.extend(
        [MixupCallback(
            input_key="grapheme_roots",
            output_key="logit_grapheme_root",
            criterion_key='ce',
            prefix='loss_gr',
        ),  
        MixupCallback(
            input_key="vowel_diacritics",
            output_key="logit_vowel_diacritic",
            criterion_key='ce',
            prefix='loss_wd',
        ),
        MixupCallback(
            input_key="consonant_diacritics",
            output_key="logit_consonant_diacritic",
            criterion_key='ce',
            prefix='loss_cd',
        ),

        CriterionAggregatorCallback(
            prefix="loss",
            loss_aggregate_fn="weighted_sum",
            loss_keys={"loss_gr": 2.0, "loss_wd": 1.0, "loss_cd": 1.0},
        ),
        
        # metrics
        HMacroAveragedRecall(),
        ]
    )

    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=dataloaders,
        logdir=config.work_dir,
        num_epochs=config.train.num_epochs,
        main_metric="hmar",
        minimize_metric=False,
        monitoring_params=None,
        callbacks=callbacks,
        verbose=True,
        fp16=False,
    )


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
