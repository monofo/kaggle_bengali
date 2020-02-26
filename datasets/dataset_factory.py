import gc
import os

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

DATA_PATH = "/home/kazuki/workspace/kaggle_bengali/data/input/"


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
    def __init__(self, images, df_label, transforms=None):
        self.df_label = df_label
        self.images = images
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        grapheme_root = self.df_label.grapheme_root.values[index]
        vowel_diacritic = self.df_label.vowel_diacritic.values[index]
        consonant_diacritic = self.df_label.consonant_diacritic.values[index]

        image = self.images[index]
        image = 255 - image
        image = np.array(Image.fromarray(image).convert("RGB"))
        if self.transforms is not None:
            image = self.transforms(image)

        image = (image).astype(np.float32) / 255.
        return {
            'images': torch.tensor(image, dtype=torch.float),
            'grapheme_roots': torch.tensor(grapheme_root, dtype=torch.long),
            'vowel_diacritics': torch.tensor(vowel_diacritic, dtype=torch.long),
            'consonant_diacritics': torch.tensor(consonant_diacritic, dtype=torch.long)
        }


def make_loader(
        phase,
        df_path='/home/kazuki/workspace/kaggle_bengali/data/input/',
        batch_size=8,
        num_workers=2,
        idx_fold=None,
        fold_csv="train_with_fold_seed12.csv",
        transforms=None,
        debug=False,
):
    if debug:
        num_rows = 100
    else:
        num_rows = None
    train = pd.read_csv(DATA_PATH + "train.csv")
    train_images = prepare_image(
        df_path, df_path, data_type='train', submission=False)


    folds = pd.read_csv(DATA_PATH + fold_csv)

    if phase == "train":
        train_ids = folds[folds["fold"]!=idx_fold].index
        train_df = train.iloc[train_ids]
        data_train = train_images[train_ids]
        image_dataset = KaggleDataset(data_train, train_df, transforms=transforms)
        # train_sampler = torch.utils.data.distributed.DistributedSampler(image_dataset)
    else:
        valid_ids = folds[folds["fold"]==idx_fold].index
        valid_df = train.iloc[valid_ids]
        data_valid = train_images[valid_ids]
        image_dataset = KaggleDataset(data_valid, valid_df, transforms=transforms)

    return DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )