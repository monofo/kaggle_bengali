import gc
import os

import cv2
# import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformars.transform_factory import GridMask, Transform, apply_aug


HEIGHT = 137
WIDTH = 236
SIZE = 128


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

        # image = np.array(Image.fromarray(image).convert("RGB"))
        image = np.stack([image, image, image]).transpose(1,2,0)
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
        num_workers=8,
        idx_fold=None,
        fold_csv="train_with_fold_seed12.csv",
        transforms=None,
        crop=True,
        debug=False,
    ):

    train_images = prepare_image(
        df_path, df_path, data_type='train', submission=False)
    train = pd.read_csv(df_path + "train.csv")
    if idx_fold != -1:
        if fold_csv == "train_v2.csv":
            df_train = pd.read_csv(df_path+ 'train_v2.csv')
            if phase == "train":
                train_idx = np.where((df_train['fold'] != idx_fold) & (df_train['unseen'] == 0))[0]
                data_train = train_images[train_idx]
                train_df = df_train.iloc[train_idx]
                image_dataset = KaggleDataset(data_train, train_df, transforms=transforms)
            else:
                valid_idx = np.where((df_train['fold'] == idx_fold) | (df_train['unseen'] != 0))[0]
                data_valid = train_images[valid_idx]
                valid_df = df_train.iloc[valid_idx]
                image_dataset = KaggleDataset(data_valid, valid_df, transforms=transforms)
        else:
            train = pd.read_csv(df_path + "train.csv")
            folds = pd.read_csv(df_path + fold_csv)

            if phase == "train":
                train_ids = folds[folds["fold"]!=idx_fold].index
                train_df = train.iloc[train_ids]
                data_train = train_images[train_ids]
                image_dataset = KaggleDataset(data_train, train_df, transforms=transforms)
            else:
                valid_ids = folds[folds["fold"]==idx_fold].index
                valid_df = train.iloc[valid_ids]
                data_valid = train_images[valid_ids]
                image_dataset = KaggleDataset(data_valid, valid_df, transforms=transforms)
    else:
        image_dataset = KaggleDataset(train_images, train, transforms=transforms)

    return DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
