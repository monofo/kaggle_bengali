import cv2
import gc
from tqdm import tqdm
import zipfile
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

HEIGHT = 137
WIDTH = 236
SIZE = 128

TRAIN = ['../data/input/train_image_data_0.parquet',
         '../data/input/train_image_data_1.parquet',
         '../data/input/train_image_data_2.parquet',
         '../data/input/train_image_data_3.parquet']

PATH = '../data/input/'


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def crop_resize(img0, size=SIZE, pad=16):
    # crop a box around pixels large than the threshold 
    # some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    # cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax, xmin:xmax]
    # remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax - xmin, ymax - ymin
    l = max(lx,ly) + pad
    # make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,(size,size))


def Resize(df,size=128):
    resized = {} 
    df = df.set_index('image_id')
    for i in tqdm(range(df.shape[0])):
       # image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))
        image0 = 255 - df.loc[df.index[i]].values.reshape(137,236).astype(np.uint8)
    #normalize each image by its max val
        img = (img0*(255.0/img0.max())).astype(np.uint8)
        image = crop_resize(img)
        resized[df.index[i]] = image.reshape(-1)
    resized = pd.DataFrame(resized).T.reset_index()
    resized.columns = resized.columns.astype(str)
    resized.rename(columns={'index':'image_id'},inplace=True)
    return resized


for i in range(4):
    data = pd.read_parquet(PATH+f'train_image_data_{i}.parquet')
    data =Resize(data)
    data.to_feather(f'../data/input/train_data_{i}_{SIZE}.feather')
    del data
    gc.collect()