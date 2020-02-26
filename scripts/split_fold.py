import numpy as np
import pandas as pd
#split data
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

#get data
nfold = 5
seed = 12

train_df = pd.read_csv("/home/kazuki/workspace/kaggle_bengali/data/input/train.csv")
train_df['id'] = train_df['image_id'].apply(lambda x: int(x.split('_')[1]))

X, y = train_df[['id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values[:,0], train_df.values[:,1:]

train_df['fold'] = np.nan

mskf = MultilabelStratifiedKFold(n_splits=nfold, random_state=seed)
for i, (_, test_index) in enumerate(mskf.split(X, y)):
    train_df.iloc[test_index, -1] = i
    
train_df =  train_df['fold'].astype('int')

#output
train_df.to_csv('/home/kazuki/workspace/kaggle_bengali/data/input/train_with_fold_seed12.csv', index = False, header=True)
