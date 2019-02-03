# %% Global imports
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# %% Pandas initialization
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 80)
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

# %% Some constant
dataset_dir = 'dataset'
ZERO_VALUE = 'zero'
NONE_VALUE = 'None'

# %% Train & test loading
train_df = pd.read_csv(Path(dataset_dir, 'train.csv'))
test_df = pd.read_csv(Path(dataset_dir, 'test.csv'))

# %% Removing outliers
# As suggested by many participants, we remove several outliers
train_df.drop(train_df[(train_df['OverallQual'] < 5) & (
        train_df['SalePrice'] > 200000)].index, inplace=True)
train_df.drop(train_df[(train_df['GrLivArea'] > 4000) & (
        train_df['SalePrice'] < 300000)].index, inplace=True)
train_df.reset_index(drop=True, inplace=True)

# %% Log Sale Price
# train_df['SalePrice'] = np.log(train_df['SalePrice'])
y_train = np.log1p(train_df["SalePrice"])

# %% Concatenate train & test
train_ids = train_df['Id']
test_ids = test_df['Id']

# Dropping also the y_train values (already stored)
train_df = train_df.drop(columns=['Id', 'SalePrice'])
test_df = test_df.drop(columns=['Id'])

train_obs = train_df.shape[0]
test_obs = test_df.shape[0]

# Merge train and test dataframes to improve features ranges
complete_df = pd.concat([train_df, test_df])

assert complete_df.shape[0] == train_obs + test_obs
qualities_dict = {'None': 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

feature = 'LowQualFinSF'

nan_mask = complete_df.isnull().any()
nan_columns = nan_mask[nan_mask]
print('Feature: {}'.format(feature))
print('Contains NaN values: {}'.format(feature in nan_columns))
print("Number of values:", len(Counter(complete_df[feature])))
print('Number of NaN in feature: {}'.format(complete_df[feature].isna().sum()))
counter = Counter(complete_df[feature])
print(counter)
print('Values missing in the training set: {}'.format(set(train_df[feature]).difference(set(test_df[feature]))))

plt.bar(counter.keys(), counter.values())
plt.show()
# for i, row in complete_df[['MiscVal', 'MiscFeature']].iterrows():
#     types = {row['MiscFeature'], row['MiscVal']}
#     if not row['MiscFeature'] is np.nan:
#         print(row['MiscFeature'], row['MiscVal'])

# for x in Counter(complete_df[feature]):
#     print(x)
#Counter({2: 1529, 1: 1308, 3: 63, 0: 12, 4: 4})
