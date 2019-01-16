from collections import Counter

import pandas as pd
import numpy as np


def count(complete_df, feature, blocking=False):
    assert feature in complete_df, '{} not in df'.format(feature)
    print(feature)
    print("Number of values:", len(Counter(complete_df[feature])))
    print('Number of NaN: {}'.format(complete_df[feature].isna().sum()))
    print(Counter(complete_df[feature]))
    if blocking:
        assert False


def check_missing_values(complete_df):
    nulls = np.sum(complete_df.isnull())
    nullcols = nulls.loc[(nulls != 0)]
    dtypes = complete_df.dtypes
    dtypes2 = dtypes.loc[(nulls != 0)]
    info = pd.concat([nullcols, dtypes2], axis=1).sort_values(by=0, ascending=False)
    # print(info)
    print("There are", len(nullcols), "columns with missing values")


def ohe(df, column):
    assert column in df
    complete_df = pd.get_dummies(df, columns=[column])
    return complete_df


def ints_encoding(df, column, mapping):
    assert column in df, "{} no in df".format(column)
    df[column] = df[column].map(mapping)
    return df
