from collections import Counter

import pandas as pd


def ohe(df, column):
    assert column in df
    complete_df = pd.get_dummies(df, columns=[column])
    return complete_df


def ints_encoding(df, column, mapping, finaltype=int):
    assert column in df, "{} no in df".format(column)
    df[column] = df[column].map(mapping)
    # print(Counter(df[column]))
    df[column] = df[column].astype(finaltype)
    return df
# todo: stiamo facendo sia ohe che ints ora


# def ints_encoding(df, column, mapping, finaltype=int):
#     return ohe(df, column)
