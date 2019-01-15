import pandas as pd


def ohe(df, column):
    assert column in df
    complete_df = pd.get_dummies(df, columns=[column])
    return complete_df


# def ints_encoding(df, column, mapping, finaltype=int):
#     assert column in df
#     df[column] = df[column].map(mapping).astype(finaltype)
#     return df

def ints_encoding(df, column, mapping, finaltype=int):
    return ohe(df, column)
