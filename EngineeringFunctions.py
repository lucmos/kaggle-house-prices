import pandas as pd


def ohe(df, column):
    complete_df = pd.get_dummies(df, columns=[column])
    return complete_df
