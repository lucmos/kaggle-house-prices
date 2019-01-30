from collections import Counter
from pprint import pprint

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from fancyimpute import KNN


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
    # info = pd.concat([nullcols, dtypes2], axis=1).sort_values(by=0, ascending=False)
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


def impute(complete_df):
    # complete_df = RobustScaler().fit_transform(complete_df)
    # complete_df = fi.NuclearNormMinimization().fit_transform(complete_df)
    check_missing_values(complete_df)

    index = complete_df.index
    columns = complete_df.columns

    # complete_df = BiScaler().fit_transform(complete_df.values)
    # complete_df = SoftImpute().fit_transform(complete_df)
    # complete_df = KNN().fit_transform(complete_df)
    # complete_df = IterativeSVD().fit_transform(complete_df)
    # complete_df = BiScaler().fit_transform(complete_df.values)
    # complete_df = NuclearNormMinimization().fit_transform(complete_df)
    complete_df = KNN(k=10).fit_transform(complete_df)

    complete_df = pd.DataFrame(complete_df, index=index, columns=columns)
    return complete_df


def add_logs(complete_df):
    # %% Add logs
    def addlogs(res, ls):
        m = res.shape[1]
        for l in ls:
            res = res.assign(newcol=pd.Series(np.log(1.01 + res[l])).values)
            res.columns.values[m] = l + '_log'
            m += 1
        return res

    loglist = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
               'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
               'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
               'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
               'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'YearRemodAdd', 'TotalSF',
               # 'MiscVal',
               ]
    loglist = [x for x in loglist if x in complete_df]
    complete_df = addlogs(complete_df, loglist)
    return complete_df


def add_squares(complete_df):
    def addSquared(res, ls):
        m = res.shape[1]
        for l in ls:
            res = res.assign(newcol=pd.Series(res[l] * res[l]).values)
            res.columns.values[m] = l + '_sq'
            m += 1
        return res

    sqpredlist = ['YearRemodAdd', 'LotFrontage_log',
                  'TotalBsmtSF_log', '1stFlrSF_log', '2ndFlrSF_log', 'GrLivArea_log',
                  'GarageCars_log', 'GarageArea_log',
                  'OverallQual', 'ExterQual', 'BsmtQual', 'GarageQual', 'FireplaceQu', 'KitchenQual']
    sqpredlist = [x for x in sqpredlist if x in complete_df]
    complete_df = addSquared(complete_df, sqpredlist)
    return complete_df


def resolve_skewness(complete_df, numeric_features):
    # %% ~~~~~ Resolve skewness ~~~~ TODO camuffa codice
    from scipy.stats import skew

    skew_features = complete_df[numeric_features].apply(lambda x: skew(x)).sort_values(ascending=False)
    skews = pd.DataFrame({'skew': skew_features})

    print()
    print('--------- SKEW OF FEATURES ----------')
    print(skew_features)
    print()

    from scipy.special import boxcox1p
    from scipy.stats import boxcox_normmax

    high_skew = skew_features[skew_features > 0.5]
    high_skew = high_skew
    skew_index = high_skew.index

    for i in skew_index:
        complete_df[i] = boxcox1p(complete_df[i], boxcox_normmax(complete_df[i] + 1))

    # Check it is adjusted
    skew_features2 = complete_df[numeric_features].apply(lambda x: skew(x)).sort_values(ascending=False)
    skews2 = pd.DataFrame({'skew': skew_features2})
    print()
    print('--------- SKEW OF FEATURES AFTER NORMALIZATION ----------')
    print(skew_features2)
    print()
    return complete_df


def show_correlation(df):
    sns.set(style="white")

    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = [(0, 1, 0, 1), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1),
            (0, 0, 0, 1)]

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.1, linecolor="gray", cbar_kws={"shrink": .5})

    plt.show()


def compute_correlation(df):
    # show_correlation(df)

    # %% Features correlation search & removal (excluding NaN/null values)
    CORRELATION = 0.75
    # Correlation map to see how features are correlated with SalePrice (ignori
    corrmat = df.corr().values

    condlist = [corrmat > CORRELATION, corrmat < -CORRELATION]
    choicelist = [True, True]

    mask = np.select(condlist, choicelist, default=False)
    np.fill_diagonal(mask, False)

    l = []
    for row_index, row in enumerate(mask):
        for col_index, cell in enumerate(row):
            if cell:
                l.append((corrmat[row_index][col_index], tuple(sorted([df.columns[row_index], df.columns[col_index]]))))

    l = list(set(l))
    l = sorted(l, key=lambda x: abs(x[0]), reverse=True)
    pprint(l)
    return

    return
    print('Shape before removal: {}'.format(df.shape))
    removed = []
    while any(corrmat.any()):
        for col in corrmat.keys():
            if corrmat[col].any():
                complete_df.drop(columns=[col], inplace=True)
                removed.append(col)
                break
        corrmat = complete_df.corr()
        corrmat = corrmat > CORRELATION
        np.fill_diagonal(corrmat.values, False)
    print('Shape after removal: {}'.format(complete_df.shape))
    print('Removed columns: {}'.format(removed))
