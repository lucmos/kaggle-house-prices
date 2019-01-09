# %% Global imports
from pathlib import Path

import numpy as np
import pandas as pd


def ohe(column):
    # TODO Beautiful things
    pass


# %% Pandas initialization
# pd.set_option('display.width', 1000)
# pd.set_option('display.max_columns', 80)
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

qualities_dict = {NONE_VALUE: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

# TODO: Handle the NaN everywhere
original_columns_to_drop = []
# MSSubClass: Identifies the type of dwelling involved in the sale.
# Categorical feature, maybe with some order.
complete_df['MSSubClass'] = ohe(complete_df['MSSubClass'])

# MSZoning: The general zoning classification
# Categorical feature, maybe we should split this feature into two (Residential/Other) because of the order.
complete_df['MSZoning'] = ohe(complete_df['MSZoning'])

# LotFrontage: Linear feet of street connected to property
# Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
complete_df["LotFrontage"] = complete_df.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

# LotArea: Lot size in square feet
# Rescaling?

# Street: Type of road access to property
# ??

# Alley: Type of alley access to property
complete_df["Alley"] = ohe(complete_df["Alley"].fillna(NONE_VALUE))

# LotShape: General shape of property
# Counter({'Reg': 1859, 'IR1': 966, 'IR2': 76, 'IR3': 15})
# So we can just transform this feature into a boolean one: IsLotShapeRegular
complete_df["IsRegularLotShape"] = (complete_df["LotShape"] == "Reg") * 1

# LandContour: Flatness of the property
# Counter({'Lvl': 2622, 'HLS': 120, 'Bnk': 115, 'Low': 59})
# So we can just transform this feature into a boolean one: IsContourLandLevel
complete_df["IsContourLandLevel"] = (complete_df["LandContour"] == "Lvl") * 1

# Utilities: Type of utilities available
# Counter({'AllPub': 2913, nan: 2, 'NoSeWa': 1})
# So it's an irrelevant feature, should be dropped all together
original_columns_to_drop.append('Utilities')

# LotConfig: Lot configuration
complete_df['LotConfig'] = ohe(complete_df['LotConfig'])

# LandSlope: Slope of property
# Counter({'Gtl': 2776, 'Mod': 124, 'Sev': 16})
# So we can just transform this feature into a boolean one: IsSlopeGentle
complete_df['IsSlopeGentle'] = (complete_df['LandSlope'] == 'Gtl') * 1
original_columns_to_drop.append('LandSlope')

# Neighborhood: Physical locations within Ames city limits
# According to other partecipants, these are the good neighborhoods: 'NridgHt', 'Crawfor', 'StoneBr', 'Somerst',
# 'NoRidge'. Let's create a new boolean feature representing the belonging to one of these good neighborhoods.
complete_df['IsGoodNeighborhood'] = np.array([x in ('NridgHt', 'Crawfor', 'StoneBr', 'Somerst', 'NoRidge') for x in
                                              complete_df['Neighborhood']]) * 1
original_columns_to_drop.append('Neighborhood')


# Condition1: Proximity to various conditions
# Condition2: Proximity to various conditions (if more than one is present)

# We can merge these two features keeping in mind that 'Norm" means "nothing particularly relevant is nearby" so we can
# ignore it when paired with some other condition.
# Furthermore there are ONLY 17 observations with more than 1 condition (both different than Norm)!
# Of those 17, only 14 all have the "Feedr" condition, so we can discard that and keep only the other one of the pair.
# Here are the last 3: {'PosA', 'Artery'} {'RRAn', 'Artery'} {'Artery', 'RRNn'}
# We can just remove "Artery" from the pairs.
# Let's not forget it's a categorical feature.
def conditions_merge(row):
    conditions = {row['Condition1'], row['Condition2']}
    condition = 'Norm'
    conditions.discard('Norm')
    if len(conditions) == 2:
        conditions.discard('Feedr' if 'Feedr' in conditions else 'Artery')
        condition = conditions.pop()
    elif len(conditions) == 1:
        condition = conditions.pop()

    return condition


complete_df["Condition"] = ohe(complete_df.apply(conditions_merge, axis=1)
                               .replace(to_replace='Norm', value=NONE_VALUE))

original_columns_to_drop.extend(['Condition1', 'Condition2'])

# BldgType: Type of dwelling
complete_df['BldgType'] = ohe(complete_df['BldgType'])

# HouseStyle: Style of dwelling
# Might have an order!
# TODO There are values for this feature that do not appear in the test set, so we should remove the columns
#  representing them (after the OneHotEncoding) to avoid overfitting: {'2.5Fin'}
complete_df['HouseStyle'] = ohe(complete_df['HouseStyle'])

# OverallQual: Rates the overall material and finish of the house
# The distribution of these values shows that they can be grouped into 3 bins, meaning: bad - average - good
# Counter({5: 825, 6: 731, 7: 600, 8: 342, 4: 225, 9: 107, 3: 40, 10: 29, 2: 13, 1: 4})
bins = {range(1, 4): 1, range(4, 7): 2, range(7, 11): 3}


def overall_qual_simplify(row):
    qual = row['OverallQual']
    for bin, simple_qual in bins.items():
        if qual in bin:
            return simple_qual


complete_df['OverallQualSimplified'] = complete_df.apply(overall_qual_simplify, axis=1)
original_columns_to_drop.append('OverallQual')


# OverallCond: Rates the overall condition of the house
# The distribution of these values shows that they can be grouped into 3 bins, meaning: bad - average - good
# Counter({5: 1643, 6: 530, 7: 390, 8: 144, 4: 101, 3: 50, 9: 41, 2: 10, 1: 7})

def overall_cond_simplify(row):
    qual = row['OverallCond']
    for bin, simple_qual in bins.items():
        if qual in bin:
            return simple_qual


complete_df['OverallCondSimplified'] = complete_df.apply(overall_cond_simplify, axis=1)
original_columns_to_drop.append('OverallCond')

# YearBuilt: Original construction date
# YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)
# YrSold: Year Sold (YYYY)

# There's a lot going on here. First of all, we can bin the years of construction (going from 1872 to 2010, so spanning
# 128 years) into 7 ranges (to obtain ~20 years for each bin):
complete_df['YearBuiltBinned'] = pd.cut(complete_df['YearBuilt'], 7)

# Now, we don't care about the exact remodel date, we just want to know if it has been remodeled:
complete_df['IsRemodeled'] = (complete_df["YearRemodAdd"] != complete_df["YearBuilt"]) * 1

# Is the remodel a very recent one (same year of the sale)?
complete_df['IsRemodelRecent'] = (complete_df["YearRemodAdd"] == complete_df["YrSold"]) * 1

# Is this house new (year of sale same as the year of construction)?
complete_df["IsNewHouse"] = (complete_df["YearBuilt"] == complete_df["YrSold"]) * 1

original_columns_to_drop.extend(['YearRemodAdd', 'YrSold', 'YearBuilt'])

# RoofStyle: Type of roof
complete_df['RoofStyle'] = ohe(complete_df['RoofStyle'])

# RoofMatl: Roof material
# TODO There are values for this feature that do not appear in the test set, so we should remove the columns
#  representing them (after the OneHotEncoding) to avoid overfitting: {'Roll', 'Metal', 'Membran'}
complete_df['RoofMatl'] = ohe(complete_df['RoofMatl'])

# Exterior1st: Exterior covering on house
# Exterior2nd: Exterior covering on house (if more than one material)

# We can merge these two features after the first OHE, keeping in mind that we must assign 1 to the 2nd relevant column.
# There is also a misspell of some value 'CmentBd' and 'Wd Shng'
complete_df['Exterior1st'].replace(to_replace='CmentBd', value='CemntBd', inplace=True)
complete_df['Exterior2nd'].replace(to_replace='CmentBd', value='CemntBd', inplace=True)

complete_df['Exterior1st'].replace(to_replace='Wd Shng', value='Wd Sdng', inplace=True)
complete_df['Exterior2nd'].replace(to_replace='Wd Shng', value='Wd Sdng', inplace=True)

# TODO
original_columns_to_drop.extend(['Exterior1st', 'Exterior2nd'])

# MasVnrType: Masonry veneer type
# MasVnrArea: Masonry veneer area in square feet
#
# First of all, if MasVnrArea is different than 0/NaN we can be sure that the house has a masonry veneer.
# The 'BrkFace' type is by large the most common one (after the 'None' type).
# So, for the MasVnrType feature, we fill NONE_VALUE when we don't get more info from the MasVnrArea,
# otherwise 'BrkFace'.
# Let's not forget that MasVnrType is a categorical feature.
temp_df = complete_df[["MasVnrType", "MasVnrArea"]].copy()
indexes_to_fill = (complete_df["MasVnrArea"] != 0) & \
                  ((complete_df["MasVnrType"] == "None") | (complete_df["MasVnrType"].isnull()))
complete_df.loc[indexes_to_fill, "MasVnrType"] = "BrkFace"
complete_df['MasVnrType'].fillna(NONE_VALUE)
complete_df['MasVnrType'] = ohe(complete_df['MasVnrType'])

# ExterQual: Evaluates the quality of the material on the exterior
# This is a categorical feature, but with an order which should be preserved!
complete_df["ExterQual"] = complete_df["ExterQual"].map(qualities_dict).astype(int)

# ExterCond: Evaluates the present condition of the material on the exterior
# This is a categorical feature, but with an order which should be preserved!
complete_df["ExterCond"] = complete_df["ExterCond"].map(qualities_dict).astype(int)

# Foundation: Type of foundation
complete_df['Foundation'] = ohe(complete_df['Foundation'])

# BsmtQual: Evaluates the height of the basement
# BsmtCond: Evaluates the general condition of the basement
# BsmtExposure: Refers to walkout or garden level walls
# BsmtFinType1: Rating of basement finished area
# BsmtFinSF1: Type 1 finished square feet
# BsmtFinType2: Rating of basement finished area (if multiple types)
# BsmtFinSF2: Type 2 finished square feet
# BsmtUnfSF: Unfinished square feet of basement area
# TotalBsmtSF: Total square feet of basement area
#
# TODO We should remove discordant data (there can't be a single basement-feature with a "no basement" meaning if at
#  least another one is present

# Heating: Type of heating
# HeatingQC: Heating quality and condition
# TODO There are values for the HeatingQC feature that do not appear in the test set, so we should remove the columns
#  representing them (after the OneHotEncoding) to avoid overfitting: {'Floor', 'OthW'}
complete_df['Heating'] = ohe(complete_df['Heating'])
complete_df['HeatingQC'] = complete_df["HeatingQC"].map(qualities_dict).astype(int)

# CentralAir: Central air conditioning
# We can just transform this feature into a boolean one.
complete_df["HasCentralAir"] = (complete_df["CentralAir"] == "Y") * 1
original_columns_to_drop.append('CentralAir')

# Electrical: Electrical system
# Counter({'SBrkr': 2668, 'FuseA': 188, 'FuseF': 50, 'FuseP': 8, 'Mix': 1, nan: 1})
# Therefore, we can set the only NaN value to 'SBrkr' which is by far the most common one.
# TODO There are values for this feature that do not appear in the test set, so we should remove the columns
#  representing them (after the OneHotEncoding) to avoid overfitting: {'Mix'}
complete_df['Electrical'] = complete_df['Electrical'].fillna("SBrkr")

# 1stFlrSF: First Floor square feet
# 2ndFlrSF: Second floor square feet
# We can build a new feature from those two and the basement info: the total area of the two floors + the basement
complete_df["TotalAreaFlrAndBsmnt"] = complete_df["1stFlrSF"] + complete_df["2ndFlrSF"] + complete_df['TotalBsmtSF']

# TODO: loglist = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
#                  'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',
#                  'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
#                  'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',
#                  'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YearRemodAdd','TotalSF']
#  def addlogs(res, ls):
#     m = res.shape[1]
#     for l in ls:
#         res = res.assign(newcol=pd.Series(np.log(1.01+res[l])).values)
#         res.columns.values[m] = l + '_log'
#         m += 1
#     return res

# TODO area_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
#                  'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF',
#                  'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'LowQualFinSF', 'PoolArea' ]
#     all_df["TotalArea"] = all_df[area_cols].sum(axis=1)

# LowQualFinSF: Low quality finished square feet (all floors)
#
# GrLivArea: Above grade (ground) living area square feet
#
# BsmtFullBath: Basement full bathrooms
#
# BsmtHalfBath: Basement half bathrooms
#
# FullBath: Full bathrooms above grade
#
# HalfBath: Half baths above grade
#
# Bedroom: Bedrooms above grade (does NOT include basement bedrooms)
#
# KitchenAbvGr: Kitchens above grade

# KitchenQual: Kitchen quality
# Counter({'TA': 1492, 'Gd': 1150, 'Ex': 203, 'Fa': 70, nan: 1})
# Let's fill the single NaN value with the most common one (that also stands for "average"!)
complete_df['KitchenQual'] = complete_df['KitchenQual'].fillna('TA')

# TotRmsAbvGrd

# Functional: Home functionality (Assume typical unless deductions are warranted)
# This is a categorical feature, but with order! (higher value means more functionalities)
# Counter({'Typ': 2715, 'Min2': 70, 'Min1': 64, 'Mod': 35, 'Maj1': 19, 'Maj2': 9, 'Sev': 2, nan: 2})
# Let's assume that the NaN values here are 'Typ' (that also stands for "typical"!)
complete_df['Functional'] = complete_df['KitchenQual'].fillna('Typ')
complete_df["Functional"] = complete_df["Functional"].map(
    {NONE_VALUE: 0, "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4,
     "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}).astype(int)

# Fireplaces: Number of fireplaces
# Counter({0: 1420, 1: 1267, 2: 218, 3: 10, 4: 1})
# FireplaceQu: Fireplace quality
# Counter({nan: 1420, 'Gd': 741, 'TA': 592, 'Fa': 74, 'Po': 46, 'Ex': 43})

# First, let's simplify the 'Fireplaces' feature into a boolean one
complete_df['HasFireplaces'] = (complete_df["Fireplaces"] >= 1) * 1

# Now, let's map the NaN values of 'FireplaceQu' to NONE_VALUE, which will be mapped to 0 meaning there's no fireplace.
# We can do this since the rows with false "HasFireplaces" are the same with NaN 'FireplaceQu'!
complete_df['FireplaceQu'] = complete_df['FireplaceQu'].fillna(NONE_VALUE).map(qualities_dict).astype(int)
# TODO Here 'FA' means that there's a fireplace in the basement, so THERE IS A BASEMENT. Check for collisions!


# GarageType: Garage location
# TODO: Feature with order? I think so
# GarageYrBlt: Year garage was built
#
# GarageFinish: Interior finish of the garage

# GarageCars: Size of garage in car capacity
#
# GarageArea: Size of garage in square feet
#
# GarageQual: Garage quality
#
# GarageCond: Garage condition
#
# TODO We should remove discordant data (there can't be a single garage-feature with a "no garage" meaning if at
#  least another one is present

# PavedDrive: Paved driveway
# Counter({'Y': 2638, 'N': 216, 'P': 62})
# Let's create a new boolean feature with the meaning "has a paved drive?"
complete_df['HasPavedDrive'] = (complete_df["PavedDrive"] == "Y") * 1
original_columns_to_drop.append('PavedDrive')

# WoodDeckSF: Wood deck area in square feet
#
# OpenPorchSF: Open porch area in square feet
#
# EnclosedPorch: Enclosed porch area in square feet
#
# 3SsnPorch: Three season porch area in square feet
#
# ScreenPorch: Screen porch area in square feet

# PoolArea: Pool area in square feet
# Counter({0: 2904, 512: 1, 648: 1, 576: 1, 555: 1, 519: 1, 738: 1, 144: 1, 368: 1, 444: 1, 228: 1, 561: 1, 800: 1})
# PoolQC: Pool quality
# Counter({nan: 2907, 'Ex': 4, 'Gd': 3, 'Fa': 2})
# Let's just merge those two features into a simple 'has a pool?'
complete_df['HasPool'] = (complete_df["PoolArea"] >= 1) * 1
original_columns_to_drop.extend(['PoolArea', 'PoolQC'])

# Fence: Fence quality
# This is a categorical feature, but with order! (higher value means better fence)
# Counter({nan: 2345, 'MnPrv': 329, 'GdPrv': 118, 'GdWo': 112, 'MnWw': 12})
# Let's map the NaN values to NONE_VALUE which will then be mapped to a 0 quality.
complete_df['Fence'].fillna(NONE_VALUE)
complete_df["Fence"] = complete_df["Fence"].map(
    {NONE_VALUE: 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}).astype(int)


# MiscFeature: Miscellaneous feature not covered in other categories
# Counter({nan: 2811, 'Shed': 95, 'Gar2': 5, 'Othr': 4, 'TenC': 1})
# MiscVal: $Value of miscellaneous feature
# Given this distribution, we can assume that the only useful info in this feature is the presence of a shed.
# Let's create a boolean feature representing that keeping in mind the value of MiscVal that could be 0 (no shed!).
def has_shed(row):
    if row['MiscFeature'] == 'Shed' and row['MiscVal'] > 0:
        return 1
    return 0


complete_df['HasShed'] = complete_df.apply(has_shed, axis=1)
original_columns_to_drop.extend(['MiscFeature', 'MiscVal'])

# MoSold: Month Sold (MM)
# TODO Someone uses this feature to retrive the "highest sales seasons", should we?

# YrSold: Year Sold (YYYY)

# SaleType: Type of sale
# Counter({'WD': 2524, 'New': 237, 'COD': 87, 'ConLD': 26, 'CWD': 12, 'ConLI': 9, 'ConLw': 8, 'Oth': 7,
# 'Con': 5, nan: 1})
# Let's fill the single NaN value to the most common one (WD)
# TODO: Is there some type of order?
complete_df['SaleType'] = complete_df['SaleType'].fillna('WD')

# SaleCondition: Condition of sale
# TODO: +.-


# Let's remove all the original features we don't need anymore (because new ones have been added or because they're
# irrelevant and so on)
complete_df.drop(columns=original_columns_to_drop)
