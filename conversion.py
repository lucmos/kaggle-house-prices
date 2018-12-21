# %% Global imports
from xgboost import XGBRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from pathlib import Path

import numpy as np
import pandas as pd


# %% Pandas initialization
#pd.set_option('display.width', 1000)
#pd.set_option('display.max_columns', 80)
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

# %% Trivial irrelevant features removal
# Utilities has practically always the same value in the train dataset
# complete_df.drop(columns=['Utilities', 'Street'], inplace=True)


# %% Features grouping by NaN "type"/"properties":
complete_df["PoolQC"] = complete_df["PoolQC"].fillna("None")
complete_df["MiscFeature"] = complete_df["MiscFeature"].fillna("None")
complete_df["Alley"] = complete_df["Alley"].fillna("None")
complete_df["Fence"] = complete_df["Fence"].fillna("None")
complete_df["FireplaceQu"] = complete_df["FireplaceQu"].fillna("None")

# Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
complete_df["LotFrontage"] = complete_df.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    complete_df[col] = complete_df[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    complete_df[col] = complete_df[col].fillna(0)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    complete_df[col] = complete_df[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    complete_df[col] = complete_df[col].fillna('None')

complete_df["MasVnrType"] = complete_df["MasVnrType"].fillna("None")
complete_df["MasVnrArea"] = complete_df["MasVnrArea"].fillna(0)

complete_df['MSZoning'] = complete_df['MSZoning'].fillna(
    complete_df['MSZoning'].mode()[0])
complete_df = complete_df.drop(['Utilities'], axis=1)
complete_df["Functional"] = complete_df["Functional"].fillna("Typ")
mode_col = ['Electrical', 'KitchenQual',
            'Exterior1st', 'Exterior2nd', 'SaleType']
for col in mode_col:
    complete_df[col] = complete_df[col].fillna(complete_df[col].mode()[0])

complete_df['MSSubClass'] = complete_df['MSSubClass'].fillna("None")

# MSSubClass=The building class
complete_df['MSSubClass'] = complete_df['MSSubClass'].apply(str)


# Changing OverallCond into a categorical variable
complete_df['OverallCond'] = complete_df['OverallCond'].astype(str)


# Year and month sold are transformed into categorical features.
complete_df['YrSold'] = complete_df['YrSold'].astype(str)
complete_df['MoSold'] = complete_df['MoSold'].astype(str)

# %%
# Features with NaN values
nan_mask = complete_df.isnull().any()
nan_columns = nan_mask[nan_mask].shape[0]


def is_a_valid_feature(x): return x in complete_df


# Feature with NaN values going to ZERO_VALUE
# Those are features with categorical value, but with an intrinsic order so "NaN" ("NA") is the lower possible element.
to_zero = list(filter(is_a_valid_feature, ["PoolQC",
                                           "Fence",
                                           "FireplaceQu",
                                           'GarageFinish', 'GarageQual', 'GarageCond', 'GarageYrBlt', 'GarageArea',
                                           'GarageCars',
                                           'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
                                           'BsmtHalfBath',
                                           'BsmtQual', 'BsmtCond',
                                           'BsmtExposure',  # TODO: mappa il valore No in 0 o 1
                                           'BsmtFinType1', 'BsmtFinType2',  # TODO: da unire
                                           'MasVnrArea']))

# Features with NaN values going to NONE_VALUE.
# Those are features with categorical value, but with no intrinsic order.
to_none = list(filter(is_a_valid_feature,
                      [
                          'MSZoning',  # TODO: Maybe we should split this in sub-features with order
                          "MiscFeature",
                          "Alley",
                          'GarageType',
                          'MasVnrType'
                      ]))

# Features with NaN values going to the MSC one.
# Those are features with unknown value and value property, so that they are mapped to the MSC value for that feature.
to_most_common = list(filter(is_a_valid_feature, [
    'Electrical',
    'KitchenQual',
    'Exterior1st',
    'Exterior2nd',
    'SaleType'
]))

other = list(filter(is_a_valid_feature, ['Functional', 'LotFrontage']))

print('Features = {} | Features with NaN values = {}'.format(
    len(complete_df.keys()), nan_columns))
print('Zero-like NaN features = {} | None-like NaN features = {} | NaN to MSC features = {} | Other = {}'
      .format(len(to_zero), len(to_none), len(to_most_common), len(other)))


# %% NaN removal
for col in to_zero:
    complete_df[col] = complete_df[col].fillna(ZERO_VALUE)

for col in to_none:
    complete_df[col] = complete_df[col].fillna(NONE_VALUE)

for col in to_most_common:
    complete_df[col] = complete_df[col].fillna(complete_df[col].mode()[0])

# Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
complete_df["LotFrontage"] = complete_df.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

complete_df["Functional"] = complete_df["Functional"].fillna("Typ")

# Features with NaN values
nan_mask = complete_df.isnull().any()
nan_columns = nan_mask[nan_mask].shape[0]

assert nan_columns == 0
print(y_train.shape)

# %% Features correlation search & removal (excluding NaN/null values)
# CORRELATION = 0.65
# # Correlation map to see how features are correlated with SalePrice (ignori
# corrmat = complete_df.corr()
# corrmat = corrmat > CORRELATION
# np.fill_diagonal(corrmat.values, False)
#
# print('Shape before removal: {}'.format(complete_df.shape))
# removed = []
# while any(corrmat.any()):
#     for col in corrmat.keys():
#         if corrmat[col].any():
#             complete_df.drop(columns=[col], inplace=True)
#             removed.append(col)
#             break
#     corrmat = complete_df.corr()
#     corrmat = corrmat > CORRELATION
#     np.fill_diagonal(corrmat.values, False)
# print('Shape after removal: {}'.format(complete_df.shape))
# print('Removed columns: {}'.format(removed))

complete_df = complete_df.drop(columns=[
    "TotalBsmtSF",
    "GrLivArea",
    "MasVnrArea",
    "BsmtHalfBath",
    "GarageYrBlt",
    "OpenPorchSF",
    "PoolArea",
    "3SsnPorch",
    "MiscVal",
    "MoSold",
    "LowQualFinSF",
    "HalfBath",
    "FullBath",
    "EnclosedPorch"
])


# %% Categories creation

categorical_mapping = {
    # Exterior material quality
    "ExterQual": ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    # Present condition of the material on the exterior
    "ExterCond": ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    # Height of the basement
    "BsmtQual": [ZERO_VALUE, 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    # General condition of the basement
    "BsmtCond": [ZERO_VALUE, 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    # Walkout or garden level basement walls
    "BsmtExposure": [ZERO_VALUE, 'No', 'Mn', 'Av', 'Gd'],
    # Quality of basement finished area
    "BsmtFinType1": [ZERO_VALUE, 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    # todo Quality of second finished area (if present)
    "BsmtFinType2": [ZERO_VALUE, 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    # Heating quality and condition
    "HeatingQC": ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    # Kitchen quality
    "KitchenQual": ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    # Home functionality rating
    "Functional": ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],
    # Fireplace quality
    "FireplaceQu":  [ZERO_VALUE, 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    # Interior finish of the garage
    "GarageFinish": [ZERO_VALUE, 'Unf', 'RFn', 'Fin'],
    # Garage quality
    "GarageQual": [ZERO_VALUE, 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    # Garage condition
    "GarageCond": [ZERO_VALUE, 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    # Pool quality
    "PoolQC": [ZERO_VALUE, 'Fa', 'TA', 'Gd', 'Ex'],
    # Fence quality
    "Fence": [ZERO_VALUE, 'MnWw', 'GdWo', 'MnPrv', 'GdPrv']
}


# %% Categorical features mapping:
count = 0
# %%
# esegui solo una volta!!
if count == 0:
    for feature, categories in categorical_mapping.items():
        print(categories)
        complete_df[feature] = pd.Categorical(
            complete_df[feature], categories=categories).codes

    # Porta a zero -> 0 le feature solo numeriche
    complete_df = complete_df.replace(to_replace=ZERO_VALUE, value=0)

    print(complete_df[list(categorical_mapping.keys())].head())

    print('Da verificare per bene il funzionamento (ma credo sia corretto)')
    count += 1


# %% Features conversion (basic infer)
complete_df = complete_df.infer_objects()

# %% Add some simple features
# #simplified features
# # complete_df['haspool'] = complete_df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
# # complete_df['has2ndfloor'] = complete_df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
# complete_df['hasgarage'] = complete_df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
# # complete_df['hasbsmt'] = complete_df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
# complete_df['hasfireplace'] = complete_df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
#
# # complete_df['Total_sqr_footage'] = (complete_df['BsmtFinSF1'] + complete_df['BsmtFinSF2'] +
# #                                  complete_df['1stFlrSF'] + complete_df['2ndFlrSF'])
#
# # complete_df['Total_Bathrooms'] = (complete_df['FullBath'] + (0.5*complete_df['HalfBath']) +
# #                                complete_df['BsmtFullBath'] + (0.5*complete_df['BsmtHalfBath']))
#
# complete_df['Total_porch_sf'] = (complete_df['OpenPorchSF'] + complete_df['3SsnPorch'] +
#                               complete_df['EnclosedPorch'] + complete_df['ScreenPorch'] +
#                              complete_df['WoodDeckSF'])

# %% One hot encoding of remaining categorical features

# Feature categoriche senza ordinamento
to_one_hot_encoding = [
    "MSSubClass",  # The building class
    "MSZoning",  # The general zoning classification
    "Street",  # Type of road access
    "Alley",  # Type of alley access
    "LotShape",  # General shape of property
    "LandContour",  # Flatness of the property
    "LotConfig",  # Lot configuration
    "LandSlope",  # Slope of property
    "Neighborhood",  # Physical locations within Ames city limits
    "Condition1",  # Proximity to main road or railroad
    "Condition2",
    "BldgType",  # Type of dwelling
    "HouseStyle",  # Style of dwelling
    "RoofStyle",  # Type of roof
    "RoofMatl",  # Roof material
    "Exterior1st",  # Exterior covering on house
    "Exterior2nd",  # Exterior covering on house (if more than one material)
    "MasVnrType",  # Masonry veneer type
    "Foundation",  # Type of foundation
    "Heating",  # Type of heating
    "CentralAir",  # Central air conditioning
    "Electrical",  # Electrical system
    "GarageType",  # Garage location
    "PavedDrive",  # Paved driveway
    "MiscFeature",  # Miscellaneous feature not covered in other categories
    "MiscVal",  # $Value of miscellaneous feature
    "SaleType",  # Type of sale todo Forse va spezzata in altre feature
    "SaleCondition"]  # Condition of sale
complete_df = pd.get_dummies(complete_df, columns=list(
    filter(is_a_valid_feature, to_one_hot_encoding)))

# %% Regression! Validation

x_train = complete_df[:train_obs]
x_test = complete_df[train_obs:]
assert train_obs == x_train.shape[0]
assert test_obs == x_test.shape[0]

print(x_train.shape)
print(y_train.shape)
x_reduced_train, x_validation, y_reduced_train, y_validation = train_test_split(
    x_train, y_train, test_size=0.40)

print(x_reduced_train.shape)

print(y_reduced_train.shape)
print(x_validation.shape)
print(y_validation.shape)
#
predictor = make_pipeline(RobustScaler(),
                          XGBRegressor(learning_rate=0.01, n_estimators=3460,
                                       max_depth=3, min_child_weight=0,
                                       gamma=0, subsample=0.7,
                                       colsample_bytree=0.7,
                                       objective='reg:linear', nthread=4,
                                       scale_pos_weight=1, seed=27,
                                       reg_alpha=0.00006))


# predictor = make_pipeline(XGBRegressor(learning_rate=0.01, n_estimators=3460,
#                                        max_depth=3, min_child_weight=0,
#                                        gamma=0, subsample=0.7,
#                                        colsample_bytree=0.7,
#                                        objective='reg:linear', nthread=4,
#                                        scale_pos_weight=1, seed=27,
#                                        reg_alpha=0.00006))


predictor.fit(x_reduced_train, y_reduced_train)
y_validation_pred = predictor.predict(x_validation)

y_validation = np.expm1(y_validation)
y_validation_pred = np.expm1(y_validation_pred)
# a = (filter(lambda  y: y <0, y_validation_pred))

# print(list(a))


err = np.sqrt(mean_squared_log_error(y_validation, y_validation_pred))
print("ERROR on validation set: {}".format(err))

print("Done validating")


# %% Test predictions
# predictor = make_pipeline(XGBRegressor(learning_rate=0.01, n_estimators=3460,
#                                        max_depth=3, min_child_weight=0,
#                                        gamma=0, subsample=0.7,
#                                        colsample_bytree=0.7,
#                                        objective='reg:linear', nthread=4,
#                                        scale_pos_weight=1, seed=27,
#                                        reg_alpha=0.00006))

predictor = make_pipeline(RobustScaler(),
                          Lasso(alpha=0.0003, random_state=1, max_iter=50000))


predictor.fit(x_train, y_train)

predictions = predictor.predict(x_test)
result_df = pd.DataFrame()
result_df['Id'] = test_ids
result_df['SalePrice'] = np.expm1(predictions)
result_df.to_csv(Path(dataset_dir, 'predictions.csv'), index=False)
#
print("DONE")
