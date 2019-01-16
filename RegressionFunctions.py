import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso, LassoCV, RidgeCV, ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

from mlxtend.regressor import StackingCVRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# %% Global variables
kfolds = KFold(n_splits=10, shuffle=True, random_state=23)
alphas_alt = [14.49, 14.61, 14.69, 14.81, 14.89, 15.01, 15.09, 15.21, 15.29, 15.41, 15.490]
alphas2 = [0.000051, 0.00009, 0.00021, 0.00029, 0.00041, 0.00051, 0.00059, 0.00071, 0.00078]


# %% Build stack gen model
def get_stack_gen_model():
    #setup models
    ridge = make_pipeline(RobustScaler(),
                          RidgeCV(alphas = alphas_alt, cv=kfolds))

    lasso = make_pipeline(RobustScaler(),
                          LassoCV(max_iter=1e7, alphas = alphas2,
                                  random_state = 42, cv=kfolds))

    e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
    e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
    elasticnet = make_pipeline(RobustScaler(),
                               ElasticNetCV(max_iter=1e7, alphas=e_alphas,
                                            cv=kfolds, l1_ratio=e_l1ratio))

    lightgbm = make_pipeline(RobustScaler(),
                            LGBMRegressor(objective='regression',num_leaves=5,
                                          learning_rate=0.05, n_estimators=720,
                                          max_bin = 55, bagging_fraction = 0.8,
                                          bagging_freq = 5, feature_fraction = 0.2319,
                                          feature_fraction_seed=9, bagging_seed=9,
                                          min_data_in_leaf =6,
                                          min_sum_hessian_in_leaf = 11))

    xgboost = make_pipeline(RobustScaler(),
                            XGBRegressor(learning_rate =0.01, n_estimators=3460,
                                         max_depth=3,min_child_weight=0 ,
                                         gamma=0, subsample=0.7,
                                         colsample_bytree=0.7,
                                         objective= 'reg:linear',nthread=4,
                                         scale_pos_weight=1,seed=27,
                                         reg_alpha=0.00006))


    #stack
    stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet,
                                                xgboost,
                                                lightgbm
                                                ),
                                   meta_regressor=xgboost,
                                   use_features_in_secondary=True)
    return stack_gen

