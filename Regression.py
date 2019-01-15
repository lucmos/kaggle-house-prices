from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from sklearn.linear_model import Lasso, LassoCV, RidgeCV, ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

from FeaturesEngineering import get_engineered_train_test

predictions_dir = "./predictions/"

((train_ids, x_train, y_train), (test_ids, x_test)) = get_engineered_train_test()

# %% Test predictions
# predictor = make_pipeline(XGBRegressor(learning_rate=0.01, n_estimators=3460,
#                                        max_depth=3, min_child_weight=0,
#                                        gamma=0, subsample=0.7,
#                                        colsample_bytree=0.7,
#                                        objective='reg:linear', nthread=4,
#                                        scale_pos_weight=1, seed=27,
#                                        reg_alpha=0.00006))
# #
# predictor = make_pipeline(RobustScaler(),
#                           Lasso(alpha=0.0003, random_state=1, max_iter=50000))
#

# alphas = [0.00005, 0.0001, 0.0003, 0.0005, 0.0007,
#           0.0009, 0.01]
# alphas2 = [0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005,
#            0.0006, 0.0007, 0.0008]
#
#
# lasso_model2 = make_pipeline(RobustScaler(),
#                              LassoCV(max_iter=1e7,
#                                     alphas = alphas2,
#                                     random_state = 42)).fit(x_train, y_train)
# predictor = lasso_model2


from mlxtend.regressor import StackingCVRegressor
from sklearn.pipeline import make_pipeline

kfolds = KFold(n_splits=10, shuffle=True, random_state=23)


alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
#setup models
ridge = make_pipeline(RobustScaler(),
                      RidgeCV(alphas = alphas_alt, cv=kfolds))

alphas2 = [0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005,
           0.0006, 0.0007, 0.0008]
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

#prepare dataframes
stackX = np.array(x_train)
stacky = np.array(y_train)
stack_gen_model = stack_gen.fit(stackX, stacky)

# em_preds = elastic_model3.predict(x_test)
# lasso_preds = lasso_model2.predict(x_test)
# ridge_preds = ridge_model2.predict(x_test)
stack_gen_preds = stack_gen_model.predict(x_test)
# xgb_preds = xgb_fit.predict(x_test)
# svr_preds = svr_fit.predict(x_test)
# lgbm_preds = lgbm_fit.predict(x_test)

predictions = stack_gen_preds
# predictor.fit(x_train, y_train)
#
# predictions = predictor.predict(x_test)
result_df = pd.DataFrame()
result_df['Id'] = test_ids
result_df['SalePrice'] = np.expm1(predictions)
result_df.to_csv(Path(predictions_dir, 'predictions_v4.csv'), index=False)
#
print("DONE")
