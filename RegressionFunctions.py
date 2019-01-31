import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso, LassoCV, RidgeCV, ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

from mlxtend.regressor import StackingCVRegressor
# from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# %% Global variables
kfolds = KFold(n_splits=10, shuffle=True, random_state=23)
alphas_alt = [14.49, 14.61, 14.69, 14.81, 14.89, 15.01, 15.09, 15.21, 15.29, 15.41, 15.490]
alphas2 = [0.000051, 0.00009, 0.00021, 0.00029, 0.00041, 0.00051, 0.00059, 0.00071, 0.00078]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]


def print_corss_score(predictor_fitted):
    alphas = np.logspace(-3, -1, 30)
    # todo completa!


def fit_predict(x_train, y_train, x_test):
    predicor1 = make_pipeline(RobustScaler(),
                              RidgeCV(alphas=alphas_alt, cv=kfolds))
    predicor2 = make_pipeline(RobustScaler(),
                              LassoCV(max_iter=1e7, alphas=alphas2,
                                      random_state=42, cv=kfolds, n_jobs=3))
    predicor3 = make_pipeline(RobustScaler(),
                              ElasticNetCV(max_iter=1e7, alphas=e_alphas,
                                           cv=kfolds, l1_ratio=e_l1ratio,
                                           n_jobs=3))

    predicor1.fit(x_train, y_train)
    predicor2.fit(x_train, y_train)
    predicor3.fit(x_train, y_train)

    predictions1 = predicor1.predict(x_test)
    predictions2 = predicor2.predict(x_test)
    predictions3 = predicor3.predict(x_test)

    predictions1 = np.expm1(predictions1)
    predictions1 = normalize_predictions(predictions1)

    predictions2 = np.expm1(predictions2)
    predictions2 = normalize_predictions(predictions2)

    predictions3 = np.expm1(predictions3)
    predictions3 = normalize_predictions(predictions3)
    predictions = np.mean([predictions1, predictions2, predictions3], axis=0)

    return predictions

    # # %% Build models
    # stack_gen_model = get_stack_gen_model()
    #
    # # %% Fit models
    # # prepare dataframes without numpy
    # stackX = np.array(x_train)
    # stacky = np.array(y_train)
    # stack_gen_model = stack_gen_model.fit(stackX, stacky)
    #
    # # %% Perform predictions on dev
    # # em_preds_dev = elastic_model3.predict(x_val)
    # # lasso_preds_dev = lasso_model2.predict(x_val)
    # # ridge_preds_dev = ridge_model2.predict(x_val)
    # stack_gen_preds_dev = stack_gen_model.predict(x_test)
    # # xgb_preds_dev = xgb_fit.predict(x_val)
    # # svr_preds_dev = svr_fit.predict(x_val)
    # # lgbm_preds_dev = lgbm_fit.predict(x_val)
    # predictions = stack_gen_preds_dev
    #
    # predictions = np.expm1(predictions)
    # predictions = normalize_predictions(predictions)
    #
    # return predictions


# %% Build stack gen model
def get_stack_gen_model():
    # setup models
    ridge = make_pipeline(RobustScaler(),
                          RidgeCV(alphas=alphas_alt, cv=kfolds))

    lasso = make_pipeline(RobustScaler(),
                          LassoCV(max_iter=1e7, alphas=alphas2,
                                  random_state=42, cv=kfolds))

    elasticnet = make_pipeline(RobustScaler(),
                               ElasticNetCV(max_iter=1e7, alphas=e_alphas,
                                            cv=kfolds, l1_ratio=e_l1ratio,
                                            n_jobs=3))

    lightgbm = make_pipeline(RobustScaler(),
                             LGBMRegressor(objective='regression', num_leaves=5,
                                           learning_rate=0.05, n_estimators=720,
                                           max_bin=55, bagging_fraction=0.8,
                                           bagging_freq=5, feature_fraction=0.2319,
                                           feature_fraction_seed=9, bagging_seed=9,
                                           min_data_in_leaf=6,
                                           min_sum_hessian_in_leaf=11,
                                           n_jobs=3))

    xgboost = make_pipeline(RobustScaler(),
                            XGBRegressor(learning_rate=0.01, n_estimators=3460,
                                         max_depth=3, min_child_weight=0,
                                         gamma=0, subsample=0.7,
                                         colsample_bytree=0.7,
                                         objective='reg:linear', nthread=4,
                                         scale_pos_weight=1, seed=27,
                                         reg_alpha=0.00006,
                                         n_jobs=3))

    # stack
    stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet,
                                                # xgboost,
                                                # lightgbm
                                                ),
                                    meta_regressor=xgboost,
                                    use_features_in_secondary=True)
    return stack_gen


def normalize_predictions(predictions):
    predictions_df = pd.DataFrame()
    predictions_df['norm'] = predictions
    q1 = predictions_df['norm'].quantile(0.0042)
    q2 = predictions_df['norm'].quantile(0.99)
    predictions_df['norm'] = predictions_df['norm'].apply(lambda x: x if x > q1 else x * 0.77)
    predictions_df['norm'] = predictions_df['norm'].apply(lambda x: x if x < q2 else x * 1.1)
    return predictions_df['norm'].values
