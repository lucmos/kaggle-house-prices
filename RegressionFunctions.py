import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import Lasso, LassoCV, RidgeCV, ElasticNetCV, Ridge, ElasticNet, LassoLarsCV, BayesianRidge, \
    ARDRegression, orthogonal_mp, OrthogonalMatchingPursuit
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

from mlxtend.regressor import StackingCVRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

RANDOM_STATE = 42


# %% Global variables


def print_corss_score(predictor_fitted):
    alphas = np.logspace(-3, -1, 30)
    # todo completa!


# def fit_predict(x_train_complete, y_train_complete, x_test):
#     nF = 20
#
#     kf = KFold(n_splits=nF, random_state=241, shuffle=True)
#
#     test_errors_l2 = []
#     train_errors_l2 = []
#     test_errors_l1 = []
#     train_errors_l1 = []
#     test_errors_GBR = []
#     train_errors_GBR = []
#     test_errors_ENet = []
#     test_errors_LGB = []
#     test_errors_stack = []
#     test_errors_ens = []
#     train_errors_ens = []
#
#     models = []
#
#     pred_all = []
#
#     ifold = 1
#
#     x_train_complete = np.asarray(x_train_complete)
#     y_train_complete = np.asarray(y_train_complete)
#     x_test = np.asarray(x_test)
#
#     for train_index, test_index in kf.split(x_train_complete):
#         print('fold: ',ifold)
#         ifold = ifold + 1
#         X_train, X_test = x_train_complete[train_index], x_train_complete[test_index]
#         y_train, y_test = y_train_complete[train_index], y_train_complete[test_index]
#
#         # ridge
#         l2Regr = Ridge(alpha=9.0, fit_intercept = True)
#         l2Regr.fit(X_train, y_train)
#         pred_train_l2 = l2Regr.predict(X_train)
#         pred_test_l2 = l2Regr.predict(X_test)
#
#         # lasso
#         l1Regr = make_pipeline(RobustScaler(), Lasso(alpha = 0.0003, random_state=1, max_iter=50000))
#         l1Regr.fit(X_train, y_train)
#         pred_train_l1 = l1Regr.predict(X_train)
#         pred_test_l1 = l1Regr.predict(X_test)
#
#         # GBR
#         myGBR = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.02,
#                                           max_depth=4, max_features='sqrt',
#                                           min_samples_leaf=15, min_samples_split=50,
#                                           loss='huber', random_state = 5)
#
#         myGBR.fit(X_train,y_train)
#         pred_train_GBR = myGBR.predict(X_train)
#
#         pred_test_GBR = myGBR.predict(X_test)
#
#         # ENet
#         ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=4.0, l1_ratio=0.005, random_state=3))
#         ENet.fit(X_train, y_train)
#         pred_train_ENet = ENet.predict(X_train)
#         pred_test_ENet = ENet.predict(X_test)
#
#         # LGB
#         myLGB = LGBMRegressor(objective='regression',num_leaves=5,
#                                   learning_rate=0.05, n_estimators=600,
#                                   max_bin = 50, bagging_fraction = 0.6,
#                                   bagging_freq = 5, feature_fraction = 0.25,
#                                   feature_fraction_seed=9, bagging_seed=9,
#                                   min_data_in_leaf = 6, min_sum_hessian_in_leaf = 11)
#         myLGB.fit(X_train, y_train)
#         pred_train_LGB = myLGB.predict(X_train)
#         pred_test_LGB = myLGB.predict(X_test)
#
#         # Stacking
#         stackedset = pd.DataFrame({'A' : []})
#         stackedset = pd.concat([stackedset,pd.DataFrame(pred_test_l2)],axis=1)
#         stackedset = pd.concat([stackedset,pd.DataFrame(pred_test_l1)],axis=1)
#         stackedset = pd.concat([stackedset,pd.DataFrame(pred_test_GBR)],axis=1)
#         stackedset = pd.concat([stackedset,pd.DataFrame(pred_test_ENet)],axis=1)
#         stackedset = pd.concat([stackedset,pd.DataFrame(pred_test_LGB)],axis=1)
#         prod = (pred_test_l2*pred_test_l1*pred_test_GBR*pred_test_ENet*pred_test_LGB) ** (1.0/5.0)
#         stackedset = pd.concat([stackedset,pd.DataFrame(prod)],axis=1)
#         Xstack = np.array(stackedset)
#         Xstack = np.delete(Xstack, 0, axis=1)
#         l1_staked = Lasso(alpha = 0.0001,fit_intercept = True)
#         l1_staked.fit(Xstack, y_test)
#         pred_test_stack = l1_staked.predict(Xstack)
#
#         models.append([l2Regr,l1Regr,myGBR,ENet,myLGB,l1_staked])
#
#         test_errors_l2.append(np.square(pred_test_l2 - y_test).mean() ** 0.5)
#         test_errors_l1.append(np.square(pred_test_l1 - y_test).mean() ** 0.5)
#         test_errors_GBR.append(np.square(pred_test_GBR - y_test).mean() ** 0.5)
#         test_errors_ENet.append(np.square(pred_test_ENet - y_test).mean() ** 0.5)
#         test_errors_LGB.append(np.square(pred_test_LGB - y_test).mean() ** 0.5)
#         test_errors_stack.append(np.square(pred_test_stack - y_test).mean() ** 0.5)
#
#     M = x_test.shape[0]
#     scores_fin = 1+np.zeros(M)
#
#     for md in models:
#         l2 = md[0]
#         l1 = md[1]
#         GBR = md[2]
#         ENet = md[3]
#         LGB = md[4]
#         l1_stacked = md[5]
#
#         l2_scores = l2.predict(x_test)
#         l1_scores = l1.predict(x_test)
#         GBR_scores = GBR.predict(x_test)
#         ENet_scores = ENet.predict(x_test)
#         LGB_scores = LGB.predict(x_test)
#
#         stackedsets = pd.DataFrame({'A' : []})
#         stackedsets = pd.concat([stackedsets,pd.DataFrame(l2_scores)],axis=1)
#         stackedsets = pd.concat([stackedsets,pd.DataFrame(l1_scores)],axis=1)
#         stackedsets = pd.concat([stackedsets,pd.DataFrame(GBR_scores)],axis=1)
#         stackedsets = pd.concat([stackedsets,pd.DataFrame(ENet_scores)],axis=1)
#         stackedsets = pd.concat([stackedsets,pd.DataFrame(LGB_scores)],axis=1)
#         prod = (l2_scores*l1_scores*GBR_scores*ENet_scores*LGB_scores) ** (1.0/5.0)
#         stackedsets = pd.concat([stackedsets,pd.DataFrame(prod)],axis=1)
#         Xstacks = np.array(stackedsets)
#         Xstacks = np.delete(Xstacks, 0, axis=1)
#         scores_fin = scores_fin * l1_stacked.predict(Xstacks)
#     scores_fin = scores_fin ** (1/nF)
#     return scores_fin


def geo_mean_overflow(iterable):
    a = np.log(iterable)
    return np.exp(a.sum() / len(a))


from scipy.stats.mstats import gmean


def fit_predict(x_train, y_train, x_test):
    kfolds = KFold(n_splits=20, shuffle=True, random_state=RANDOM_STATE)

    ridge_alphas = list(np.linspace(4, 15, 50)) + [14.49, 14.61, 14.69, 14.81, 14.89, 15.01, 15.09, 15.21, 15.29, 15.41,
                                                   15.490]

    lasso_alpha = list(np.linspace(0.0001, 3, 100)) + [0.000051, 0.00009, 0.00021, 0.00029, 0.00041, 0.00051, 0.00059,
                                                   0.00071, 0.00078]

    e_l1ratio = list(np.linspace(0.1, 1, 20)) + [0.8, 0.85, 0.9, 0.95, 0.99, 1]
    e_alphas = list(np.linspace(0.00095, 1, 20)) + [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

    predictors = [
        (
            make_pipeline(
                RobustScaler(),
                RidgeCV(alphas=ridge_alphas, cv=kfolds, fit_intercept=True)),
            1
        ),
        (
            make_pipeline(
                RobustScaler(),
                LassoCV(max_iter=1e8, alphas=lasso_alpha, selection='random', verbose=True, random_state=RANDOM_STATE,
                        cv=kfolds, n_jobs=12, fit_intercept=True)),
            1
        ),
        # (
        #     make_pipeline(
        #         RobustScaler(),
        #         LassoLarsCV(max_iter=1e7, verbose=True, cv=kfolds, n_jobs=12, fit_intercept=True)),
        #     1
        # ),
        (
            make_pipeline(
                RobustScaler(),
                ElasticNetCV(max_iter=1e7, alphas=e_alphas, verbose=True, random_state=RANDOM_STATE, cv=kfolds, l1_ratio=e_l1ratio, n_jobs=12)),
            1
        ),
        (
            make_pipeline(
                RobustScaler(),
                GradientBoostingRegressor(n_estimators=3000, verbose=True, learning_rate=0.02,
                                          max_depth=4, max_features='sqrt',
                                          min_samples_leaf=15, min_samples_split=50,
                                          loss='huber', random_state=5)),
            1
        ),
        # (
        #     make_pipeline(
        #         RobustScaler(),
        #         LGBMRegressor(objective='regression', num_leaves=5,
        #                       learning_rate=0.05, n_estimators=600,
        #                       max_bin=50, bagging_fraction=0.6,
        #                       bagging_freq=5, feature_fraction=0.25,
        #                       feature_fraction_seed=9, bagging_seed=9,
        #                       min_data_in_leaf=6, min_sum_hessian_in_leaf=11)),
        #     1
        # )
          (
            make_pipeline(
                RobustScaler(),
                BayesianRidge(fit_intercept=True, verbose=True, n_iter=10000)),

            1
        ),
    ]


    for predictor, _ in predictors:
        predictor.fit(x_train, y_train)

    predictions = [predictor.predict(x_test) for predictor, _ in predictors]
    weights = [w for _, w in predictors]

    # predictions = np.average(predictions, weights=weights, axis=0)
    predictions = gmean(predictions, axis=0)
    # todo use gmean()

    predictions = np.expm1(predictions)
    predictions = normalize_predictions(predictions)

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
                                  random_state=RANDOM_STATE, cv=kfolds))

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
                            XGBRegressor(learning_rate=0.01, n_estimators=1000,
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
