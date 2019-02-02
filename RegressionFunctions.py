import numpy as np
import pandas as pd
from mlxtend.regressor import StackingCVRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, LassoCV, RidgeCV, ElasticNetCV, Ridge, ElasticNet, BayesianRidge, \
    LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

RANDOM_STATE = 42


# %% Global variables


def print_corss_score(predictor_fitted):
    alphas = np.logspace(-3, -1, 30)
    # todo completa!


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
        make_pipeline(
            RobustScaler(),
            RidgeCV(alphas=ridge_alphas, cv=None, fit_intercept=True, store_cv_values=True)),
        make_pipeline(
            RobustScaler(),
            LassoCV(max_iter=1e8, alphas=lasso_alpha, selection='random', verbose=True, random_state=RANDOM_STATE,
                    cv=kfolds, n_jobs=12, fit_intercept=True)),
        make_pipeline(
            RobustScaler(),
            ElasticNetCV(max_iter=1e7, alphas=e_alphas, verbose=True, random_state=RANDOM_STATE, cv=kfolds,
                         l1_ratio=e_l1ratio, n_jobs=12)),
        make_pipeline(
            RobustScaler(),
            GradientBoostingRegressor(n_estimators=3000, verbose=True, learning_rate=0.02,
                                      max_depth=4, max_features='sqrt',
                                      min_samples_leaf=15, min_samples_split=50,
                                      loss='huber', random_state=5)),
        make_pipeline(
            RobustScaler(),
            BayesianRidge(fit_intercept=True, verbose=True, n_iter=10000)),
    ]

    for predictor in predictors:
        predictor.fit(x_train, y_train)

    predictions = [predictor.predict(x_test) for predictor in predictors]

    x_train_sta = np.asarray(x_train)
    y_train_sta = np.asarray(y_train)
    x_test_sta = np.asarray(x_test)
    stacked, ridge_meta, ridge, lasso, elasti, grad, baye = get_stack_gen_model()
    stacked.fit(x_train_sta, y_train_sta)
    pred_sta = stacked.predict(x_test_sta)

    # print("Meta ridge alpha: {}".format(ridge_meta.coef_))
    #
    # print("ridge alpha: {}".format(ridge.coef_))
    # print("lasso alpha: {}".format(lasso.coef_))
    # print("elasti lapha: {} \t elastic l1_ratio: {}".format(elasti.alpha_, elasti.l1_ratio_))

    predictions.append(pred_sta)

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
    kfolds = KFold(n_splits=20, shuffle=True, random_state=RANDOM_STATE)

    ridge_alphas = list(np.linspace(0.1, 15, 100)) + [14.49, 14.61, 14.69, 14.81, 14.89, 15.01, 15.09, 15.21, 15.29,
                                                      15.41,
                                                      15.490]

    lasso_alpha = list(np.linspace(0.0001, 3, 100)) + [0.000051, 0.00009, 0.00021, 0.00029, 0.00041, 0.00051, 0.00059,
                                                       0.00071, 0.00078]

    e_l1ratio = list(np.linspace(0.1, 1, 20)) + [0.8, 0.85, 0.9, 0.95, 0.99, 1]
    e_alphas = list(np.linspace(0.00095, 1, 20)) + [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

    # best ma 40 minuti
    # ridge_meta = RidgeCV(alphas=ridge_alphas, cv=kfolds, fit_intercept=True)
    # meta_regr = make_pipeline(
    #     RobustScaler(),
    #     ridge_meta)
    #
    # ridge = RidgeCV(alphas=ridge_alphas, cv=kfolds, fit_intercept=True)
    # lasso = LassoCV(max_iter=1e8, alphas=lasso_alpha, selection='random', verbose=True, random_state=RANDOM_STATE,cv=kfolds, n_jobs=12, fit_intercept=True)
    # elasti = ElasticNetCV(max_iter=1e7, alphas=e_alphas, verbose=True, random_state=RANDOM_STATE, cv=kfolds, l1_ratio=e_l1ratio, n_jobs=12)
    # grad = GradientBoostingRegressor(n_estimators=3000, verbose=True, learning_rate=0.02, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=50, loss='huber', random_state=5)
    # baye = BayesianRidge(fit_intercept=True, verbose=True, n_iter=10000)

    # TODO  QUELLO CHE HA FATTO SCENDERE SOTTO LA SOGLIA DI 113 È QUESTO ALPHA! O.O
    meta = Lasso(alpha=0.0007, random_state=RANDOM_STATE, max_iter=50000)
    meta_regr = make_pipeline(
        RobustScaler(),
        meta)

    ridge = Ridge(alpha=15.0, fit_intercept=True)
    lasso = Lasso(alpha=0.0003, random_state=RANDOM_STATE, max_iter=50000)
    elasti = ElasticNet(alpha=4.0, l1_ratio=0.005, random_state=3)
    grad = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.02,
                                     max_depth=4, max_features='sqrt',
                                     min_samples_leaf=15, min_samples_split=50,
                                     loss='huber', random_state=5)
    baye = BayesianRidge(fit_intercept=True, verbose=True, n_iter=10000)

    predictors = [
        make_pipeline(
            RobustScaler(),
            ridge),
        make_pipeline(
            RobustScaler(),
            lasso),
        make_pipeline(
            RobustScaler(),
            elasti),
        make_pipeline(
            RobustScaler(),
            grad),
        make_pipeline(
            RobustScaler(),
            baye),
    ]

    # stack
    stack_gen = StackingCVRegressor(regressors=predictors,
                                    meta_regressor=meta_regr,
                                    use_features_in_secondary=True,
                                    )

    print(meta_regr)
    return stack_gen, meta, ridge, lasso, elasti, grad, baye


def transform(x):
    round_value = 1000
    int_price = int(x)
    remainder = int_price % round_value
    if remainder >= round_value / 2:
        int_price += (round_value - remainder)
    else:
        int_price -= remainder

    return int_price


def normalize_predictions(predictions):
    predictions_df = pd.DataFrame()
    predictions_df['norm'] = predictions
    q1 = predictions_df['norm'].quantile(0.0042)
    q2 = predictions_df['norm'].quantile(0.99)
    predictions_df['norm'] = predictions_df['norm'].apply(lambda x: x if x > q1 else x * 0.77)
    predictions_df['norm'] = predictions_df['norm'].apply(lambda x: x if x < q2 else x * 1.1)

    return [transform(x) for x in predictions_df['norm'].values]
