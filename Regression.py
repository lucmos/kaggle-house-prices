from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

from FeaturesEngineering import get_engineered_train_test
from RegressionFunctions import  *


from constants import *

# %% Prepare data
((train_ids, x_train, y_train), (test_ids, x_test)) = get_engineered_train_test()


# -------------------------------------- DEV --------------------------------------

# %% Split into train and dev
x_train_red, x_dev, y_train_red, y_dev = train_test_split(
    x_train, y_train, test_size=0.40)


# %% Build models
stack_gen_model_dev = get_stack_gen_model()


# %% Fit models
# prepare dataframes without numpy
stackX_dev = np.array(x_train_red)
stacky_dev = np.array(y_train_red)
stack_gen_model_dev = stack_gen_model_dev.fit(stackX_dev, stacky_dev)


# %% Perform predictions on dev
# em_preds_dev = elastic_model3.predict(x_dev)
# lasso_preds_dev = lasso_model2.predict(x_dev)
# ridge_preds_dev = ridge_model2.predict(x_dev)
stack_gen_preds_dev = stack_gen_model_dev.predict(x_dev)
# xgb_preds_dev = xgb_fit.predict(x_dev)
# svr_preds_dev = svr_fit.predict(x_dev)
# lgbm_preds_dev = lgbm_fit.predict(x_dev)
predictions_dev = stack_gen_preds_dev


# %% Normalize labels
y_dev = np.expm1(y_dev)
predictions_dev = np.expm1(predictions_dev)

# %% Compute error on DEV
err = np.sqrt(mean_squared_log_error(y_dev, predictions_dev))
print("ERROR on validation set: {}".format(err))

print("Done validating")


# -------------------------------------- TEST --------------------------------------

# %% Build models
stack_gen_model_test = get_stack_gen_model()


# %% Fit models
# prepare dataframes without numpy
stackX_test = np.array(x_train)
stacky_test = np.array(y_train)
stack_gen_model_test = stack_gen_model_test.fit(stackX_test, stacky_test)


# %% Perform predictions on test
# em_preds_test = elastic_model3.predict(x_test)
# lasso_preds_test = lasso_model2.predict(x_test)
# ridge_preds_test = ridge_model2.predict(x_test)
stack_gen_preds_test = stack_gen_model_test.predict(x_test)
# xgb_preds_test = xgb_fit.predict(x_test)
# svr_preds_test = svr_fit.predict(x_test)
# lgbm_preds_test = lgbm_fit.predict(x_test)
predictions_test = stack_gen_preds_test


# %% Normalize predictions_test && save to file
result_df_test = pd.DataFrame()
result_df_test['Id'] = test_ids
result_df_test['SalePrice'] = np.expm1(predictions_test)
result_df_test.to_csv(Path(predictions_dir, 'predictions_test.csv'), index=False)
print("DONE")
