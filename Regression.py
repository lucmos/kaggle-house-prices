from pathlib import Path

from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

from FeaturesEngineering import get_engineered_train_test
from RegressionFunctions import *
from constants import *

# %% Prepare data
((train_ids, x_train, y_train), (test_ids, x_test)) = get_engineered_train_test()


# -------------------------------------- DEV --------------------------------------
NUMBER_OF_RANDOM_SPLITS = 50
TEST_SIZE = 0.50
PERFORM_VALIDATION = True
PERFORM_PREDICTIONS = True


def get_error_random_dev(title=""):

    if title:
        print(title)

    # %% Split into train and dev
    x_dev, x_val, y_dev, y_val = train_test_split(
        x_train, y_train, test_size=TEST_SIZE)

    predictions_dev = fit_predict(x_dev, y_dev, x_val)

    # %% Compute error on DEV
    y_val = np.expm1(y_val)
    err = np.sqrt(mean_squared_log_error(y_val, predictions_dev))

    print("ERROR on validation set: {}".format(err))
    return err


if PERFORM_VALIDATION:
    print("Performing validation")
    dev_errors = [get_error_random_dev("{}/{}".format(i+1, NUMBER_OF_RANDOM_SPLITS)) for i in range(NUMBER_OF_RANDOM_SPLITS)]
    print("\n\nDEV ERROR ~ Stats over {} random splits with {} test\n"
          "> mean: {}\n"
          "> variance: {}\n"
          "> stdev: {}\n\n".format(NUMBER_OF_RANDOM_SPLITS,
                                 TEST_SIZE,
                                 np.mean(dev_errors),
                                 np.var(dev_errors),
                                 np.std(dev_errors)))
    print("Done validating")


# -------------------------------------- TEST --------------------------------------
if PERFORM_PREDICTIONS:
    print("Performing predictions")
    predictions_test = fit_predict(x_train, y_train, x_test)

    predictions_df = pd.DataFrame()
    predictions_df.insert(0, 'Id', test_ids)
    predictions_df['SalePrice'] = predictions_test
    predictions_df.to_csv(Path(predictions_dir, 'predictions_test.csv'), index=False)
    print("DONE")
