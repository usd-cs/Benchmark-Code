"""
Purpose:  Deciding a best predictor model for a singular buoy

Authors: Marissa Esteban, Gabe Krishnadasan, Diana Montoya-Herrera, Gabe Seidl, Madeleine Woo
Date: 2/20/2024
"""

# import libraries needed
from seebuoy import NDBC
from IPython.display import display
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, skew, probplot
from scipy.special import boxcox1p
import warnings
warnings.filterwarnings('ignore')
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import sys
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

#from kriging import kriging

def get_buoy_data(buoy_num):
    ndbc = NDBC()
    
    # Information on NDBC's ~1800 buoys and gliders
    wave_df = ndbc.stations()

    # list all available data for all buoys
    df_data = ndbc.available_data()

    return ndbc.get_data(buoy_num)

def handle_missing_data(buoy):
    """
    The data has some missing values. We impute these values by mode, mean, and interpolation.
    """

    buoy = buoy.reset_index()

    # missing data
    total = buoy.isnull().sum().sort_values(ascending=False)
    percent = (buoy.isnull().sum() / buoy.isnull().count()).sort_values(
        ascending=False
    )
    missing_data = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
    print(missing_data)

    # dropping rows where average_period is null
    buoy.dropna(subset=['average_period'], inplace=True)

    # dropping rows wehre wave_height is null
    buoy.dropna(subset=['wave_height'], inplace=True)

    # dropping cols where there is 100% NA
    buoy.dropna(axis=1, how='all', inplace=True)

    # IMPUTATIONS

    columns_to_fill = [
        "wind_speed",
        "wind_gust",
        "dominant_period",
        "mean_wave_direction",
        "pressure",
        "pressure_tendency",
        "water_temp"
    ]

    # Replace missing data with mode
    buoy_mode = buoy.copy()
    for column in columns_to_fill:
        buoy_mode[column] = buoy_mode[column].fillna(buoy_mode[column].mode()[0])

    # Replace missing data with mean
    buoy_mean = buoy.copy()
    for column in columns_to_fill:
        buoy_mean[column] = buoy_mean[column].fillna(buoy_mean[column].mean())

    # Interpolate missing values using spline interpolation
    buoy_interpolated = buoy.copy()
    for column in columns_to_fill:
        buoy_interpolated[column] = buoy_interpolated[column].fillna(buoy_interpolated[column].interpolate(method='spline', order=2))
    
    # check if there are any additional missing values
    for column in columns_to_fill:
        if column in buoy_interpolated.columns and buoy_interpolated[column].isnull().any():
            buoy_interpolated = buoy_interpolated.drop(column, axis=1)

     

    # Remove non finite values
    # buoy_mode = buoy_mode[np.isfinite(buoy_mode['wave_height'])]
    # buoy_mean = buoy_mean[np.isfinite(buoy_mean['wave_height'])]
    # buoy_interpolated = buoy_interpolated[np.isfinite(buoy_interpolated['wave_height'])]
    #buoy_kriging = buoy_kriging[np.isfinite(buoy_kriging['wave_height'])]


    #return (buoy_mean, buoy_mode, buoy_interpolated, buoy_kriging)
    return (buoy_mean, buoy_mode, buoy_interpolated)


def time_series_split_regression(
    data,
    regressor,
    date_column="date",
    target_column="wave_height",
    cols_to_ignore=[],
    n_splits=5,
):
    """
    Perform time series split on a pandas DataFrame based on a date column and
    train a regression model, calculating RMSE for each split.

    Parameters:
    - data: pandas DataFrame
    - regressor: scikit-learn regressor object
        The regression algorithm to use.
    - date_column: str, default="date"
        The name of the date column in the DataFrame.
    - target_column: str, default="wave_height"
        The name of the target column in the DataFrame.
    - n_splits: int, default=5
        Number of splits for TimeSeriesSplit.
    - tune_hyperparameters: bool, default=False

    Returns:
    - result_df: pandas DataFrame
        DataFrame containing the Id, actual value, predicted value, fold, and whether it was in the test or train set.
    - rmse_scores: list of floats
        List of RMSE scores for each split.
    - split_dates: list of tuples
        List of (min_date, max_date) tuples for each split.
    - num_records: list of tuples
        List of (train_size, test_size) tuples for each split.
    """

    # Sort the DataFrame based on the date column

    print(data.head())
    data = data.sort_values(by=date_column)

    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)

    rmse_scores = []
    split_dates = []
    num_records = []
    all_predictions = []

    # Perform the time series split and train regression model for each split
    for fold, (train_index, test_index) in enumerate(tscv.split(data)):
        train_data, test_data = data.iloc[train_index], data.iloc[test_index]

        cols_to_ignore = cols_to_ignore + [target_column, date_column]

        X_train = train_data.drop(cols_to_ignore, axis=1)
        X_test = test_data.drop(cols_to_ignore, axis=1)
        y_train, y_test = train_data[target_column], test_data[target_column]

        # Record the minimum and maximum dates for each split
        min_date, max_date = test_data[date_column].min(), test_data[date_column].max()
        split_dates.append((min_date, max_date))

        # Train regression model
        regressor.fit(
            X_train, np.log1p(y_train)
        )  # Apply log1p transformation to the target variable during training

        # Make predictions
        y_pred_log = regressor.predict(X_test)
        y_pred_train_log = regressor.predict(X_train)

        # Inverse transform predictions to get back the original scale
        # TODO: does this apply to our data too? Still kind of confused on what this does.
        y_pred = np.expm1(y_pred_log)
        y_pred_train = np.expm1(y_pred_train_log)

        # Check for NaN or infinity values in y_pred or y_test
        if (
            np.isnan(y_pred).any()
            or np.isinf(y_pred).any()
            or np.isnan(y_test).any()
            or np.isinf(y_test).any()
        ):
            print(
                f"Warning: NaN or infinity values found in predictions or true values. Imputing 0 for problematic values in y_pred for fold {fold}."
            )
            y_pred[np.isnan(y_pred) | np.isinf(y_pred)] = 0
            # Optionally, you can also handle y_test in a similar way if needed
            # y_test[np.isnan(y_test) | np.isinf(y_test)] = 0

        # Calculate RMSE on the original scale
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_scores.append((rmse, fold))

        # Record results for 'date', 'Actual', 'Predicted', 'Fold', and 'Set' in a list
        # TODO: what it doooooo?????
        fold_predictions = list(
            zip(
                test_data["date"],
                y_test,
                y_pred,
                [fold] * len(test_data),
                ["test"] * len(test_data),
            )
        )
        fold_predictions += list(
            zip(
                train_data["date"],
                y_train,
                y_pred_train,
                [fold] * len(train_data),
                ["train"] * len(train_data),
            )
        )
        all_predictions.extend(fold_predictions)

        # Calculate the size of each train-test split
        num_records.append((len(train_data), len(test_data)))

    # Create a DataFrame from the results
    result_df = pd.DataFrame(
        all_predictions, columns=["date", "Actual", "Predicted", "Fold", "Set"]
    )

    return result_df, rmse_scores, split_dates, num_records


def compute_rmse_std(tuple_list):
    """
    Computes the standard deviation of the root mean squared errors

    @param tuple_list: List of RMSE scores for each split.
    @return tuple of mean and standard deviation
    """
    first_elements = [t[0] for t in tuple_list]
    mean = np.mean(first_elements)
    std = np.std(first_elements)
    return mean, std

def print_rmse_and_dates(model_rmse, model_split_dates, num_records, model_name):
    """
    Prints the RMSE for each of the train test splits

    @param model_rmse: List of RMSE scores for each split.
    @param model_split_dates: List of (min_date, max_date) tuples for each split.
    @param num_records: List of (train_size, test_size) tuples for each split.
    @model_name: a string indicating the name of the model
    """
    # Print RMSE scores and split dates for each split
    for i, (rmse, dates, records) in enumerate(
        zip(model_rmse, model_split_dates, num_records)
    ):
        min_date, max_date = dates
        num_train_records, num_test_records = records

        min_date = min_date.date()
        max_date = max_date.date()

        print(
            f"Split {i + 1}: Min Date: {min_date}, Max Date: {max_date}, RMSE: {rmse[0]}, Train Records: {num_train_records}, Test Records: {num_test_records}"
        )

    rmse_std = compute_rmse_std(model_rmse)
    print(model_name, "RMSE score: {:.4f} ({:.4f})\n".format(rmse_std[0], rmse_std[1]))


def train_model(buoy):
    """
    Train the model for linear regression and random forest
    @param ny_buoy: the data we're training on
    @return
    """
    ##################### LINEAR REGRESSION MODELS ###############
    # 3 types here: with intercept, without intercept, and Elastic Net (both L1 and L2 regularization)
    lr_w_int = LinearRegression()
    lr_no_int = LinearRegression(fit_intercept=False)
    elastic_net = ElasticNet(
        alpha=0.01, l1_ratio=0.1
    )  # Adjust alpha and l1_ratio as needed
    (
        lr_w_int_preds_df,
        lr_w_int_rmse,
        lr_w_int_split_dates,
        num_records,
    ) = time_series_split_regression(
        buoy,
        regressor=lr_w_int,
    )
    # Print RMSE scores and split dates for each split
    print_rmse_and_dates(
        lr_w_int_rmse, lr_w_int_split_dates, num_records, "Linear Regression (w/ Intercept)"
    )

    ###################### RANDOM FOREST MODEL ##################
    rf = RandomForestRegressor(n_estimators=100)

    rf_preds_df, rf_rmse, rf_split_dates, num_records = time_series_split_regression(
        buoy,
        regressor=rf,
    )
    print_rmse_and_dates(rf_rmse, rf_split_dates, num_records, "Random Forest")


def handle_analytics(buoy, model):
    """
    """
    train_df = buoy[buoy.columns.difference(['wave_height', 'average_period'])]
    
    # Validation function for wave height
    n_folds = 5

    kf=KFold(n_splits=n_folds)
    rmse = np.sqrt(-cross_val_score(model, train_df, buoy.wave_height, scoring="neg_mean_squared_error", cv = kf))
    return rmse

def predict(buoy, model, predictedAttribute, testSize):
    """
    Used to make predictions with a selected buoy df, model, and attribute

    @param buoy: Pandas Dataframe of buoy data
    @param model: The model selected to make predictions
    @param predictedAttribute: String value of "wave_height" or "dominant_period"
    @param testSize: How large the test size should be (0 < test_size < 1)
    """

    #This should be changed, but I have it in because I cant figure out how to fit a model using float vals, maybe
    buoy = buoy * 10

    #Sets up the train and test data based on target variable
    if predictedAttribute == "wave_heigth":
        cols = ['wind_speed', 'wind_gust', 'dominant_period',
                'average_period', 'mean_wave_direction', 'pressure', 'water_temp']
        X_train, X_test, y_train, y_test = train_test_split(buoy[cols], buoy['wave_height'], test_size=testSize, random_state=42)
    elif predictedAttribute == "dominant_period":
        cols = ['wind_speed', 'wind_gust', 'wave_height',
                'average_period', 'mean_wave_direction', 'pressure', 'water_temp']
        X_train, X_test, y_train, y_test = train_test_split(buoy[cols], buoy['dominant_period'], test_size=testSize, random_state=42)
    else:
        print("UH OH")

    #Fit the model to the train data
    model.fit(X_train, y_train.astype('float'))

    #Make predictions on the test data
    predictions = model.predict(X_test)
    actual = list(y_test)

    #Reverts the predictions to original scale, need to change with line 93
    for i in range(len(predictions)):
        predictions[i] = predictions[i] / 10
        actual[i] = actual[i] / 10

    sumOfDifferences = 0

    for i in range(len(predictions)):
        sumOfDifferences += abs(actual[i] - predictions[i])


    print("Model used: ", model)
    print("Predicted attribute: ", predictedAttribute)
    print("Avg off: ", sumOfDifferences / len(predictions))

buoy = get_buoy_data("44065")

# handle missing data
buoy_mean, buoy_mode, buoy_interpolated = handle_missing_data(buoy)

# train the model

print("Test with mean imputation\n")
train_model(buoy_mean)

print("Test with mode imputation\n")
train_model(buoy_mode)

print("Test with interpolation imputation\n")
train_model(buoy_interpolated)

# lr_w_int = LinearRegression()
# lr_no_int = LinearRegression(fit_intercept=False)
# rf = RandomForestRegressor(n_estimators=100)

# modelList = [lr_w_int, lr_no_int, rf]
# predictAttributesList = ["wave_heigth", "dominant_period"]

# for i in range(len(buoyTup)):
#     for model in modelList:
#         for attribute in predictAttributesList:
#             if i == 0:
#                 print("Imputation method: Mean")
#             elif i == 1:
#                 print("Imputation method: Mode")
#             elif i == 2:
#                 print("Imputation method: Interpolation")
#             else:
#                 print("Imputation method: Kriging")
#             predict(buoyTup[i], model, attribute, 0.2)

# i = 0
# for buoy in buoyTup:
#     if i == 0:
#         print("lr_w_int")
#     elif i == 1:
#         print("lr_no_int")
#     else:
#         print("rf")
#     i += 1
#     for model in modelList:
#         print(handle_analytics(buoy, model))