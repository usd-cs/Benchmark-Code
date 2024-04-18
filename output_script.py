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
from datetime import timedelta, datetime

def gather_data(buoy_num):
    """
    Imports data from the National Data Buoy Center
    @param buoy_num: the buoy number of the closest x,y coordinates
    """
    ndbc = NDBC()
    
    # Information on NDBC's ~1800 buoys and gliders
    wave_df = ndbc.stations()

    # list all available data for all buoys
    df_data = ndbc.available_data()

    # Get info on a single buoy
    see_buoy = ndbc.get_data(buoy_num)

    return see_buoy

def handle_missing_data(buoy):
    """
    The data has some missing values. We impute these values by mode, mean, and interpolation.

    @param buoy: the data imported from Seebuoy
    @param return: a tuple containing the buoy imputed with mean, mode, and interpolation 
    (buoy_mean, buoy_mode, buoy_interpolated)
    """
    # missing data
    total = buoy.isnull().sum().sort_values(ascending=False)
    percent = (buoy.isnull().sum() / buoy.isnull().count()).sort_values(
        ascending=False
    )
    missing_data = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])

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
        if column in buoy_mode:
            buoy_mode[column] = buoy_mode[column].fillna(buoy_mode[column].mode()[0])

    # Replace missing data with mean
    buoy_mean = buoy.copy()
    for column in columns_to_fill:
        if column in buoy_mean:
            buoy_mean[column] = buoy_mean[column].fillna(buoy_mean[column].mean())

    # Interpolate missing values using spline interpolation
    buoy_interpolated = buoy.copy()
    for column in columns_to_fill:
        if column in buoy_interpolated:
            buoy_interpolated[column] = buoy_interpolated[column].fillna(buoy_interpolated[column].interpolate(method='spline', order=2))
    
    # check if there are any additional missing values
    for column in columns_to_fill:
        if column in buoy_interpolated.columns and buoy_interpolated[column].isnull().any():
            buoy_interpolated = buoy_interpolated.drop(column, axis=1)

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
    data = data.sort_values(by=date_column)
    # data = data.set_index('date')
    data = data.reset_index(drop=True)

    print("DEBUG TIME SERIES SPLIT")
    display(data)

    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Time Series Split training
    rmse_scores = []
    split_dates = []
    num_records = []
    all_predictions = []

    regressor = train_time_series(tscv, data, target_column, date_column, split_dates, num_records, all_predictions, rmse_scores, regressor, cols_to_ignore)
    
    # Make prediction for the future
    # Extend time horizon for predictions (two weeks from now)
    start_date = datetime.now()
    end_date = start_date + timedelta(days=14)  # Extend two weeks
    prediction_dates = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Create a DataFrame from prediction dates
    prediction_dates_df = pd.DataFrame(prediction_dates, columns=["date"])

    # Add additional columns with empty values
    prediction_dates_df["wind_speed"] = pd.NA
    prediction_dates_df["wind_gust"] = pd.NA
    prediction_dates_df["average_period"] = pd.NA
    prediction_dates_df["mean_wave_direction"] = pd.NA
    prediction_dates_df["water_temp"] = pd.NA

    # make a new DataFrame with columns from old data
    extended_data = data.iloc[:, :]
    extended_data = pd.concat([extended_data, prediction_dates_df], axis=0)

    buoy_mean, buoy_mode, buoy_interpolated = handle_missing_data(extended_data)
    print("EXTENDED DATA AFTER HANDLING MISSING")
    display(buoy_interpolated)
    
    buoy_interpolated = buoy_interpolated.drop(["date", "wave_height"], axis=1)
    two_week_predictions = regressor.predict(buoy_interpolated)

    # Create a DataFrame from the results
    result_df = pd.DataFrame(
        all_predictions, columns=["date", "Actual", "Predicted", "Fold", "Set"]
    )

    return result_df, rmse_scores, split_dates, num_records, two_week_predictions

def train_time_series(tscv, data, target_column, date_column, split_dates, num_records, all_predictions, rmse_scores, regressor, cols_to_ignore):
    """
    Do the folds and training for each split

    Parameters:
    - tscv: Time series split object
    - data: pandas DataFrame
    - regressor: scikit-learn regressor object
        The regression algorithm to use.
    - date_column: str, default="date"
        The name of the date column in the DataFrame.
    - target_column: str, default="wave_height"
        The name of the target column in the DataFrame.
    - split_dates: list, empty list to be used in folds
    - num_records: list, empty list to be used to show the size of each train test split
    - all_predictions: list, empty list of predictions to be appended onto 
    - rmse_scores: list, empty list of rmse scores to be calculated and appended for each test
    - regressor: the regression algorithm to use
    - cols_to_ignore: list, columns to drop before training

    Returns:
    - regressor: the trained model

    """
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

        return regressor


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
    print("TRAIN MODEL DEBUG")
    display(buoy)
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
        two_week_predictions_linear,
    ) = time_series_split_regression(
        buoy,
        regressor=lr_w_int,
    )
    # Print RMSE scores and split dates for each split
    #print_rmse_and_dates(lr_w_int_rmse, lr_w_int_split_dates, num_records, "Linear Regression (w/ Intercept)")

    ###################### RANDOM FOREST MODEL ##################
    rf = RandomForestRegressor(n_estimators=100)

    rf_preds_df, rf_rmse, rf_split_dates, num_records, two_week_predictions_rf = time_series_split_regression(
        buoy,
        regressor=rf,
    )
    #print_rmse_and_dates(rf_rmse, rf_split_dates, num_records, "Random Forest")

    return lr_w_int_preds_df, rf_preds_df, two_week_predictions_linear, two_week_predictions_rf

def display_results(predictions):
    """
    Show a line plot with the trained data, real data, and predictions
    @param: predictions, a dataframe of predictions
    """
    plt.figure(figsize=(16,8))
    plt.plot(predictions["date"], predictions['Predicted'], color='green', label = 'Predicted Wave Height')
    plt.plot(predictions["date"], predictions['Actual'], color = 'red', label = 'Real Wave Height')
    plt.title('Wave Height Prediction (Linear Reg)')
    plt.xlabel('Time')
    plt.ylabel('Wave Height')
    plt.legend()
    plt.grid(True)
    plt.savefig('lin_reg.pdf')
    plt.show()

def filter_fold(data, fold_num):
    """
    Filter the dataframe by fold number (ex: reduce the dataframe to frame 0 only)
    @data: the dataframe containing the predictions
    @fold_num: an int indicating the fold number
    """

    filtered_df = data[data['Fold'] == fold_num]
    return filtered_df

def calculate_rmse():
    # Get the current date
    current_date = datetime.now()

    # Calculate the date of 14 days ago
    date_14_days_ago = current_date - timedelta(days=14)
    # past = forecast[forecast['date'] <= date_14_days_ago]

    # Get data for 45 days before 14 days ago

    # Train and predict for the past 14 days 

    # Compare results with actual previous 14 days 

# prompt user for a buoy number
buoy_num = input("Please enter a buoy number\n")
buoy = gather_data(buoy_num)
buoy = buoy.reset_index()
buoy_mean, buoy_mode, buoy_interpolated = handle_missing_data(buoy)

# output for only buoy_interpolated
print("Test with interpolation imputation\n")
display(buoy_interpolated)
lr_w_int_preds_df, rf_preds_df, two_week_predictions_linear, two_week_predictions_rf = train_model(buoy_interpolated)

# LINEAR PREDICTIONS   
start_date = datetime.now()
end_date = start_date + timedelta(days=14)  # Extend two weeks
prediction_dates = pd.date_range(start=start_date, end=end_date, freq='H')

# Convert prediction_dates to a DataFrame
linear_prediction = pd.DataFrame(prediction_dates, columns=['date'])
num_predictions = len(linear_prediction)

# Merge prediction_dates_df and two_week_predictions_linear_df
linear_prediction['prediction'] = two_week_predictions_linear[-num_predictions:]

print("\nLinear Predictions\n")
display(linear_prediction)

# RANDOM FOREST PREDICTIONS

# Convert prediction_dates to a DataFrame
rf_predictions = pd.DataFrame(prediction_dates, columns=['date'])

# Merge prediction_dates_df and two_week_predictions_linear_df
rf_predictions['prediction'] = two_week_predictions_rf[-num_predictions:]

print("\nRandom Forest Predictions\n")
display(rf_predictions)



