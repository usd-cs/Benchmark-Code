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
from prophet_rse import *

def gather_data(buoy_num):
    """
    Imports data from the National Data Buoy Center (NOAA)
    @param buoy_num: the buoy number of the closest x,y coordinates
    @return see_buoy: the buoy data in a dataframe
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
    The data has some missing values. We impute these values with interpolation.

    @param buoy: the data imported from Seebuoy
    @return buoy_interpolated: a dataframe of buoy data where we impute the missing values
    with interpolation
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


    # Interpolate missing values using spline interpolation
    buoy_interpolated = buoy.copy()
    for column in columns_to_fill:
        if column in buoy_interpolated:
            buoy_interpolated[column] = buoy_interpolated[column].fillna(buoy_interpolated[column].interpolate(method='spline', order=2))
    
    # check if there are any additional missing values
    for column in columns_to_fill:
        if column in buoy_interpolated.columns and buoy_interpolated[column].isnull().any():
            buoy_interpolated = buoy_interpolated.drop(column, axis=1)

    return buoy_interpolated

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
    - target_column: str default="wave_height"
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
    - two_week_predictions: dataframe containing the predictions for the target two weeks into the future.
    """

    # Sort the DataFrame based on the date column
    data = data.sort_values(by=date_column)
    # data = data.set_index('date')
    data = data.reset_index(drop=True)

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
    prediction_dates_df["mean_wave_direction"] = pd.NA
    prediction_dates_df["water_temp"] = pd.NA
    if target_column == "wave_height":
        prediction_dates_df["average_period"] = pd.NA
    elif target_column == "average_period":
        prediction_dates_df["wave_height"] = pd.NA

    # make a new DataFrame with columns from old data
    extended_data = data.iloc[:, :]
    extended_data = pd.concat([extended_data, prediction_dates_df], axis=0)

    buoy_interpolated = handle_missing_data(extended_data)
    
    buoy_interpolated = buoy_interpolated.drop(["date", target_column], axis=1)
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

    @return: none
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


def train_model(buoy, target):
    """
    Train the model for linear regression and random forest
    Parameters:
    - buoy: the data we're training on
    - target: str the thing we're trying to predict (either
    wave_height or average_period)

    Returns:
    - lr_w_int_preds_df: linear regression with intercept results containing the columns: "date", "Actual", "Predicted", "Fold", "Set"
    - rf_preds_df: random forest with intercept results containing the columns: "date", "Actual", "Predicted", "Fold", "Set"
    - two_week_predictions_linear: the two week forecast for linear regression
    - two_week_predictions_rf: the two week forecast for random forest
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
        two_week_predictions_linear,
    ) = time_series_split_regression(
        buoy,
        regressor=lr_w_int,
        target_column=target
    )

    ###################### RANDOM FOREST MODEL ##################
    rf = RandomForestRegressor(n_estimators=100)

    rf_preds_df, rf_rmse, rf_split_dates, num_records, two_week_predictions_rf = time_series_split_regression(
        buoy,
        regressor=rf,
    )

    return lr_w_int_preds_df, rf_preds_df, two_week_predictions_linear, two_week_predictions_rf

def display_results(predictions):
    """
    Show a line plot with the trained data, real data, and predictions
    @param: predictions, a dataframe of predictions
    @return: none
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

def calculate_rmse(merged_linear, merged_rf):
    """
    Calculates the root mean squared error of the linear regression
    predictions and random forest predictions

    Parameters:
    merged_linear:
    merged_rf:

    target : either "wave_height" or "average_period". Variable we want to traiun and predict on.
    """

    rmse_linear = rmse(merged_linear["wave_height"].tolist(), merged_linear["prediction"].tolist())
    rmse_rf = rmse(merged_rf["wave_height"].tolist(), merged_rf["prediction"].tolist())

    return rmse_linear, rmse_rf

def predict(f, c, target, data, buoy_interpolated):
    """
    Returns predictions using random forest regression and linear regression.
    This function is for testing the accuracy of our model. It lags the data 
    so we can calculate the daily error for the past two weeks.

    Parameters:
    - f: the number of days we used to train (ex: 45 days)
    - c: the number of days we are forecasting (ex: 14 days)
    - target: the thing we're trying to predict (average period or wave height)
    - data: dataframe of not cleaned data
    - buoy_interpolated: dataframe of cleaned data

    Returns:
    - merged_linear: dataframe of linear regression predictions and the recent 14 days of data (predictions and actual data)
    - merged_rf: dataframe of random forest predictions and the recent 14 days of data
    """
    # Sets up date objects and floor and ceiling
    today_date = datetime.today().date()
    floor = f + c
    ceiling = c

    # Calculate the floordate and the ceiling date
    floorDate = today_date - timedelta(days=floor)
    ceilingDate = today_date - timedelta(days=ceiling)

    # Splits up the df into recent and past
    # Recent holds the 14 most recent days of data
    # Past holds all data from the floor to the ceiling
    recent_df = data[data['date'] > pd.Timestamp(ceilingDate)]
    past_df = buoy_interpolated[(buoy_interpolated['date'] > pd.Timestamp(floorDate)) & (buoy_interpolated['date'] < pd.Timestamp(ceilingDate))]

    lr_w_int_preds_df, rf_preds_df, two_week_predictions_linear, two_week_predictions_rf = train_model(past_df, target)

    # LINEAR PREDICTIONS   
    start_date = datetime.now()
    end_date = start_date + timedelta(days=14)  # Extend two weeks
    prediction_dates = pd.date_range(start=start_date, end=end_date, freq='D').strftime('%Y-%m-%d %H:%M:%S')

    # Convert prediction_dates to a DataFrame
    linear_prediction = pd.DataFrame(prediction_dates, columns=['date'])
    num_predictions = len(linear_prediction)
    
    # Merge prediction_dates_df and two_week_predictions_linear_df
    linear_prediction['prediction'] = two_week_predictions_linear[-num_predictions:]

    # RANDOM FOREST PREDICTIONS
    # Convert prediction_dates to a DataFrame
    rf_predictions = pd.DataFrame(prediction_dates, columns=['date'])

    # Merge prediction_dates_df and two_week_predictions_linear_df
    rf_predictions['prediction'] = two_week_predictions_rf[-num_predictions:]

    # Assuming 'date' column in recent_df is not datetime type
    recent_df['date'] = pd.to_datetime(recent_df['date'])

    # Assuming 'date' column in linear_prediction is not datetime type
    linear_prediction['date'] = pd.to_datetime(linear_prediction['date'])
    rf_predictions['date'] = pd.to_datetime(rf_predictions['date'])
    
    # Merge linear_prediction and recent_df using merge_asof
    merged_linear = pd.merge_asof(linear_prediction, recent_df, on='date', direction='nearest')

    # Merge rf_predictions and recent_df using merge_asof
    merged_rf = pd.merge_asof(rf_predictions, recent_df, on='date', direction='nearest')

    return merged_linear, merged_rf

def rmse(y_true, y_pred):
    """
    Compute Root Mean Squared Error (RMSE).
    
    Parameters:
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
        
    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    Returns:
    float
        The RMSE value.
    """
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def to_table(future):
    """
    Create a table of predictions with corresponding day and time

    @param future: the dataframe of predicted and actual values for the future
    @return table of predictions with corresponding day and time
    """
    table = []
    
    for index, row in future.iterrows():
        day = datetime.strptime(str(row['date']), "%Y-%m-%d %H:%M:%S").strftime("%A %B %d")
        predicted = str(round(row['prediction'],2))
        
        table.append([day, predicted])
    
    return table

def calculate_std(predictions):
    """
    Calculates the standard deviation of the predictions

    @param predictions: the array of predictions we have
    @return standard_dev: the standard deviation of the predictions
    """
    standard_dev = predictions.std()
    return standard_dev


def graph_std(target, prophet_std, linear_std, rf_std):
    """
    Graphs the standard deviations of the predictions for
    all three algorithms for either wave height or wave period

    @param target: the target we're predicting for (average period or wave height)
    @param prophet_std: Prophet algorithm's standard deviation
    @param linear_std: the linear standard deviation
    @param rf_std: random forest standard deviation

    @return: none
    """
    # Plotting the bars
    plt.bar(['Prophet', 'Linear Regression', 'Random Forest'], [prophet_std, linear_std, rf_std])

    if target == "wave_height":
        plt.title('Standard Deviation of Wave Height Predictions')
    elif target == "average_period":
        plt.title('Standard Deviation of Average Period Predictions')

    # Show plot
    plt.tight_layout()
    plt.show()


def graph_daily_error(merged_linear, merged_rf, prediction_type, df):
    """
    Uses the calculated daily error from daily_error() and displays it in graph form

    @param: merged_linear
    @param: prediction_type (wave height or average period)

    @return: none
    """
    daily_error_linear = daily_error(merged_linear["wave_height"].tolist(), merged_linear["prediction"].tolist())
    daily_error_rf = daily_error(merged_rf["wave_height"].tolist(), merged_rf["prediction"].tolist())

    plt.plot(merged_linear.index, daily_error_linear["daily_error"], label='Linear Regression')
    
    plt.plot(merged_rf.index, daily_error_rf["daily_error"], label='Random Forest')

    # add Prophet's daily error
    df["difference"] = df['difference'] = df['yhat'] - df[f'{prediction_type}_interpolated']
    df['difference'] = df['difference'].abs()

    plt.plot(df.reset_index().index, df['difference'], label='Prophet')
    
    if prediction_type == "wave_height":
        plt.title('Daily Error Comparison Wave Height')
    elif prediction_type == "average_period":
        plt.title('Daily Error Comparison Average Period')
    plt.xlabel('Date')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

def predict_prophet(prediction_type, buoy_num):
    """
    Calls the prophet model with our given params. 
    rse_per_day() comes from the prophet_rse script.

    @param prediction_type: the target variable we're predicting for
    @param buoy_num: buoy number to predict for

    @return: the dataframe of predicted values using prophet
    """
    df = rse_per_day(720, 15, prediction_type, buoy_num)
    return df


def daily_error(y_true, y_pred):
    """
    Compute Daily Error.

    Parameters:
        y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
        
    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    Returns:
        dataframe of y_true, y_pred, and daily error
    """
    # Check if lengths of y_true and y_pred are equal
    if len(y_true) != len(y_pred):
        raise ValueError("Lengths of y_true and y_pred must be equal.")

    # Calculate residuals
    daily_error = abs(np.array(y_true) - np.array(y_pred))
    
    # Create DataFrame
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'daily_error': daily_error})
    
    return df


def main():
    """
    Predict wave_height and average_period 2 weeks in advance
    """

    variable_list = ['wave_height', 'average_period']
    buoy_num = "44091"

    # # prompting user to enter a target variable
    target_variable_choice = 0
    while target_variable_choice != 1 and target_variable_choice != 2:
        print("1 - Wave Height \n2 - Wave Period")
        target_variable_choice = int(input("Enter prediction choice: "))


    buoy = gather_data(buoy_num) #gather_data(buoy_list[location_choice-1])
    buoy = buoy.reset_index()
    target_variable = variable_list[target_variable_choice - 1]

    buoy_cleaned = handle_missing_data(buoy)

    period_linear, period_rf = predict(45, 14, "average_period", buoy, buoy_cleaned)
    rmse_linear, rmse_rf = calculate_rmse(period_linear, period_rf)
    print(f"LINEAR AVERAGE PERIOD RMSE: {rmse_linear} \nRF AVERAGE PERIOD RMSE: {rmse_rf}\n")
    table_period = to_table(period_linear) 

    height_linear, height_rf = predict(45, 14, "wave_height", buoy, buoy_cleaned)
    rmse_linear, rmse_rf = calculate_rmse(height_linear, height_rf)
    print(f"LINEAR WAVE HEIGHT RMSE: {rmse_linear} \nRF WAVE HEIGHT RMSE: {rmse_rf}\n")
    table_height = to_table(height_linear) 

    ###################### GRAPHING ERROR ######################
    prophet_wave_height = predict_prophet('wave_height', buoy_num)
    prophet_avg_period = predict_prophet('average_period', buoy_num)
    graph_daily_error(height_linear, height_rf, 'wave_height', prophet_wave_height)
    graph_daily_error(period_linear, period_rf, 'average_period', prophet_avg_period) 

    ###################### GRAPHING STANDARD DEVIATION ######################

    # wave height std dev
    linear_std = calculate_std(height_linear['prediction'])
    rf_std = calculate_std(height_rf['prediction'])
    prophet_std = calculate_std(prophet_wave_height['yhat'])
    graph_std('wave_height', prophet_std, linear_std, rf_std)

    # avg period std dev
    linear_std = calculate_std(period_linear['prediction'])
    rf_std = calculate_std(period_rf['prediction'])
    prophet_std = calculate_std(prophet_avg_period['yhat'])
    graph_std('average_period', prophet_std, linear_std, rf_std)



if __name__ == "__main__":
    main()