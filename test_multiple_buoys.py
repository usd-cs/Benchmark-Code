from seebuoy import NDBC
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

# TODO: add parameters to specify x,y location 
def data_ingestion():
    """
    @param: none
    @return: ny_buoy, a dataframe containing data for all buoys in the NY area
    """
    # create NDBC object from seeabuoy to access buoy data
    ndbc = NDBC()

    # Information on NDBC's ~1800 buoys and gliders
    wave_df = ndbc.stations()

    # list all available data for all buoys
    df_data = ndbc.available_data()

    df_buoys = ndbc.stations()
    m = df_buoys["closest_state"] == "New York"
    df_ny = df_buoys[m]

    df_available = ndbc.available_data(dataset="all")

    # subset down to ny stations
    m = df_available["station_id"].isin(df_ny["station_id"])
    df_ny_avail = df_available[m] # using the mask

    piv_ny = pd.pivot_table(
        df_ny_avail, 
        index="station_id", 
        columns="dataset", 
        aggfunc=len, 
        values="file_name"
    )

    ny_station_ids = piv_ny.index.tolist()

    # creating a df for the combined data for all buoys around NY
    ny_buoys_standard = pd.DataFrame()

    for station_id in ny_station_ids:

        # get the standard data for a singular buoy
        df_station = ndbc.get_data(station_id, dataset="standard")
        
        # Add a column for station ID
        df_station['station_id'] = station_id
        
        # Concatenate the current station's data to the combined DataFrame
        ny_buoys_standard = pd.concat([ny_buoys_standard, df_station], ignore_index=False)

    ny_buoy = ny_buoys_standard
    print("finished data ingestion\n")
    return ny_buoy

def handle_missing_data(ny_buoy):
    """
    @param: ny_buoy, data for training
    @return: a tuple of ny_buoy_mode, ny_buoy_mean, and ny_buoy_interpolated, 
    the dataframes for ny_buoy but with imputed data and
    no outliers
    """

    # dropping cols where there is 100% NA
    ny_buoy.dropna(axis=1, how='all', inplace=True)

    # dropping rows where average_period is null
    ny_buoy.dropna(subset=['average_period'], inplace=True)

    # dropping rows wehre wave_height is null
    ny_buoy.dropna(subset=['wave_height'], inplace=True)

    # Replace missing data with mode
    imputer = SimpleImputer(strategy='most_frequent')
    imputer.fit(ny_buoy)
    ny_buoy_mode = pd.DataFrame(imputer.transform(ny_buoy), columns=ny_buoy.columns)

    # Replace missing data with mean
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(ny_buoy)
    ny_buoy_mean = pd.DataFrame(imputer.transform(ny_buoy), columns=ny_buoy.columns)

    # Interpolate missing values using spline interpolation
    ny_buoy_interpolated = ny_buoy.interpolate(method='spline', order=2)

    # Remove non finite values
    ny_buoy_mode = ny_buoy_mode[np.isfinite(ny_buoy_mode['wave_height'])]
    ny_buoy_mean = ny_buoy_mean[np.isfinite(ny_buoy_mean['wave_height'])]
    ny_buoy_interpolated = ny_buoy_interpolated[np.isfinite(ny_buoy_interpolated['wave_height'])]

    return (ny_buoy_mode, ny_buoy_mean, ny_buoy_interpolated)

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


    # Example usage:
    # result_df, rmse_scores, split_dates, num_records = time_series_split_regression(data, 'date_column', 'target_column', regressor)

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

    return average # TODO: why this here????

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

def train_model(ny_buoy):
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
        ny_buoy,
        regressor=lr_w_int,
    )
    # Print RMSE scores and split dates for each split
    print_rmse_and_dates(
        lr_w_int_rmse, lr_w_int_split_dates, num_records, "Linear Regression (w/ Intercept)"
    )

    ###################### RANDOM FOREST MODEL ##################
    rf = RandomForestRegressor(n_estimators=100)

    rf_preds_df, rf_rmse, rf_split_dates, num_records = time_series_split_regression(
        ny_buoy,
        regressor=rf,
    )
    print_rmse_and_dates(rf_rmse, rf_split_dates, num_records, "Random Forest")


# PUT THE WHOLE SHEBANG TOGETHER

# ingest data
ny_buoy = data_ingestion()

# handle missing data
ny_buoy_mode, ny_buoy_mean, ny_buoy_interpolated = handle_missing_data(ny_buoy)

# train the model
print("Test with mode imputation\n")
train_model(ny_buoy_mode)

print("Test with mean imputation\n")
train_model(ny_buoy_mean)

print("Test with interpolation imputation\n")
train_model(ny_buoy_interpolated)

