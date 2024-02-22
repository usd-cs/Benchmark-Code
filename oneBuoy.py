"""
Purpose:  Deciding a best predictor model for a singular buoy

Authors: Marissa Esteban, Gabe Krishnadasan, Diana Montoya-Herrera, Gabe Seidl, Madeleine Woo
Date: 2/20/2024
"""

# import libraries needed
import pandas as pd
pd.set_option('display.max_columns', None)
import seaborn as sns
import numpy as np
from seebuoy import NDBC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

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
    # dropping cols where there is 100% NA
    buoy.dropna(axis=1, how='all', inplace=True)

    # dropping rows where target value us null
    buoy.dropna(subset=['average_period'], inplace=True)
    buoy.dropna(subset=['wave_height'], inplace=True)

    # IMPUTATIONS
    # Replace missing data with mode
    imputer = SimpleImputer(strategy='most_frequent')
    imputer.fit(buoy)
    buoy_mode = pd.DataFrame(imputer.transform(buoy), columns=buoy.columns)

    # Replace missing data with mean
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(buoy)
    buoy_mean = pd.DataFrame(imputer.transform(buoy), columns=buoy.columns)

    # Interpolate missing values using spline interpolation
    buoy_interpolated = buoy.interpolate(method='spline', order=2)
    del buoy_interpolated["pressure_tendency"]
    buoy_interpolated = buoy_interpolated.dropna()

    # Remove non finite values
    buoy_mode = buoy_mode[np.isfinite(buoy_mode['wave_height'])]
    buoy_mean = buoy_mean[np.isfinite(buoy_mean['wave_height'])]
    buoy_interpolated = buoy_interpolated[np.isfinite(buoy_interpolated['wave_height'])]


    return (buoy_mean, buoy_mode, buoy_interpolated)

def handle_analytics(buoy, model):
    """
    """
    train_df = buoy[buoy.columns.difference(['wave_height', 'average_period'])]
    
    # Validation function for wave height
    n_folds = 5

    kf=KFold(n_splits=n_folds)
    rmse = np.sqrt(-cross_val_score(model, train_df, buoy.wave_height, scoring="neg_mean_squared_error", cv = kf))
    return rmse


buoy = get_buoy_data("44065")

buoyTup = handle_missing_data(buoy)

lr_w_int = LinearRegression()
lr_no_int = LinearRegression(fit_intercept=False)
rf = RandomForestRegressor(n_estimators=100)

modelList = [lr_w_int, lr_no_int, rf]

i = 0
for buoy in buoyTup:
    if i == 0:
        print("lr_w_int")
    elif i == 1:
        print("lr_no_int")
    else:
        print("rf")
    i += 1
    for model in modelList:
        print(handle_analytics(buoy, model))
