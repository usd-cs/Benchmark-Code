"""
Purpose:  Deciding a best predictor model for a singular buoy

Authors: Marissa Esteban, Gabe Krishnadasan, Diana Montoya-Herrera, Gabe Seidl, Madeleine Woo
Date: 2/20/2024
"""

# import libraries needed
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, skew, probplot
from scipy.special import boxcox1p
import warnings
warnings.filterwarnings('ignore')
from seebuoy import NDBC
%matplotlib inline 
from sklearn.impute import SimpleImputer


ndbc = NDBC()
wave_df = ndbc.stations()
df_data = ndbc.available_data()

# Get info on NY Harbor Buoy
station_id = "44065"
ny_buoy = ndbc.get_data(station_id)

def handle_missing_data(ny_buoy):
    """
    The data has some missing values. We impute these values by mode, mean, and interpolation.
    """


    # dropping cols where there is 100% NA
    ny_buoy.dropna(axis=1, how='all', inplace=True)

    # dropping rows where target value us null
    ny_buoy.dropna(subset=['average_period'], inplace=True)
    ny_buoy.dropna(subset=['wave_height'], inplace=True)


    # IMPUTATIONS
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

    return (ny_buoy_mean, ny_buoy_mode, ny_buoy_interpolated)

