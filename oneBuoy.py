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

def get_buoy_data(buoy_num):
    ndbc = NDBC()
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

    return (buoy_mean, buoy_mode, buoy_interpolated)



def handle_outliers(buoy):
    """
    """

    # Remove non finite values
    ny_buoy_mode = ny_buoy_mode[np.isfinite(ny_buoy_mode['wave_height'])]
    ny_buoy_mean = ny_buoy_mean[np.isfinite(ny_buoy_mean['wave_height'])]
    ny_buoy_interpolated = ny_buoy_interpolated[np.isfinite(ny_buoy_interpolated['wave_height'])]



def handle_analytics(buoy, model):
    """
    """

    train_df = ny_buoy[ny_buoy_mode.columns.difference(['wave_height', 'average_period'])]
    
    # Validation function for wave height
    n_folds = 5

    def rmse_cv(model,n_folds):
        kf=KFold(n_splits=n_folds)
        rmse = np.sqrt(-cross_val_score(model, train_df, ny_buoy_mode.wave_height, scoring="neg_mean_squared_error", cv = kf))
        return rmse


buoy = get_buoy_data("44065")