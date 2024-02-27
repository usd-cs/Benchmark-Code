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
from sklearn.model_selection import train_test_split

from kriging import kriging

def get_buoy_data(buoy_num):
    ndbc = NDBC()

    return ndbc.get_data(buoy_num)

def handle_missing_data(buoy):
    """
    The data has some missing values. We impute these values by mode, mean, and interpolation.
    """
    # dropping cols where there is 100% NA
    no_na = buoy.dropna(axis=1, how='all', inplace=True)

    # dropping rows where target value us null
    buoy = no_na.dropna(subset=['average_period'], inplace=True)
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

    # Interpolate missing vals using kriging
    buoy_kriging = kriging.kriging(no_na)
    buoy_kriging.dropna(subset=['average_period'], inplace=True)
    buoy_kriging.dropna(subset=['wave_height'], inplace=True)
     

    # Remove non finite values
    buoy_mode = buoy_mode[np.isfinite(buoy_mode['wave_height'])]
    buoy_mean = buoy_mean[np.isfinite(buoy_mean['wave_height'])]
    buoy_interpolated = buoy_interpolated[np.isfinite(buoy_interpolated['wave_height'])]
    buoy_kriging = buoy_kriging[np.isfinite(buoy_kriging['wave_height'])]


    return (buoy_mean, buoy_mode, buoy_interpolated, buoy_kriging)

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

buoyTup = handle_missing_data(buoy)

lr_w_int = LinearRegression()
lr_no_int = LinearRegression(fit_intercept=False)
rf = RandomForestRegressor(n_estimators=100)

modelList = [lr_w_int, lr_no_int, rf]
predictAttributesList = ["wave_heigth", "dominant_period"]

for i in range(len(buoyTup)):
    for model in modelList:
        for attribute in predictAttributesList:
            if i == 0:
                print("Imputation method: Mean")
            elif i == 1:
                print("Imputation method: Mode")
            elif i == 2:
                print("Imputation method: Interpolation")
            else:
                print("Imputation method: Krigging")
            predict(buoyTup[i], model, attribute, 0.2)

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


# waveHeightCols = ['wind_speed', 'wind_gust', 'dominant_period',
#        'average_period', 'mean_wave_direction', 'pressure', 'water_temp']
# dominantPeriodCols = ['wind_speed', 'wind_gust', 'wave_height',
#        'average_period', 'mean_wave_direction', 'pressure', 'water_temp']

# mean = buoyTup[0]
# mean = mean * 10

# X_train, X_test, y_train, y_test = train_test_split(mean[waveHeightCols], mean['wave_height'], test_size=0.2, random_state=42)

# y_train = y_train.astype('float')

# rf.fit(X_train, y_train)

# t = rf.predict(X_test)

# print(len(y_test))

# f = list(y_test)

# for i in range(len(t)):
#     t[i] = t[i] / 10
#     f[i] = f[i] / 10

# s = 0
# for i in range(len(t)):
#     s += abs(f[i] - t[i])
#     print(f[i] - t[i])

# print("Avg off: ", s / 660)