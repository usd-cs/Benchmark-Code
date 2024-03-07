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
    if predictedAttribute == "wave_height":
        cols = ['wind_speed', 'wind_gust', 'dominant_period',
                'average_period', 'mean_wave_direction', 'pressure', 'water_temp']
        X_train, X_test, y_train, y_test = train_test_split(buoy[cols], buoy['wave_height'], test_size=testSize, random_state=42)
    elif predictedAttribute == "dominant_period":
        cols = ['wind_speed', 'wind_gust', 'wave_height',
                'average_period', 'mean_wave_direction', 'pressure', 'water_temp']
        X_train, X_test, y_train, y_test = train_test_split(buoy[cols], buoy['dominant_period'], test_size=testSize, random_state=42)

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

    avgOff = sumOfDifferences / len(predictions)

    print("Model used: ", model)
    print("Predicted attribute: ", predictedAttribute)
    print("Avg off: ", sumOfDifferences / len(predictions))

    return avgOff

buoy = get_buoy_data("44065")

buoyTup = handle_missing_data(buoy)

lr_w_int = LinearRegression()
lr_no_int = LinearRegression(fit_intercept=False)
rf = RandomForestRegressor(n_estimators=100)

modelList = [lr_w_int, lr_no_int, rf]
predictAttributesList = ["wave_height", "dominant_period"]

########################################
"""
TEST FOR DATA FILLING
"""
# meanCount = 0
# modeCount = 0
# interpolateCount = 0

# for i in range(30):
#     temp = []
#     for buoy in buoyTup:
#         temp.append(predict(buoy, rf, "dominant_period", 0.2))

#     if temp[0] < temp[1] and temp[0] < temp[2]:
#         meanCount += 1
#     elif temp[1] < temp[0] and temp[1] < temp[2]:
#         modeCount += 1
#     else:
#         interpolateCount += 1


# print("mean: ", meanCount)
# print("mode: ", modeCount)
# print("interpolate: ", interpolateCount)

########################################
"""
TEST FOR MODEL USAGE
"""
wIntCount = 0
noIntCount = 0
rfCount = 0

for i in range(30):
    temp = []
    for model in modelList:
        temp.append(predict(buoyTup[0], model, "dominant_period", 0.2))

    if temp[0] < temp[1] and temp[0] < temp[2]:
        wIntCount += 1
    elif temp[1] < temp[0] and temp[1] < temp[2]:
        noIntCount += 1
    else:
        rfCount += 1


print("With int: ", wIntCount)
print("No int: ", noIntCount)
print("RF: ", rfCount)
########################################


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
