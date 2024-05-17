from output_script import *
"""
The purpose of this script is to run all three models on over 30 separate
buoys located around the east and west coasts of the United States.

The output is a graph of the average daily error for all three models.
"""

def test_30_buoys():
    buoys = [44091, 44089, 44100] # list of compatible buoys

    # create an empty dataframe to track the daily error for all three models
    # Define the number of rows and column names
    num_rows = 12
    column_names = ["linear regression", "random forest", "prophet"]

    # Create an empty DataFrame with float values for wave height
    daily_error_height = pd.DataFrame(0.0, index=range(num_rows), columns=column_names)
    daily_error_height.insert(0, 'day', range(1, num_rows + 1))    

    # Create an empty DataFrame with int values for wave period
    daily_error_period = pd.DataFrame(0, index=range(num_rows), columns=column_names)
    daily_error_period.insert(0, 'day', range(1, num_rows + 1))    


    for buoy_num in buoys:
        buoy = gather_data(str(buoy_num)) 

        display(buoy)

        buoy = buoy.reset_index()

        buoy_cleaned = handle_missing_data(buoy)

        # make predictions for average period and wave height
        period_linear, period_rf = predict(45, 14, "average_period", buoy, buoy_cleaned)
        height_linear, height_rf = predict(45, 14, "wave_height", buoy, buoy_cleaned)
        prophet_wave_height = predict_prophet('wave_height', str(buoy_num))
        prophet_avg_period = predict_prophet('average_period', str(buoy_num))

        # make a table of the daily errors and keep a running sum of daily error for the 14 days

        ############### WAVE HEIGHT DAILY ERROR ###############
        derror_lin_height = daily_error(height_linear["wave_height"].tolist(), height_linear["prediction"].tolist())
        derror_rf_height = daily_error(height_rf["wave_height"].tolist(), height_rf["prediction"].tolist())
        prophet_wave_height["difference"] = prophet_wave_height['difference'] = prophet_wave_height['yhat'] - prophet_wave_height['wave_height_interpolated']
        prophet_wave_height['difference'] = prophet_wave_height['difference'].abs()
        prophet_wave_height.reset_index().index

        # add daily errors to the columns in daily_error df
        daily_error_height['linear regression'] = daily_error_height['linear regression'] + derror_lin_height['daily_error']
        daily_error_height['random forest'] = daily_error_height['random forest'] + derror_rf_height['daily_error']
        daily_error_height['prophet'] = daily_error_height['prophet'] + prophet_wave_height['difference']

        ############### AVERAGE PERIOD DAILY ERROR ###############
        derror_lin_period = daily_error(period_linear["average_period"].tolist(), period_linear["prediction"].tolist())
        derror_rf_period = daily_error(period_rf["average_period"].tolist(), period_rf["prediction"].tolist())
        prophet_avg_period["difference"] = prophet_wave_height['difference'] = prophet_wave_height['yhat'] - prophet_wave_height['average_period_interpolated']
        prophet_avg_period['difference'] = prophet_wave_height['difference'].abs()
        prophet_avg_period.reset_index().index    

        # add daily errors to the columns in daily_error df
        daily_error_period['linear regression'] = daily_error_period['linear regression'] + derror_lin_period['daily_error']
        daily_error_period['random forest'] = daily_error_period['random forest'] + derror_rf_period['daily_error']
        daily_error_period['prophet'] = daily_error_period['prophet'] + prophet_avg_period['difference']
    
    avg_derror_height = daily_error_height/len(buoys)
    avg_derror_period = daily_error_period/len(buoys)

    plot_avg_error("wave_height", avg_derror_height)
    plot_avg_error("average_period", avg_derror_period)


def plot_avg_error(target, avg_derror):
    """
    plots the average daily error for 30+ buoys
    @param target: the target value we're predicting for (wave height or average period)
    @param avg_derror: dataframe of the average daily error for each model
    """
    plt.plot(avg_derror.index, avg_derror["linear regression"], label='Linear Regression')
    plt.plot(avg_derror.index, avg_derror["random forest"], label='Random Forest')
    plt.plot(avg_derror.index, avg_derror['prophet'], label='Prophet')
    
    if target == "wave_height":
        plt.title('Average Daily Error on 30+ buoys - Wave Height')
        plt.ylabel('Error (meters)')
    elif target == "average_period":
        plt.title('Average Daily Error on 30+ buoys - Avg Period')
        plt.ylabel('Error (seconds)')

    plt.xlabel('Day')
    plt.legend()
    plt.show()
    
test_30_buoys()

