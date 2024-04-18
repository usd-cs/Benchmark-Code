from seebuoy import NDBC
import pandas as pd
from datetime import datetime, timedelta
from prophet import Prophet

def buoySetUp(buoyNum):  
    """
    Parameters:
    buoyNum: A string value that represents a buoy in the NDBC system

    Returns: A pandas df that contains the historical data of the selected buoy
    """
    ndbc = NDBC(timeframe="historical")

    valid = False
    while (not valid):
        try:
            df_avail = ndbc.available_data(station_id=buoyNum)
            df_data = ndbc.get_data(buoyNum)
            valid = True
        except ValueError:
            buoyNum = input("Please enter a valid buoy number: ")
        
    df_data.dropna(axis=1, how='all', inplace=True)
    df_data = df_data.reset_index()

    if 'wave_height' not in df_data.columns or 'average_period' not in df_data.columns:
        print("Not enough data")
        return False

    # lets limit the df to 2 columns: date and wave height
    buoy_df = df_data[["date","wave_height", "average_period"]]

    # Set 'date' column as the index
    buoy_df = buoy_df.set_index("date")

    buoy_df['wave_height_interpolated'] = buoy_df['wave_height'].interpolate(method='time') # interpolate missing values based on time
    buoy_df['average_period_interpolated'] = buoy_df['average_period'].interpolate(method='time') # interpolate missing values based on time

    return buoy_df


def doit(buoy_df, targetVariable):
    """
    """
    buoy_df = buoy_df.reset_index()

    today_date = datetime.today().date()

    trainDate = today_date - timedelta(days=365)

    training_df = buoy_df[buoy_df['date'] > pd.Timestamp(trainDate)]

    #Sets up the modeling df with the date and target variable column as well as the cap for logistic growth prophet algo
    modeling_df = training_df[["date",f'{targetVariable}_interpolated']]
    modeling_df = modeling_df.rename(columns={"date": "ds", f'{targetVariable}_interpolated': "y"})
    cap = modeling_df['y'].max() + 1

    # Initialize Prophet model with the cap
    model = Prophet(growth="logistic")
    modeling_df['cap'] = cap
    model.fit(training_df)

    #Makes prediction
    future = model.make_future_dataframe(periods=15)
    future['cap'] = cap
    forecast = model.predict(future)

    forecast['date'] = forecast['ds']

def main():

    # prompting user to enter a port of interest
    num = 0
    while not (1<=num and num <=5):
        print("1 - Booklyn, NY \n2 - New Bedford, MA \n3 - Salem, MA \n4 - Los Angeles, CA \n5 - Santa Barbara, CA")
        num = int(input("Enter a port of interest (1-5)"))

    # list of valid buoys (based on research)
    buoy_list = ["44065", "44085", "44013", "46253", "46053"]
    buoy_df = buoySetUp(buoy_list[num-1])


    doit(365, 15, "wave_height", "41064")
        

if __name__ == "__main__":
    main()