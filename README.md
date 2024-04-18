# Benchmark-Code

For this project we are injesting [NDBC](https://www.ndbc.noaa.gov/) data using the open source library called [seebuoy](https://www.seebuoy.com/).  We are using this data to train a prediction model that will predict the wave height and period up to 15 days into the future.

## Contents
[Exploration and Preprocessing](#headers)<br>
[AWS](#headers)<br>
[Install](#headers)<br>
[Run](#headers)<br>
[Interpreting Results](#headers)<br>

### Exploration and Preprocessing
We decided to use the data outputted by a single buoy that is closest to the given coordinate.  There is no standard interval for buoy output and there are inconsistencies among buoys that result in missing data, or no data collected for certain measurements at all.  We use an interpolation method to impute the missing data.

#### Current exploration: 
* Comparing the performance of different prediction algorithms (linear regression, Facebook's [Prophet](https://facebook.github.io/prophet/), and the forecasting predicition model [ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html).
* How much historical data to train the chosen model on (ex. 45 days, 1 year, 8 years etc.)

### AWS
Our goal is to integrate the continuous ingestion of data into the prediction model using seebuoy, OR to use seebuoy to produce real time data on demand to make a prediction.  These predictions should be produced for any given location up to 15 days into the future.

### Install
* Check [requirements.txt]()
* pip install -r ~/requirements.txt

### Run
By the end of this project the executable script will prompt the user to enter a location and desired time into the future to produce a prediction output.

Seebuoy produces a dataframe indexed by date where each column is a paramater measured by the buoys sensors.  We interpolate the missing values and drop any rows where there is not enough data to impute the missing data. These dataframes are used to train the model in predicting our target variables.  These values will be displayed as a time series graph.

### Ports and buoy numbers
The following lists of cities, states, and buoy numbers that are avalible for prediction locations
Brooklyn, NY: 44065
New Bedford, MA: 44085, 44097
Salem, MA: 44018, 44013 
Los Angeles, CA: 46253, 46222
Santa Barbara, CA: 46054, 46053

### Interpreting Results

