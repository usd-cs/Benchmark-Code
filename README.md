# Benchmark-Code

For this project we are injesting [NDBC](https://www.ndbc.noaa.gov/) data using the open source library called [seebuoy](https://www.seebuoy.com/).  We are using this data to train a prediction model that will predict the wave height and period up to 15 days into the future.

### Authors
Marissa Esteban (marissanicoleesteban@sandiego.edu)<br>
Gabriel Krishnadasan (gkrishnadasan@sandiego.edu)<br>
Diana Montoya-Herrera (dmontoyaherrera@sandiego.edu)<br>
Gabriel Seidel (gseidel@sandiego.edu)<br>
Madeleine Woo (madeleinewoo@sandiego.edu)<br>

## Preprocessing
We decided to use the data outputted by a single buoy that is closest to the given coordinate.  There is no standard time interval for buoy output and there are inconsistencies among buoys that result in missing data.  We use an interpolation method to impute the missing data and standardize the data frames returned by each buoy.

## Prediction methods
We used three different prediction models: [Prophet](https://facebook.github.io/prophet/#:~:text=Prophet%20is%20a%20procedure%20for,several%20seasons%20of%20historical%20data.), Linear Regression, and Random Forest to forecast the two target variables.

## Ports and buoy numbers
The following lists of cities, states, and buoy numbers that are avalible for prediction locations for the Prophet model.  Linear Regression and Random Forest will predict for buoy # 44065
* Brooklyn, NY: 44065
* New Bedford, MA: 44085, 44097
* Salem, MA: 44018, 44013 
* Los Angeles, CA: 46253, 46222
* Santa Barbara, CA: 46054, 46053

## AWS
We integrated the Prophet model ```ProphetScript.py``` file in an AWS Lambda handler which runs every 6 hours to make predictions every 6 hours for the five different locations. The link can be found [here](https://iw5kuyzxuugtjwshexza4tp4ey0zmurd.lambda-url.us-west-1.on.aws)

## Install
* Check [requirements.txt]()
* pip install -r ~/requirements.txt

## Interpreting Results and Prediction Accuracy
The file ```output_script.py``` file will produce the RMSE of all three models as well as their daily errors as the predictions are forecasted further into the future.  These values are ouputted within the terminal and pyplot graphs.

