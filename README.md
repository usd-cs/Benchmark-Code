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
The following lists of buoy numbers that are avalible for prediction locations for the Prophet model, Linear Regression, and Random Forest. These buoys are pulled from all along both the east and west coasts of the United States.

44091, 44089, 44100, 44086, 41117, 42036, 46232, 46047, 46219, 46251, 46221, 46268, 46222, 46253, 46224, 46275, 46277, 46256, 46274, 46225, 46266, 46014, 46013, 46214, 46026, 46237, 46239, 46011, 46218, 46054, 46053

## AWS
We integrated the Prophet model ```ProphetScript.py``` file in an AWS Lambda handler which runs every 6 hours to make predictions every 6 hours for the five different locations. The link can be found [here](https://iw5kuyzxuugtjwshexza4tp4ey0zmurd.lambda-url.us-west-1.on.aws)

## Install
* Check [requirements.txt]()
* pip install -r ~/requirements.txt

## Interpreting Results and Prediction Accuracy
The file ```output_script.py``` file will produce the RMSE of all three models (for a single buoy) as well as their daily errors as the predictions are forecasted further into the future.  These values are ouputted within the terminal and pyplot graphs.

## Comparing daily error across 30 buoys
The file ```test_30_buoys.py``` outputs a graph of the average daily error for 31 buoys to show which models overall perform the best. 
![derror30buoysheight](https://github.com/usd-cs/Benchmark-Code/assets/143650066/6818031f-c198-4a99-b3db-8ac07102b933)

![derror30buoysperiod](https://github.com/usd-cs/Benchmark-Code/assets/143650066/7a7159fb-efec-4b01-a31c-ff0080e46e0f)
