import openturns as ot
from seebuoy import NDBC
import pandas as pd
import numpy as np
import datetime
from sklearn.metrics import root_mean_squared_error



def kriging(buoy_frame):
    temp_bf = buoy_frame
    for attribute in buoy_frame.columns:
        if (attribute != "wave_height" and attribute != "average_period"):
            training_vars = (list(buoy_frame[attribute]))
            final_vars = (list(buoy_frame[attribute])) #list of variables to train on, will remove nan, and list to insert values back into
            dates = (list((buoy_frame.index))) #all orig dates
            dates_ind = [] #index of dates to run on krig algo
            nan_dates = [] #indexes where there is nan

            #fill index set with indexes 0 to n
            for i in range(len(dates)):
                dates_ind.append(float(i))

            #if pressure is nan, remove from pressure list and move related index to nan_dates
            for j in range(len(training_vars) - 1, -1, -1):
                if pd.isna(training_vars[j]):
                    training_vars.pop(j)
                    nan_dates.insert(0,float(dates_ind.pop(j)))
            

            if (len(nan_dates) != len(dates) and len(nan_dates) != 0):
                #inputs
                time_train = ot.Sample([[x] for x in dates_ind])
                #outputs
                pressure_train = ot.Sample([[y] for y in training_vars])

                # Fit
                total = 0
                inputDimension = 1
                basis = ot.ConstantBasisFactory(inputDimension).build()
                covarianceModel = ot.SquaredExponential([1.]*inputDimension, [1.0])
                algo = ot.KrigingAlgorithm(time_train, pressure_train, covarianceModel, basis)
                algo.run()
                result = algo.getResult()
                krigingMetamodel = result.getMetaModel()

                p_predictions = []
                #create a list of the predictions
                for x in range(len(nan_dates)):
                    input_time = [nan_dates[x]]
                    predict_pressure = str(krigingMetamodel(input_time))
                    p_predictions.append(float(predict_pressure[1:len(predict_pressure) - 1]))

                #replace all nan with predictions    
                for y in range(len(nan_dates)):
                    final_vars[int(nan_dates[y])] = p_predictions[y]

                for z in range(len(final_vars)):
                    temp_bf[attribute][z] = final_vars[z]
    return temp_bf



