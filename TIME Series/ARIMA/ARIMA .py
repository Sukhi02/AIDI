# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 10:07:36 2018

@author: Harinder Singh
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 09:35:14 2018

@author: Harinder Singh
"""


import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from pyramid.arima import auto_arima
import matplotlib.pyplot as plt
import os
import numpy as np



os.chdir("C:/Users/Raju/Desktop/paper/ARIMA")



# UPDATE MONTHLY: Choose starting and ending dates for ARIMA model.
start = '8/10/2011 4:00' 
# 8/10/2012 4:00,35.5,31.3,31.3,40.2,1008201203
end =   '08-10-2012 11:00'



# UPDATE MONTHLY: Choose starting and ending dates for forecast ahead period.
start_f = '08-10-2012 12:00' 
# 8/10/2012 5:00,34.9,30.7,30.7,39.6,1008201204

end_f =  '11/22/2012 5:00'




# Load BLS payrolls data (NSA, SA, and implied SA factors).
# Source: https://data.bls.gov/timeseries/CEU0000000001
df = pd.read_csv('all in 1 moisture.csv', index_col=0)
df =df.drop(['F1'], axis = 1 )

df.index = pd.to_datetime(df.index)
index = df.index

df.describe()





##########################################################################
# Run auto ARIMA proceedure for NSA payrolls and implied BLS SA factors.
##########################################################################

# Auto ARIMA for NSA payrolls.

model_nsa = auto_arima(df.loc[start:end,'Soil moisture at 10m']) 
model_nsa.summary()

# Auto ARIMA for implied BLS SA factors.
model_nsa_25 = auto_arima(df.loc[start:end,'Soil moisture at 25m'])
model_nsa_25.summary()


model_sa_factors_50 = auto_arima(df.loc[start:end,'Soil moisture at 50 m'])
model_sa_factors_50.summary()

model_sa_factors_80 = auto_arima(df.loc[start:end,'Soil moisture at 80m'])
model_sa_factors_80.summary()


#################################################
# Create 12-month ahead forecast of NSA payrolls. 
#################################################

# Create training and forecast data frames.
train = df.loc[start:end,'Soil moisture at 10m']
test = df.loc[start_f:end_f, 'Soil moisture at 10m']

# Fit arima model on NSA payrolls time series. 
model_nsa.fit(train)

# Forecast 12 months ahead. 
forecast_nsa = model_nsa.predict(n_periods = 2488)

# Create forecast dataframe.
forecast_nsa_df = pd.DataFrame(forecast_nsa,  columns=['Soil moisture at 10m'])


###########################################################
# Create 12-month ahead forecast of BLS implied SA factors. 
###########################################################

# Create training and forecast data frames.
train_nsa_25 = df.loc[start:end,'Soil moisture at 25m']
test_nsa_25 = df.loc[start_f:end_f, 'Soil moisture at 25m']

# Fit arima model on BLS implied SA factors time series.implied_sa_factors
model_nsa_25.fit(train_nsa_25)



# Forecast 12 months ahead. 
forecast_model_nsa_25 = model_nsa_25.predict(n_periods = 2489)

# Create forecast dataframe.
forecast_model_nsa_25 = pd.DataFrame(forecast_model_nsa_25,  columns=['Soil moisture at 25m'])

#################################################
# Create 12-month ahead forecast of NSA payrolls. 
#################################################

# Create training and forecast data frames.
train_50 = df.loc[start:end,'Soil moisture at 50 m']
test_50 = df.loc[start_f:end_f, 'Soil moisture at 50 m']

# Fit arima model on NSA payrolls time series. 
model_sa_factors_50.fit(train_50)

# Forecast 12 months ahead. 
forecast_nsa_50 = model_sa_factors_50.predict(n_periods = 2489)

# Create forecast dataframe.
forecast_nsa_50_df = pd.DataFrame(forecast_nsa_50,  columns=['Soil moisture at 50 m'])

#################################################
# Create 12-month ahead forecast of NSA payrolls. 
#################################################

# Create training and forecast data frames.
train_80 = df.loc[start:end,'Soil moisture at 80m']
test_80 = df.loc[start_f:end_f, 'Soil moisture at 80m']

# Fit arima model on NSA payrolls time series. 
model_sa_factors_80.fit(train_80)


# Forecast 78 days ahead. 
forecast_nsa_80 = model_sa_factors_80.predict(n_periods = 2489)

# Create forecast dataframe.
forecast_nsa_80_df = pd.DataFrame(forecast_nsa_80,  columns=['Soil moisture at 80m'])


###########################################
# Merge full forecast data together.
###########################################

# Merge together forecasts of NSA payrolls and implied BLS SA factors. 
df_forecast = pd.concat([forecast_nsa_df,forecast_model_nsa_25,forecast_nsa_50_df,forecast_nsa_80_df],axis=1)





### O R I IG I N A L df.plot()



new_test = df.loc[start_f:end_f]


new_test.plot()
plt.show()


plt.plot(new_test['Soil moisture at 10m' ], color='green', label='Moisture Level at 10 m' )
plt.xlabel('Time')
plt.ylabel('Moisture Level at 10m')
plt.title('Actual Moisture')
plt.savefig('test_Moisture Level at 10m')
plt.legend()
plt.show()


plt.plot(new_test['Soil moisture at 25m' ], color='blue', label='Moisture Level at 25 m' )
plt.xlabel('Time')
plt.ylabel('Moisture Level at 25m')
plt.title('Actual Moisture')
plt.savefig('test_Moisture Level at 25m')
plt.legend()
plt.show()




plt.plot(new_test['Soil moisture at 50 m' ], color='brown', label='Moisture Level at 50 m' )
plt.xlabel('Time')
plt.ylabel('Moisture Level at 50m')
plt.title('Actual Moisture')
plt.savefig('test_Moisture Level at 50 m')
plt.legend()
plt.show()



plt.plot(new_test['Soil moisture at 80m' ], color='grey', label='test_Moisture Level at 80 m' )
plt.xlabel('Time')
plt.ylabel('Moisture Level at 80m')
plt.title('Actual Moisture')
plt.savefig('test_Moisture Level at 80m')
plt.legend()
plt.show()


plt.plot( new_test['Soil moisture at 10m'], color='g', label='Moisture Level at 10 m')
plt.plot( new_test['Soil moisture at 25m'], color='blue', label='Moisture Level at 25 m')
plt.plot( new_test['Soil moisture at 50 m'], color = 'brown', label='Moisture Level at 50 m')
plt.plot( new_test['Soil moisture at 80m'], color = 'grey', label='Moisture Level at 80 m')
plt.xlabel('Time')
plt.ylabel('Moisture Level')
plt.title('Actual Moisture')
plt.legend()
plt.savefig('test_Moisture Level at 10, 25,50, 80 ')
plt.show()




















index = test.index
plt.plot(index, forecast_nsa_df, color='g' ,label='Moisture Level at 10 m')
plt.xlabel('Time')
plt.ylabel('Moisture Level at 10m')
plt.title('Forecasted moisture')
plt.savefig('Moisture Level at 10 m')
plt.legend()
plt.show()

plt.plot(index, forecast_model_nsa_25, color='blue',label='Moisture Level at 25 m')
plt.xlabel('Time')
plt.ylabel('Moisture Level at 25m')
plt.title('Forecasted moisture')
plt.savefig('Moisture Level at 25 m')
plt.legend()
plt.show()

plt.plot(index, forecast_nsa_50_df, color='brown' ,label='Moisture Level at 50 m')
plt.xlabel('Time')
plt.ylabel('Moisture Level at 50m')
plt.title('Forecasted moisture')
plt.savefig('Moisture Level at 50 m')
plt.legend()
plt.show()

plt.plot(index, forecast_nsa_80_df, color='grey', label='Moisture Level at 80 m' )
plt.xlabel('Time')
plt.ylabel('Moisture Level at 80m')
plt.title('Forecasted moisture')
plt.savefig('Moisture Level at 80m')
plt.legend()
plt.show()




plt.plot(index, forecast_nsa_df, color='g', label='Moisture Level at 10 m')
plt.plot(index, forecast_model_nsa_25, color='blue', label='Moisture Level at 25 m')
plt.plot(index, forecast_nsa_50_df, color='brown', label='Moisture Level at 50 m')
plt.plot(index, forecast_nsa_80_df, color='grey', label='Moisture Level at 80 m')
plt.xlabel('Time')
plt.ylabel('Moisture Level')
plt.title('Forecasted moisture')
plt.legend()
plt.savefig('Moisture Level at 10, 25,50, 80 ')
plt.show()

# Push out 12-month SA nonfarm payrolls forecast to CSV.
#
df_forecast.to_csv('forecast1.csv')


### end ###

"""
n = 100

A_forecast_nsa = model_nsa.predict(n_periods = n)
A_forecast_model_nsa_25 = model_nsa_25.predict(n_periods = n)
A_forecast_nsa_50 = model_sa_factors_50.predict(n_periods = n)
A_forecast_nsa_80 = model_sa_factors_80.predict(n_periods = n)



#A_df_forecast_1 = pd.concat([A_forecast_nsa,A_forecast_model_nsa_25,A_forecast_nsa_50,A_forecast_nsa_80],axis=1)

"""
from sklearn.metrics import mean_squared_error

#mean_squared_error(y_true, y_pred)

#mean_squared_error(test_nsa_25, )

forecast_nsa_df,forecast_model_nsa_25,forecast_nsa_50_df,


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



a_10 = mean_absolute_percentage_error(test, forecast_nsa_df  )
print("Mean absolute percentage error :",a_10)

a_25= mean_absolute_percentage_error(test_nsa_25, forecast_model_nsa_25 )
print("Mean absolute percentage error :",a_25)

a_50= mean_absolute_percentage_error(test_50,forecast_nsa_50_df )
print("Mean absolute percentage error :",a_50)

a_80 = mean_absolute_percentage_error(test_80,forecast_nsa_80_df )
print("Mean absolute percentage error :",a_80)

a=(( a_10+a_25+a_50+a_80) / 4)
print("Final Mean absolute percentage error :" ,a)
