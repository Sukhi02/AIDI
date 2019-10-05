
import pandas as pd
from pandas import read_csv
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
  

import os
import numpy as np
import matplotlib.pyplot as plt

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def Error(y_true, y_pred): 
     return ((np.array(y_true)- np.array(y_pred)))
     
from sklearn.preprocessing import MinMaxScaler




from pandas.plotting import autocorrelation_plot
from pandas import Series
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf










################ Soil Moisture at 10 m #########################################

os.chdir("C:/Users/Raju/Desktop/paper/IIT K/ARIMA")



raw_dataset = read_csv('all in 1 moisture.csv' , parse_dates=["date_time"], index_col= "date_time")



raw_dataset1 = read_csv('all in 1 moisture.csv')
plt.plot(dataset)
plt.show()






dataset = raw_dataset.iloc[:,1:5].values
dataset_10 = raw_dataset.iloc[:, 0:1].values
dataset_25 = raw_dataset.iloc[:, 1:2].values
dataset_50 = raw_dataset.iloc[:, 2:3].values
dataset_80 = raw_dataset.iloc[:, 3:4].values
 

   
    # split into train and test sets
train_size = int(len(dataset) * 0.70)
test_size= len(dataset) - train_size
size = train_size+1


train_data_10 = dataset_10[:train_size]
train_data_25 = dataset_25[:train_size]
train_data_50 = dataset_50[:train_size]
train_data_80 = dataset_80[:train_size]
test_data_10  = dataset_10[train_size+1: ]
test_data_25  = dataset_25[train_size+1: ]
test_data_50  = dataset_50[train_size+1: ]
test_data_80  = dataset_80[train_size+1: ]
    
# normalize the dataset
# Feature Scaling
""""sc = MinMaxScaler(feature_range = (0, 1))
X_10 = sc.fit_transform(train_data_10)
X_25 = sc.fit_transform(train_data_25)
X_50 = sc.fit_transform(train_data_50)
X_80 = sc.fit_transform(train_data_80)
Y_10 = sc.fit_transform(test_data_10)
Y_25 = sc.fit_transform(test_data_25)
Y_50 = sc.fit_transform(test_data_50)
Y_80 = sc.fit_transform(test_data_80)

"""





upsampled =raw_dataset['Soil moisture at 10m'].resample('D')
interpolated_10 = upsampled.interpolate(method='spline', order=2)
print(interpolated_10.head(32))
#interpolated_10.plot()

autocorrelation_plot(interpolated_10)

plt.ylabel('AutoCorrelation for the 10 m')
plt.title('ACF for 10 m ')
plt.savefig('ACF for 10 m ')
plt.legend()
plt.show()

plot_acf(interpolated_10)
pyplot.show()
plot_pacf(interpolated_10, lags=20)
plt.ylabel('Partial AutoCorrelation for the 10 m')
plt.title('PACF for 10 m ')
plt.legend()
plt.savefig('PACF for 10 m ')
plt.show()



"""
Downsample = raw_dataset['Soil moisture at 25m'].resample('Q')
quarterly_mean_sales = Downsample.mean()
print(quarterly_mean_sales.head())
quarterly_mean_sales.plot()
pyplot.show()
autocorrelation_plot(quarterly_mean_sales)
plot_acf(interpolated)
pyplot.show()
plot_pacf(quarterly_mean_sales, lags=20)
pyplot.show()
"""

history = [x for x in train_data_10]
predictions = list()
for t in range(len(test_data_10)):
	model = ARIMA(history, order=(1,0,1))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test_data_10[t]
	history.append(obs)
	print('10m ',t ,' predicted=%f, expected=%f' % (yhat, obs))  
testPredict_10  = predictions  


#trainPredict_10 = sc.inverse_transform(Y_10)
#testPredict_10 = sc.inverse_transform(testPredict_10)
error_10 = mean_squared_error(test_data_10 , testPredict_10)
print('Test MSE: %.3f' % error_10)

MAE_10= mean_absolute_error(test_data_10 , testPredict_10)
print('Test MAE: %.3f' % MAE_10)

rmse = sqrt(mean_squared_error(test_data_10 , testPredict_10))
print('Test RMSE: %.3f' % rmse)

# plot
plt.plot(testPredict_10, c = 'red', label = 'Predicted')
plt.plot(test_data_10, c = 'green', label = 'Actual')
plt.xlabel('Time')
plt.ylabel('Moisture Level 10 cm')
plt.title('Forecasted moisture 10 cm')
plt.legend()
plt.savefig('Moisture Level at 10 ARIMA ')
plt.show()






################ Soil Moisture at 25 m #########################################


upsampled =raw_dataset['Soil moisture at 25m'].resample('D')
interpolated_25 = upsampled.interpolate(method='spline', order=2)
print(interpolated_25.head(32))
#interpolated_25.plot()

autocorrelation_plot(interpolated_25)
plt.ylabel('AutoCorrelation for the 25 m')
plt.title('ACF for 25 m ')
plt.savefig('ACF for 25 m ')
plt.legend()
plt.show()

plot_acf(interpolated_25)
pyplot.show()
plot_pacf(interpolated_25, lags=20)
plt.ylabel('Partial AutoCorrelation for the 25 m')
plt.title('PACF for 25 m ')
plt.legend()
plt.savefig('PACF for 25 m ')
plt.show()


""""
Downsample = raw_dataset['Soil moisture at 25m'].resample('Q')
quarterly_mean_sales = resample.mean()
print(quarterly_mean_sales.head())
quarterly_medpan_sales.plot()
pyplot.show()
autocorrelation_plot(interpolated)
plot_acf(interpolated)
pyplot.show()
plot_pacf(interpolated, lags=24)
pyplot.show()
"""




history = [x for x in train_data_25]
predictions = list()
for t in range(len(test_data_25)):
	model = ARIMA(history, order=(1,0,1))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test_data_25[t]
	history.append(obs)
	print(' 25',t ,'/2488 predicted=%f, expected=%f' % (yhat, obs))
testPredict_25  = predictions  


#trainPredict_25 = sc.inverse_transform(Y_25)
#testPredict_25 = sc.inverse_transform(testPredict_25)
error_25 = mean_squared_error(test_data_25 , testPredict_25)
print('Test MSE: %.3f' % error_25)

rmse = sqrt(mean_squared_error(test_data_25 , testPredict_25))
print('Test RMSE: %.3f' % rmse)

MAE_25= mean_absolute_error(test_data_25 , testPredict_25)
print('Test MAE: %.3f' % MAE_25)


plt.plot(testPredict_25, c = 'red', label = 'Predicted')
plt.plot(test_data_25, c = 'green', label = 'Actual')
plt.xlabel('Time')
plt.ylabel('Moisture Level 25 cm')
plt.title('Forecasted moisture 25 cm')
plt.legend()
plt.savefig('Moisture Level at 25 ARIMA ')
plt.show()



################ Soil Moisture at 50 m #########################################




upsampled =raw_dataset['Soil moisture at 50 m'].resample('D')
interpolated_50 = upsampled.interpolate(method='spline', order=2)
print(interpolated_50.head(32))
#interpolated_50.plot()

autocorrelation_plot(interpolated_50)
plt.ylabel('AutoCorrelation for the 50 cm')
plt.title('ACF for 50 m ')
plt.savefig('ACF for 50 m ')
plt.legend()
plt.show()

plot_acf(interpolated_50)
pyplot.show()
plot_pacf(interpolated_50, lags=20)
plt.ylabel('Partial AutoCorrelation for the 50 m')
plt.title('PACF for 50 m ')
plt.legend()
plt.savefig('PACF for 50 m ')
plt.show()

"""
Downsample = raw_dataset['Soil moisture at 50 m'].resample('Q')
quarterly_mean_sales = Downsample.mean()
print(quarterly_mean_sales.head())
quarterly_mean_sales.plot()
pyplot.show()
autocorrelation_plot(interpolated)
plot_acf(interpolated)
pyplot.show()
plot_pacf(interpolated, lags=24)
pyplot.show()
"""




history = [x for x in train_data_50]
predictions = list()
for t in range(len(test_data_50)):
	model = ARIMA(history, order=(2,1,1))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test_data_50[t]
	history.append(obs)
	print('50',t ,' predicted=%f, expected=%f' % (yhat, obs))
testPredict_50  = predictions  


#trainPredict_50 = sc.inverse_transform(Y_50)
#testPredict_50 = sc.inverse_transform(testPredict_50)
error_50 = mean_squared_error(test_data_50 , testPredict_50)
print('Test MSE: %.3f' % error_50)


MAE_50= mean_absolute_error(test_data_50, testPredict_50)
print('Test MAE: %.3f' % MAE_50)


rmse = sqrt(mean_squared_error(test_data_50, testPredict_50))
print('Test RMSE: %.3f' % rmse)



plt.plot(testPredict_50, c = 'red', label = 'Predicted')
plt.plot(test_data_50, c = 'green', label = 'Actual')
plt.xlabel('Time')
plt.ylabel('Moisture Level 50 cm')
plt.title('Forecasted moisture 50 cm')
plt.legend()
plt.savefig('Moisture Level at 50 ARIMA ')
plt.show()


################ Soil Moisture at 80 m #########################################



upsampled =raw_dataset['Soil moisture at 80m'].resample('D')
interpolated_80 = upsampled.interpolate(method='spline', order=2)
print(interpolated_80.head(32))
#interpolated_80.plot()

autocorrelation_plot(interpolated_80)
plt.ylabel('AutoCorrelation for the 80 m')
plt.title('ACF for 80 cm ')
plt.savefig('ACF for 80 m ')
plt.legend()
plt.show()

plot_acf(interpolated_80)
pyplot.show()
plot_pacf(interpolated_80, lags=20)
plt.ylabel('Partial AutoCorrelation for the 80 cm')
plt.title('PACF for 80 cm ')
plt.legend()
plt.savefig('PACF for 80 m ')
plt.show()

"""
Downsample = raw_dataset['Soil moisture at 80m'].resample('Q')
quarterly_mean_sales = Downsample.mean()
print(quarterly_mean_sales.head())
quarterly_mean_sales.plot()
pyplot.show()
autocorrelation_plot(interpolated)
plot_acf(interpolated)
pyplot.show()
plot_pacf(interpolated, lags=24)
pyplot.show()
"""






history = [x for x in train_data_80]
predictions = list()
for t in range(len(test_data_80)):
	model = ARIMA(history, order=(1,0,1))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test_data_80[t]
	history.append(obs)
	print('.80m' ,t ,' predicted=%f, expected=%f' % (yhat, obs))
testPredict_80  = predictions  


#trainPredict_80 = sc.inverse_transform(Y_80)
#testPredict_80 = sc.inverse_transform(testPredict_80)

error_80 = mean_squared_error(test_data_80 , testPredict_80)
print('Test MSE: %.3f' % error_80)

MAE_80= mean_absolute_error(test_data_80, testPredict_80)
print('Test MAE: %.3f' % MAE_80)

from math import sqrt
# calculate RMSE
rmse = sqrt(mean_squared_error(test_data_80, testPredict_80))
print('Test RMSE: %.3f' % rmse)


plt.plot(testPredict_80, c = 'red', label = 'Predicted')
plt.plot(test_data_80 , c = 'green', label = 'Actual')
plt.xlabel('Time')
plt.ylabel('Moisture Level 80 cm')
plt.title('Forecasted moisture 80 cm')
plt.legend()
plt.savefig('Moisture Level at 80 ARIMA ')
plt.show()


###################################


series_index = raw_dataset1['date_time']

index_test = series_index[size:]


#trainPredict_10 , testPredict_10

df_P_10m = pd.DataFrame(testPredict_10,index = index_test, columns=['Predicted 10 m'])
df_A_10m = pd.DataFrame(test_data_10, index = index_test, columns=[' Actual 10 m'])
df_P_25m = pd.DataFrame(testPredict_25, index = index_test, columns=['Predicted 25 m'])
df_A_25m = pd.DataFrame(test_data_25, index = index_test, columns=[' Actual25 m'])
df_P_50m = pd.DataFrame(testPredict_50, index = index_test, columns=['Predicted 50 m'])
df_A_50m = pd.DataFrame(test_data_50,index = index_test, columns=[' Actual 50 m'])
df_P_80m = pd.DataFrame(testPredict_80, index = index_test,columns=['Predicted 80 m'])
df_A_80m = pd.DataFrame(test_data_80,index = index_test, columns=[' Actual 80 m'])


Err_10m = Error(df_A_10m, df_P_10m)
Err_25m = Error(df_A_25m, df_P_25m)
Err_50m = Error(df_A_50m, df_P_50m)
Err_80m = Error(df_A_80m, df_P_80m)

#Err_10m = round(Err_10m,2)

Err_10m = pd.DataFrame(Err_10m, index = index_test, columns=['Error at 10 m '])
Err_25m = pd.DataFrame(Err_25m, index = index_test, columns=['Error at 25 m '])
Err_50m = pd.DataFrame(Err_50m, index = index_test, columns=['Error at 50 m '])
Err_80m = pd.DataFrame(Err_80m, index = index_test, columns=['Error at 80 m '])

df = pd.concat([df_A_10m, df_P_10m, Err_10m, df_A_25m, df_P_25m, Err_25m, df_A_50m, 
                    df_P_50m, Err_50m, df_A_80m, df_P_80m, Err_80m ], axis =1)
df.to_csv("Soil moisture ARIMA.csv")




a_10= mean_absolute_percentage_error(df_A_10m, df_P_10m )
print("\n\n\tMean absolute percentage error  :",a_10)
print(".......Accuracy for 10 m  ...", 100-a_10)


a_25= mean_absolute_percentage_error(df_A_25m, df_P_25m )
print("\n\n \t Mean absolute percentage error  :",a_25)
print(".......Accuracy for 25 m  ...", 100-a_25)


a_50= mean_absolute_percentage_error(df_A_50m, df_P_50m )
print("\n\n \tMean absolute percentage error   :",a_50)
print(".......Accuracy for 50 m  ...", 100-a_50)


a_80= mean_absolute_percentage_error(df_A_80m, df_P_80m )
print("\n\n \tMean absolute percentage error  :",a_80)
print(".......Accuracy for 80 m  ...", 100-=a_80)


