
import pandas as pd
from pandas import read_csv
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import os
import numpy as np
import matplotlib.pyplot as plt
  
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def Error(y_true, y_pred): 
     return ((np.array(y_true)- np.array(y_pred)))
     
from sklearn.preprocessing import MinMaxScaler





################ Soil Moisture at 10 m #########################################


os.chdir("C:/Users/Raju/Desktop/paper/paper/Ubuntu Mar/ARIMA")

series = read_csv('all in 1 moisture.csv')
train_data_10 = series['Soil moisture at 10m'].values
sc = MinMaxScaler(feature_range = (0, 1))
X_10 = sc.fit_transform(train_data_10)





size = int(len(X_10) * 0.70)

train_10, test_10 = X_10[0:size], X_10[size:len(X_10)]
history = [x for x in train_10]
predictions = list()
for t in range(len(test_10)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test_10[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
    
testPredict_10  = predictions  
    
error_10 = mean_squared_error(test_10, testPredict_10)
print('Test MSE: %.3f' % error_10)


# plot
plt.plot(testPredict_10, c = 'red', label = 'Predicted')
plt.plot(test_10, c = 'green', label = 'Actual')
plt.xlabel('Time')
plt.ylabel('Moisture Level')
plt.title('Forecasted moisture 10 M')
plt.savefig('Moisture Level at 10 ARIMA ')
plt.legend()
plt.show()






################ Soil Moisture at 25 m #########################################

X_25 = series['Soil moisture at 25m'].values


train_25, test_25 = X_25[0:size], X_25[size:len(X_25)]
history = [x for x in train_25]
predictions = list()
for t in range(len(test_25)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test_25[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
    
testPredict_25  = predictions  
    
error_25 = mean_squared_error(test_25, testPredict_25)
print('Test MSE: %.3f' % error_25)

plt.plot(testPredict_25, c = 'red', label = 'Predicted')
plt.plot(test_25, c = 'green', label = 'Actual')
plt.xlabel('Time')
plt.ylabel('Moisture Level 25 M')
plt.title('Forecasted moisture 25 M')
plt.savefig('Moisture Level at 25 ARIMA ')
plt.legend()
plt.show()



################ Soil Moisture at 50 m #########################################

X_50 = series['Soil moisture at 50 m'].values


train_50, test_50 = X_50[0:size], X_50[size:len(X_50)]
history = [x for x in train_50]
predictions = list()
for t in range(len(test_50)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test_50[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
    
testPredict_50  = predictions  
    
error_50 = mean_squared_error(test_50, testPredict_50)
print('Test MSE: %.3f' % error_50)

plt.plot(testPredict_50, c = 'red', label = 'Predicted')
plt.plot(test_50, c = 'green', label = 'Actual')
plt.xlabel('Time')
plt.ylabel('Moisture Level 50 M')
plt.title('Forecasted moisture 50 M')
plt.savefig('Moisture Level at 50 ARIMA ')
plt.legend()
plt.show()


################ Soil Moisture at 80 m #########################################

X_80 = series['Soil moisture at 80m'].values

train_80, test_80 = X_80[0:size], X_80[size:len(X_80)]
history = [x for x in train_80]
predictions = list()
for t in range(len(test_80)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test_80[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
    
testPredict_80  = predictions  
    
error_80 = mean_squared_error(test_80, testPredict_80)
print('Test MSE: %.3f' % error_80)



plt.plot(testPredict_80, c = 'red', label = 'Predicted')
plt.plot(test_80 , c = 'green', label = 'Actual')
plt.xlabel('Time')
plt.ylabel('Moisture Level 80 M')
plt.title('Forecasted moisture 80 M')
plt.savefig('Moisture Level at 80 ARIMA ')
plt.legend()
plt.show()


###################################



series_index = series['date_time']
index_test = series_index[size:]


df_P_10m = pd.DataFrame(testPredict_10,index = index_test, columns=['Predicted 10 m'])
df_A_10m = pd.DataFrame(test_10, index = index_test, columns=[' Actual 10 m'])
df_P_25m = pd.DataFrame(testPredict_25, index = index_test, columns=['Predicted 25 m'])
df_A_25m = pd.DataFrame(test_25, index = index_test, columns=[' Actual25 m'])
df_P_50m = pd.DataFrame(testPredict_50, index = index_test, columns=['Predicted 50 m'])
df_A_50m = pd.DataFrame(test_50,index = index_test, columns=[' Actual 50 m'])
df_P_80m = pd.DataFrame(testPredict_80, index = index_test,columns=['Predicted 80 m'])
df_A_80m = pd.DataFrame(test_80,index = index_test, columns=[' Actual 80 m'])


Err_10m = Error(df_A_80m, df_P_10m)
Err_25m = Error(df_A_80m, df_P_25m)
Err_50m = Error(df_A_80m, df_P_50m)
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
print(".......Accuracy for 80 m  ...", 100-a_80)


