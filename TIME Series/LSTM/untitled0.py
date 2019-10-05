
n_epochs = 50

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt


import os    
import numpy
import matplotlib.pyplot as plt
import pandas
from sklearn.preprocessing import MinMaxScaler


   # convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
   
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = numpy.array(y_true), numpy.array(y_pred)
    return numpy.mean(numpy.abs((y_true - y_pred) / y_true)) * 100


def Error(y_true, y_pred): 
     return ((numpy.array(y_true)- numpy.array(y_pred)))
     

##os.chdir("C:/Users/Harinder Singh/Google Drive/ML")

os.chdir('C:/Users/Raju/Desktop/paper/LSTM paper')


raw_dataset = pandas.read_csv('all in 1 moisture.csv')
raw_dataset = pandas.read_csv('all in 1 moisture.csv', parse_dates=["date_time"], index_col= "date_time")

dataset = raw_dataset.iloc[:,1:5].values
dataset_10 = raw_dataset.iloc[:, 1:2].values
 
   
    # split into train and test sets
train_size = int(len(dataset) * 0.70)
test_size= len(dataset) - train_size



train_data_10 = dataset_10[:train_size]
test_data_10  = dataset_10[train_size+1: ]
    


 
# normalize the dataset
# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))
trainScaled_10 = sc.fit_transform(train_data_10)
testScaled_10 = sc.fit_transform(test_data_10)


#print(len(train), len(tes))
    
    
    # reshape into X=t and Y=t+1
look_back = 24
trainX_10, trainY_10 = create_dataset(trainScaled_10, look_back)
testX_10, testY_10 = create_dataset(testScaled_10, look_back)

# reshape input to be [samples, time steps, features]
trainX_10 = numpy.reshape(trainX_10, (trainX_10.shape[0], 1, 24))
testX_10 = numpy.reshape(testX_10, (testX_10.shape[0], 1, testX_10.shape[1]))


# Recurrent Neural Networks


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

model_10 = Sequential()
model_25 = Sequential()
model_50 = Sequential()
model_80 = Sequential()

######################10 m ######################################

model_10.add(LSTM(units = 40, return_sequences= True, input_shape=(1, look_back)))
model_10.add(Dropout(0.2))

model_10.add(LSTM(units = 10, return_sequences= True))
model_10.add(Dropout(0.2))

model_10.add(LSTM(units = 5))


model_10.add(Dense(1))

#model_10.add(Dense(1))


model_10.compile(loss='mean_squared_error', optimizer='adam')

model_10.fit(trainX_10, trainY_10, epochs= n_epochs, batch_size=10, verbose=2)

print('################ M o d e l  build for 10 m   #################################')

      
      
      
      
trainPredict_10 = model_10.predict(trainX_10)

testPredict_10 = model_10.predict(testX_10)

# invert predictions
trainPredict_10 = sc.inverse_transform(trainPredict_10)

trainY_10 = sc.inverse_transform([trainY_10])

testPredict_10 = sc.inverse_transform(testPredict_10)

testY_10 = sc.inverse_transform([testY_10])

testY_10 = testY_10.T



###################### TO CSV #########################
index_test = raw_dataset.iloc[:test_size-(look_back+2), 0].values

df_P_10m = pandas.DataFrame(testPredict_10,index = index_test, columns=['Predicted 10 m'])
df_A_10m = pandas.DataFrame(testY_10, index = index_test, columns=[' Actual 10 m'])

Err_10m = Error(df_A_10m, df_P_10m)


#Err_10m = round(Err_10m,2)

Err_10m = pandas.DataFrame(Err_10m, index = index_test, columns=['Error at 10 m '])


###################### A C C U R A C Y  #################################
######################  P. A. M. E.  ####################################

a_10= mean_absolute_percentage_error(df_A_10m, df_P_10m )
#print("\n\n\tMean absolute percentage error  :",a_10)
#print(".......Accuracy for 10 m  ...", 100-a_10)

print('Test MAPE: %.3f' % a_10)


error_10 = mean_squared_error(df_A_10m, df_P_10m)
print('Test MSE: %.3f' % error_10)

MAE_10= mean_absolute_error(df_A_10m, df_P_10m)
print('Test MAE: %.3f' % MAE_10)

rmse = sqrt(mean_squared_error(df_A_10m, df_P_10m))
print('Test RMSE: %.3f' % rmse)

