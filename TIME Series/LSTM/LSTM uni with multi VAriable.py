#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 09:54:47 2019

@author: sukhee
"""



n_epochs = 100

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
dataset = raw_dataset.iloc[:,1:5].values
dataset_10 = raw_dataset.iloc[:, 1:2].values
dataset_25 = raw_dataset.iloc[:, 2:3].values
dataset_50 = raw_dataset.iloc[:, 3:4].values
dataset_80 = raw_dataset.iloc[:, 4:5].values
 
   
    # split into train and test sets
train_size = int(len(dataset) * 0.70)
test_size= len(dataset) - train_size



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
sc = MinMaxScaler(feature_range = (0, 1))
trainScaled_10 = sc.fit_transform(train_data_10)
trainScaled_25 = sc.fit_transform(train_data_25)
trainScaled_50 = sc.fit_transform(train_data_50)
trainScaled_80 = sc.fit_transform(train_data_80)
testScaled_10 = sc.fit_transform(test_data_10)
testScaled_25 = sc.fit_transform(test_data_25)
testScaled_50 = sc.fit_transform(test_data_50)
testScaled_80 = sc.fit_transform(test_data_80)


#print(len(train), len(tes))
    
    
    # reshape into X=t and Y=t+1
look_back = 24
trainX_10, trainY_10 = create_dataset(trainScaled_10, look_back)
trainX_25, trainY_25 = create_dataset(trainScaled_25, look_back)
trainX_50, trainY_50 = create_dataset(trainScaled_50, look_back)
trainX_80, trainY_80 = create_dataset(trainScaled_80, look_back)
testX_10, testY_10 = create_dataset(testScaled_10, look_back)
testX_25, testY_25 = create_dataset(testScaled_25, look_back)
testX_50, testY_50 = create_dataset(testScaled_50, look_back)
testX_80, testY_80 = create_dataset(testScaled_80, look_back)

# reshape input to be [samples, time steps, features]
trainX_10 = numpy.reshape(trainX_10, (trainX_10.shape[0], 1, 24))
trainX_25 = numpy.reshape(trainX_25, (trainX_10.shape[0], 1, 24))
trainX_50 = numpy.reshape(trainX_50, (trainX_10.shape[0], 1, 24))
trainX_80 = numpy.reshape(trainX_80, (trainX_10.shape[0], 1, 24))
testX_10 = numpy.reshape(testX_10, (testX_10.shape[0], 1, testX_10.shape[1]))
testX_25 = numpy.reshape(testX_25, (testX_25.shape[0], 1, testX_25.shape[1]))
testX_50 = numpy.reshape(testX_50, (testX_50.shape[0], 1, testX_50.shape[1]))
testX_80 = numpy.reshape(testX_80, (testX_80.shape[0], 1, testX_80.shape[1]))


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

model_10.add(LSTM(units = 50, return_sequences= True,  input_shape=(1, look_back)))
model_10.add(Dropout(0.2))

model_10.add(LSTM(units = 25, return_sequences= True))
model_10.add(Dropout(0.2))

model_10.add(LSTM(units = 10, return_sequences= False))


model_10.add(Dense(1))

model_10.compile(loss='mean_squared_error', optimizer='adam')

model_10.fit(trainX_10, trainY_10, epochs= n_epochs, batch_size=25, verbose=2)

print('################ M o d e l  build for 10 m   #################################')

######################### 25 m ####################################


model_25.add(LSTM(units = 50, input_shape=(1, look_back)))
model_25.add(Dropout(0.2))

model_25.add(Dense(1))

model_25.add(Dense(1))

model_25.compile(loss='mean_squared_error', optimizer='adam')

model_25.fit(trainX_25, trainY_25, epochs=n_epochs, batch_size=25, verbose=2)


print('################ M o d e l  build for 25 m   #################################')


######################  50 m  ##########################################

model_50.add(LSTM(units = 50, input_shape=(1, look_back)))
model_50.add(Dropout(0.2))

model_50.add(Dense(2))

#model_50.add(Dense(1))

model_50.compile(loss='mean_squared_error', optimizer='adam')

model_50.fit(trainX_50, trainY_50, epochs=n_epochs, batch_size=25, verbose=2)


print('################ M o d e l  build for 50 m   #################################')

######################  80 m  ##########################################

model_80.add(LSTM(units = 50, input_shape=(1, look_back)))
model_80.add(Dropout(0.2))

model_80.add(Dense(1))

model_80.add(Dense(1))

model_80.compile(loss='mean_squared_error', optimizer='adam')

model_80.fit(trainX_80, trainY_80, epochs=n_epochs, batch_size=25, verbose=2)


print('################ M o d e l  build for 80 m   #################################')


################ P R E D I C T I O N S  #################################
print('################ P R E D I C T I O N S  #################################')
      
trainPredict_10 = model_10.predict(trainX_10)
trainPredict_25 = model_25.predict(trainX_25)
trainPredict_50 = model_10.predict(trainX_50)
trainPredict_80 = model_10.predict(trainX_80)

testPredict_10 = model_10.predict(testX_10)
testPredict_25 = model_25.predict(testX_25)
testPredict_50 = model_10.predict(testX_50)
testPredict_80 = model_10.predict(testX_80)


# invert predictions
trainPredict_10 = sc.inverse_transform(trainPredict_10)
trainPredict_25 = sc.inverse_transform(trainPredict_25)
trainPredict_50 = sc.inverse_transform(trainPredict_50)
trainPredict_80 = sc.inverse_transform(trainPredict_80)

trainY_10 = sc.inverse_transform([trainY_10])
trainY_25 = sc.inverse_transform([trainY_25])
trainY_50 = sc.inverse_transform([trainY_50])
trainY_80 = sc.inverse_transform([trainY_80])





testPredict_10 = sc.inverse_transform(testPredict_10)
testPredict_25 = sc.inverse_transform(testPredict_25)
testPredict_50 = sc.inverse_transform(testPredict_50)
testPredict_80 = sc.inverse_transform(testPredict_80)

testY_10 = sc.inverse_transform([testY_10])
testY_25 = sc.inverse_transform([testY_25])
testY_50 = sc.inverse_transform([testY_50])
testY_80 = sc.inverse_transform([testY_80])

testY_10 = testY_10.T
testY_25 = testY_25.T
testY_50 = testY_50.T
testY_80 = testY_80.T



###################### TO CSV #########################
index_test = raw_dataset.iloc[:test_size-(look_back+2), 0].values

df_P_10m = pandas.DataFrame(testPredict_10,index = index_test, columns=['Predicted 10 m'])
df_A_10m = pandas.DataFrame(testY_10, index = index_test, columns=[' Actual 10 m'])
df_P_25m = pandas.DataFrame(testPredict_25, index = index_test, columns=['Predicted 25 m'])
df_A_25m = pandas.DataFrame(testY_25, index = index_test, columns=[' Actual25 m'])
df_P_50m = pandas.DataFrame(testPredict_50, index = index_test, columns=['Predicted 50 m'])
df_A_50m = pandas.DataFrame(testY_50,index = index_test, columns=[' Actual 50 m'])
df_P_80m = pandas.DataFrame(testPredict_80, index = index_test,columns=['Predicted 80 m'])
df_A_80m = pandas.DataFrame(testY_80,index = index_test, columns=[' Actual 80 m'])


Err_10m = Error(df_A_10m, df_P_10m)
Err_25m = Error(df_A_25m, df_P_25m)
Err_50m = Error(df_A_50m, df_P_50m)
Err_80m = Error(df_A_80m, df_P_80m)

#Err_10m = round(Err_10m,2)

Err_10m = pandas.DataFrame(Err_10m, index = index_test, columns=['Error at 10 m '])
Err_25m = pandas.DataFrame(Err_25m, index = index_test, columns=['Error at 25 m '])
Err_50m = pandas.DataFrame(Err_50m, index = index_test, columns=['Error at 50 m '])
Err_80m = pandas.DataFrame(Err_80m, index = index_test, columns=['Error at 80 m '])

df = pandas.concat([df_A_10m, df_P_10m, Err_10m, df_A_25m, df_P_25m, Err_25m, df_A_50m, 
                    df_P_50m, Err_50m, df_A_80m, df_P_80m, Err_80m ], axis =1)
df.to_csv("Soil moisture LSTM.csv")


####################### P L O T     ##########################
plt.plot(testPredict_10, c = 'red', label = 'Predicted')
plt.plot(testY_10, c = 'green', label = 'Actual')
plt.xlabel('Time')
plt.ylabel('Moisture Level')
plt.title('Forecasted moisture 10 M')
plt.savefig('Moisture Level at 10 VLSTM ')
plt.legend()
plt.show()

plt.plot(testPredict_25, c = 'red', label = 'Predicted')
plt.plot(testY_25, c = 'green', label = 'Actual')
plt.xlabel('Time')
plt.ylabel('Moisture Level 25 M')
plt.title('Forecasted moisture 25 M')
plt.savefig('Moisture Level at 25 VLSTM ')
plt.legend()
plt.show()


plt.plot(df_P_50m, c = 'red', label = 'Predicted')
plt.plot(df_A_50m, c = 'green', label = 'Actual')
plt.xlabel('Time')
plt.ylabel('Moisture Level 50 M')
plt.title('Forecasted moisture 50 M')
plt.savefig('Moisture Level at 50 VLSTM ')
plt.legend()
plt.show()


plt.plot(testPredict_80, c = 'red', label = 'Predicted')
plt.plot(testY_80, c = 'green', label = 'Actual')
plt.xlabel('Time')
plt.ylabel('Moisture Level 80 M')
plt.title('Forecasted moisture 80 M')
plt.savefig('Moisture Level at 80 VLSTM ')
plt.legend()
plt.show()


plt.plot(dataset, c = 'blue')
plt.show()


###################### A C C U R A C Y  #################################
######################  P. A. M. E.  ####################################

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


