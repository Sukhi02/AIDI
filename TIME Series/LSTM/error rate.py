# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 19:36:01 2019

@author: Raju
"""


a = [0.16,0.44,0.19,0.35]
b = [0.11,0.32,0.12,0.33]

dev_a = pd.DataFrame(a)

index = ['MAE','MAPE','MSE', 'RMSE']

dev_a.index = index

dev_b.index = index

dev_b = pd.DataFrame(b)

plt.plot(dev_a, c = 'red', label = 'ARIMA')
plt.plot(dev_b, c = 'green', label = 'LSTM')
plt.xlabel('performance metrics')
plt.ylabel('Error rate')
plt.title('Comparison of the models (ARIMA vs LSTM)')
plt.legend()

plt.savefig('Error rate')
plt.show()
