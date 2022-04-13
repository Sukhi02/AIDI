# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 04:43:09 2022

@author: hp
"""

import os 
os.chdir("D:/AIDI/1005/Project-1/Project-1")

import pandas as pd
file = 'winequality-red.csv'
data = pd.read_csv(file)

print(data.shape)
print(data.size)
print(type(data))



# Take a look at the first few rows
print(data.head())


# Looking at the ST_NUM column
print((data.isnull()).sum())
print((data.isna()).sum())


# Outlier
data.describe()

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

import numpy as np
from scipy import stats
def is_normal(x, treshhold = 0.05):
    k2,p = stats.normaltest(x)
    print(p)
    print(p > treshhold)
    print('\n')
    return p > treshhold

for name in list(data):
    is_normal(np.array(data))
    









import seaborn as sb
import matplotlib.pyplot as plt


 
# plotting correlation heatmap
dataplot=sb.heatmap(data.corr())
  
# displaying heatmap
plt.show()




















 
 


import matplotlib.pyplot as plt
fig, axes = plt.subplots(6, 2, figsize=(16,9))
axes[0,0].set_title("fixed acidity")
axes[0,0].hist(data['fixed acidity'], bins=20);

axes[0,1].set_title("volatile acidity")
axes[0,1].hist(data['volatile acidity'], bins=20);

axes[1,0].set_title("citric acid")
axes[1,0].hist(data['citric acid'], bins=20);

axes[1,1].set_title("residual sugar")
axes[1,1].hist(data['residual sugar'], bins=20);

axes[2,0].set_title("chlorides")
axes[2,0].hist(data['chlorides'], bins=20);

axes[2,1].set_title("free sulfur dioxide")
axes[2,1].hist(data['free sulfur dioxide'], bins=20);

axes[3,0].set_title("total sulfur dioxide")
axes[3,0].hist(data['total sulfur dioxide'], bins=20);

axes[3,1].set_title("density")
axes[3,1].hist(data['density'], bins=20);

axes[4,0].set_title("pH")
axes[4,0].hist(data['pH'], bins=20);

axes[4,1].set_title("sulphates")
axes[4,1].hist(data['sulphates'], bins=20);

axes[5,0].set_title("alcohol")
axes[5,0].hist(data['alcohol'], bins=20);

axes[5,1].set_title("quality")
axes[5,1].hist(data['quality'], bins=20);










    
    