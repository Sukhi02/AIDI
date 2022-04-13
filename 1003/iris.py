import pandas as pd
import matplotlib as plt
import os 
path= "D:/AIDI/1003"
os.chdir(path)

data = pd.read_csv('iris.csv')
data.info()
data.shape
data.info()


data.describe(include='all')

data.duplicated().sum() 
# Total no of duplicates in the dataset3

data[data.duplicated()] 
data['variety'].value_counts()
data.isnull().sum(axis=0)

# Perform EDA and Explore the features using histograms
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(16,9))
axes[0,0].set_title("Distribution of Sepal Width")
axes[0,0].hist(data['sepal.width'], bins=20);
axes[0,1].set_title("Distribution of Sepal Length")
axes[0,1].hist(data['sepal.length'], bins=20);
axes[1,0].set_title("Distribution of Petal Width")
axes[1,0].hist(data['petal.width'], bins=20);
axes[1,1].set_title("Distribution of Petal Length")
axes[1,1].hist(data['petal.length'], bins=20);


#Separating dependent and independent values..
X=data.iloc[:, :-1].values
X
y=data.iloc[:, -1].values
y
	
import numpy as np
# Encode the target variable ie convert it to numeric type
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(X,y)



#Experiment using two different ratios of training, validation and test data ie 60-20-20, 80-10-10. On the two different split ratios do the following
import pandas as pd
from sklearn.model_selection import train_test_split

#Implement KFold Cross Validation
# Let's say we want to split the data in 80:10:10 for train:valid:test dataset
train_size=0.8
# In the first step we will split the data in training and remaining dataset
X_train, X_rem, y_train, y_rem = train_test_split(X,y, train_size=0.8)

# Now since we want the valid and test size to be equal (10% each of overall data). 
# we have to define valid_size=0.5 (that is 50% of remaining data)
test_size = 0.5
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)

print("X_train.shape >> "+ str(X_train.shape) + " ||  y_train.shape >> " + str(y_train.shape))
print("X_valid.shape >> "+ str(X_valid.shape) + "  ||  y_valid.shape >>  " + str(y_valid.shape))
print("X_test.shape  >> "+ str(X_test.shape) +   "  ||  y_test.shape  >>  " + str(y_test.shape))














#Implement KFold Cross Validation
# evaluate a logistic regression model using k-fold cross-validation
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
# create dataset
# prepare the cross-validation procedure
cv = KFold(n_splits=10, random_state=1, shuffle=True)
# create model
LR_model = LogisticRegression()
# evaluate model
scores = cross_val_score(LR_model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
LR_model = LR_model.fit(X_train, y_train)
y_predicts = LR_model.predict(X_test)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

#Logistic Regression (Grid Search) Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_predicts)




#Implement Grid Search to find optimal hyperparameters for any 3 algorithms (out of LR, SVM, MLP, RF, Boosting)
# example of grid searching key hyperparametres for logistic regression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# define models and parameters
model = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)

grid_result = grid_search.fit(X_train, y_train)

y_predicts = grid_result.predict(X_test)
# summarize results
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#Analyze the results on Validation set and test set and mention which model performed the best and why?
# New Model Evaluation metrics 
print('Accuracy Score : ' + str(accuracy_score(y_test,y_predicts)))
print("Precision Score : ",precision_score(y_test, y_predicts, average='micro'))
print("Recall Score : ",recall_score(y_test, y_predicts, average='micro'))
print('F1 Score : ' + str(f1_score(y_test,y_predicts, average='micro')))

#Logistic Regression (Grid Search) Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_predicts)


#Compare the performance of models(using precision, recall, accuracy, latency). What was the best proportion or split ratio of data from the set of experiments you conducted?



# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 02:34:01 2022

@author: hp
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import copy

'''
Lightweight script to test many models and find winners:param X_train: training split
:param y_train: training target vector
:param X_test: test split
:param y_test: test target vector
:return: DataFrame of predictions
'''

dfs = []
models = [
      ('LR', LogisticRegression()), 
      ('RF', RandomForestClassifier()),
      ('SVM', SVC()),       ]

results = []
names = []
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
target_names = ['Setosa', 'Versicolor',"Virginca"]

for name, model in models:
    kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=1)
    cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
    cv_clf = copy.deepcopy(cv_results)
    clf = model.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(name)
    print(classification_report(y_test, y_pred, target_names=target_names))
    print(confusion_matrix(y_test, y_pred), target_names)
    results.append(cv_results)
    names.append(name)
    this_df = pd.DataFrame(cv_results)
    this_df['model'] = name
    dfs.append(this_df)
    print()
    final = pd.concat(dfs, ignore_index=True)

    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('IRIS ->> Confusion matrix')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + target_names)
    ax.set_yticklabels([''] + target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    
# summarize results
means = clf.cv_results['mean_test_score']

stds = cv_clf['std_test_score']
params = clf.cv_results['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


    
