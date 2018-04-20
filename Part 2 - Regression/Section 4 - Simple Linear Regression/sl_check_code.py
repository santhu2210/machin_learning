#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 19:48:57 2017

@author: shanthakumarp
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values  # exlude last column only
y = dataset.iloc[:, 1].values

# split dataset into train, test sets
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.33, random_state=0)

# add feature scalling if you need but all model having it build-in

# adding simple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#test to predict data
y_pred = regressor.predict(X_test)

#y2_test = np.array([[6]])
#y2_pred = regressor.predict(y2_test)

# visualing test set result
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary Vs Experience (Testing set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#visualing training set result
#plt.scatter(X_train, y_train, color='red')
#plt.plot(X_train, regressor.predict(X_train), color='blue')
#plt.title('Salary Vs Experience (Training set)')
#plt.xlabel('Years of Experience')
#plt.ylabel('Salary')
#plt.show()

regressor.score(X_train, y_train)
coef = regressor.coef_          # a
regressor.intercept_     # b


from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, y_pred))


