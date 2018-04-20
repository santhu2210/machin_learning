#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:20:40 2017

@author: shanthakumarp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Encoding Categorial data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
X[:, 3] = labelencoder_x.fit_transform(X[:, 3])

onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoid Dummy variable
X = X[:, 1:]

#spliting training and testing data
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0, test_size=0.20)

# Fitting multiple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predict test set using training regressor model
y_pred = regressor.predict(X_test)


# visualizing training dataset plot
#plt.scatter(X_train, y_train, color='red')
#plt.plot(X_train, regressor.predict(X_train), color='blue')


# Make an optimal model using backward elimination
import statsmodels.formula.api as sm
#X_try = np.append(arr = np.ones((50,1)).astype(int), values=X, axis=0)
X = np.append(arr = np.ones((50,1)).astype(int), values=X, axis=1)
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# process to reduce column those which are highest P value then siginifance level(0.05)
# refer regressor_summary P>\t\ column
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#  fit, predict dataset after  backward elemination
X_train_opt,X_test_opt,y_train_opt,y_test_opt = train_test_split(X_opt,y,random_state=0, test_size=0.20)
regressor.fit(X_train_opt, y_train_opt)

y_pred_opt = regressor.predict(X_test_opt)
# after backward elimination model predict accuracy much better

#z = np.append([56,3,324,5,56,56]).astype(float)
#y_test_opt2D = y_test.reshape(-1,1).astype(int)
#y_pred_opt2D = y_pred_opt.reshape(-1,1).astype(int)
#y_pred_2D = y_pred.reshape(-1,1)

#regressor.score(X_train, y_train)
#coef = regressor.coef_          # a
#regressor.intercept_     # b
#
#regressor.score(X_test, y_test)

from sklearn import metrics
from math import sqrt

root_mean_sqr_err = sqrt(metrics.mean_squared_error(y_test, y_pred))
mean_abs_err = metrics.mean_absolute_error(y_test, y_pred)
mean_sqr_err = metrics.mean_squared_error(y_test, y_pred)


