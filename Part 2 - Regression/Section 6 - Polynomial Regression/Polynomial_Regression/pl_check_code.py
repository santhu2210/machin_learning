#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 19:29:36 2017

@author: shanthakumarp
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# no to split X_train, y_train, we directly use X, y as datasets
# create a liner regression model for comparison
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# create a polynomial regressin model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

#visualing Linear regression model
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Linear Regression model')
plt.xlabel('Level of job ')
plt.ylabel('Salary')
plt.show()

#visualing Polynominal regression model
plt.scatter(X, y , color='red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color='blue')
#plt.plot(X, lin_reg2.predict(X_poly), color='blue')
plt.title('Poly Linear Regression')
plt.xlabel('Level of job')
plt.ylabel('Salary')
plt.show()

# visualaize more accurate line/curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y , color='red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Poly Linear Regression')
plt.xlabel('Level of job')
plt.ylabel('Salary')
plt.show()

# predicting salary using linear regression
y_pred_li = lin_reg.predict(7.5)

# predicting salary using polynomial regression
y_pred_ply  = lin_reg2.predict(poly_reg.fit_transform(7.5))




