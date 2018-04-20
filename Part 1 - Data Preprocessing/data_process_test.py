#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 22:25:30 2017

@author: shanthakumarp
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#plt.plot([1,2,3,4])
#plt.show()

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


# fill Missing data
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values="NaN", strategy="mean", axis=0) 
imp = imp.fit(X[:, 1:3]) 
X[:, 1:3] = imp.transform(X[:, 1:3])
 
# catergory text in data set
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencode_X = LabelEncoder()
X[:, 0] = labelencode_X.fit_transform(X[:, 0])

# making dummy variables/ Encoding categorial data
onehotencode_x = OneHotEncoder(categorical_features= [0])
X = onehotencode_x.fit_transform(X).toarray()

labelencode_y = LabelEncoder()
y = labelencode_y.fit_transform(y)

# spliting train, test datasets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
std_x = StandardScaler()
X_train = std_x.fit_transform(X_train)
X_test = std_x.transform(X_test)



