#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 03:45:23 2020

@author: ayman
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

# Importing the dataset
dataset = pd.read_csv('covid19_symptoms.csv')
X = dataset.iloc[:, 1:-1].values
dfx = pd.DataFrame(X)
y = dataset.iloc[:, 7].values.reshape(-1, 1)
dfy = pd.DataFrame(y)


# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder

# Replace categorical data by matrix
onehotencoder = OneHotEncoder(categories = 'auto')
y = onehotencoder.fit_transform(y).toarray()

    
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.4, random_state = 0)

# Fitting the MLR model to the training set:  
from sklearn.linear_model import LinearRegression  
regressor= LinearRegression()  
regressor.fit(train_x, train_y)  

# Predicting the Test set result;  
y_pred= regressor.predict(test_x)

print('Train Score: ', regressor.score(train_x, train_y))  
print('Test Score: ', regressor.score(test_x, test_y))

# Plot graph for taring values
plt.scatter(X, y, color = 'red')
plt.plot(train_x, regressor.predict(train_x), color = 'blue')
plt.title('Detecting covid 19 cases based on symptoms')
plt.xlabel('Symptoms of covid 19')
plt.ylabel('Case Condition')
plt.show()

# Plot graph for test values
plt.scatter(test_x, test_y, color = 'red')
plt.plot(train_x, regressor.predict(train_x), color = 'blue')
plt.title('Detecting covid 19 cases based on symptoms')
plt.xlabel('Symptoms of covid 19')
plt.ylabel('Case Condition')
plt.show()