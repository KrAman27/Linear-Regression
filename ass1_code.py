# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:39:06 2020

@author: aman kumar
"""

"""The dataset contains a walkatime data of past students and how they performed in
the evaluation exam. The task is to predict the score you will get given the amount of time you
spend on coding daily.
Input : You are given one feature corresponding to time noted by walkatime.
Output : A scalar denoting the level of performance student achieved by devoting the given time.
TASK : Build a Linear Regression model on the dataset"""


#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
X_train = pd.read_csv('Linear_X_Train.csv')
X_test = pd.read_csv('Linear_X_Test.csv')
y_train = pd.read_csv('Linear_Y_Train.csv')

#fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting the test set results
y_pred = regressor.predict(X_test)

#visualising the training set results
plt.scatter(X_train,y_train,color = 'red')
plt.scatter(X_test,y_pred,color = 'black')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('y_prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.show()


