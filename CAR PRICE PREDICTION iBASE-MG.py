# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 14:54:38 2023

@author: Mohit Gedam
"""

#Importing Libraries
import pandas as pd

#Import train_test_split
from sklearn.model_selection import train_test_split

#Import Linear Regression 
from sklearn.linear_model import LinearRegression

# Importing numpy as np
import numpy as np

# Importing the dataset from the local storage
df = pd.read_csv("D:/CAR PRICE PREDICTION.csv")

#features is the independent varible
#target is the dependent varible
x=df[['Year', 'Average']]

y=df['Price']

# Splitting the dataset into training and testing datasets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)

# Making model as an object of LinearRegression
model=LinearRegression()

# Fitting the Model with the training datasets
model.fit(x_train.values, y_train.values)

# We are predicting the output on the basis of the value which is passed in the predict function 
y_pred=model.predict(x_test)

#predicting the output which is passed to the predict function
Year=2015
Average=20
price=model.predict([[Year, Average]])
print("PREDICTED PRICE FOR YEAR",Year, "& AVERAGE", Average,  "IS", price)