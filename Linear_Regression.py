# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 22:36:51 2019

@author: rajkumar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset=pd.read_csv('E:\\data science\\subject vedios\\simple-Linear-Regression-master\\simple-Linear-Regression-master\\Salary_Data.csv')
dataset

x=dataset.iloc[:,:-1]
x
y=dataset.iloc[:,1]
y

#train and test

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)


#regression classifier

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

y_prid=lr.predict(x_test)
y_prid


#visualising the training dataset
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,lr.predict(x_train),color='blue')

