import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset=pd.read_csv('E:\\data science\\subject vedios\\Polynomial-Linear-Regression-master\\Polynomial-Linear-Regression-master\\Position_Salaries.csv')
dataset

x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#Train and test dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_selection=0)


#linear regression cassifier
from sklearn.linear_model import LinearRegression
linear_reg1=LinearRegression()
linear_reg1.fit(x,y)
