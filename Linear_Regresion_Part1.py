import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing,svm
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import random
data = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\python_folder\Top10VideoGameStocks.csv")
print(data.describe())
print(data.info())
dummies = pd.get_dummies(data,columns=['Ticker Symbol'],dtype='int64')
print(dummies)
data1 = dummies.drop(columns=['Date'])
print(data1)
print(data1.columns)
X = data1 [['Open']]
Y = data1 ['Volume']
print(X)
print(Y)
sns.scatterplot(data1)
plt.plot(X,Y)
random.seed(1)
X_train, X_test , Y_train, Y_test = train_test_split(X, Y, test_size= .30)
regr = LinearRegression()
regr.fit(X_train, Y_train)
print(regr.score(X_test, Y_test))