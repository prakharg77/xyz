import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import random
data = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\python_folder\Top10VideoGameStocks.csv")
print(data.describe())
print(data.info())
dummies = pd.get_dummies(data, columns=['Ticker Symbol'], dtype='int64')
if 'Date' in dummies.columns:
    data1 = dummies.drop(columns=['Date'])
else:
    data1 = dummies
print(data1)
print(data1.columns)
X = data1[['Open']]
Y = data1['Volume']
random.seed(1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)
knn = KNeighborsRegressor(n_neighbors=7)
knn.fit(X_train, Y_train)
r2_score = knn.score(X_test, Y_test)
print("R2 Score: {:.2f}".format(r2_score))