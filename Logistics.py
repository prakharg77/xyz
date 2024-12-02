import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\python_folder\Top10VideoGameStocks.csv")
print(data.describe())
print(data.info())
print(data.head())

dummies = pd.get_dummies(data, columns=['Ticker Symbol'], dtype='int64')
print(dummies)
data1 = dummies.drop(columns=['Date'])
print(data1.columns)

X = data1[['Open', 'High', 'Low', 'Close', 'Adj Close']]
y = data1['Volume']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R^2 Score: {r2:.2f}")
