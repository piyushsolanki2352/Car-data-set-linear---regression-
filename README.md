# Car-data-set-(linear-regression)-

import pandas as pd
path = '/content/Customer Purchasing Behaviors.xlsx'
df = pd.read_excel(path)
print(df.head)

x = df["annual_income"].values
y = df["purchase_amount"].values

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg = reg.fit(x, y)
y_pred = reg.predict(x)
print(y_pred)

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y,y_pred))
print(rmse)

from sklearn.metrics import r2_score
r2 = r2_score(y,y_pred)
print(r2)

import matplotlib.pyplot as plt
m = len(x)
x = x.reshape((m, 1))
