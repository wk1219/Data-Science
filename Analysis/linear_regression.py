from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("random-linear-regression/test.csv")
df.head()

X = df["x"]
y = df["y"]
line_fitter = LinearRegression()
line_fitter.fit(X.values.reshape(-1,1), y)
plt.plot(X, y, 'o')
plt.plot(X, line_fitter.predict(X.values.reshape(-1,1)))
plt.show()