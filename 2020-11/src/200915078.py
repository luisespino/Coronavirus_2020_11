
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


y = np.array([[4],[5],[6],[7],[8],[9],[10],[11]])
X = np.array([[0],[0],[8],[10],[11],[4],[7],[1]])

X_tn, X_ts, y_tn, y_ts = train_test_split(X, y, test_size = 0.4)

regressor = LinearRegression()
regressor.fit(X_tn, y_tn)

y_prediction = regressor.predict(X_ts)



pl.scatter(X_ts, y_ts, color = 'orange')
pl.scatter(X_ts, y_prediction, color = 'green')
pl.plot(X_tn, regressor.predict(X_tn), color = 'grey')
pl.xlabel('Months')
pl.ylabel('Decline')
pl.show()

