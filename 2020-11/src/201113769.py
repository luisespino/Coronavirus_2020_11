
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('../data/201113769.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_t, X_test, y_t, y_test = train_test_split(X, y, test_size = 0.2)


reg = LinearRegression()
reg.fit(X_t, y_t)

y_pred = reg.predict(X_test)

df = pd.DataFrame({'Real Values':y_test, 'Predicted Values':y_pred})
df

plt.scatter(X_test, y_test, color = 'red')
plt.scatter(X_test, y_pred, color = 'blue')
plt.plot(X_t, reg.predict(X_t), color = 'black')
plt.title('Casos confirmados vs Resultados entregados')
plt.xlabel('Resultados entregados')
plt.ylabel('Casos confirmados')
plt.show()

