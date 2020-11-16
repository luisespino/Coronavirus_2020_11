import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

datos = pd.read_csv('../data/201213336.csv', sep=';')

X = datos['dia'].values.reshape(-1, 1)
y = datos['confirmados'].values.reshape(-1, 1)
z = datos['muertos'].values.reshape(-1,1)

model = LinearRegression()
model.fit(X=X, y=y)

predict = [365]
predict = np.array(predict).reshape(-1,1) 

prediccionConfirmado = model.predict(predict)

model.fit(X=X, y=z)
prediccionesMuerto = model.predict(predict)

print(prediccionConfirmado) #255,556
print(prediccionesMuerto) #8844
