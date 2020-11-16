# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 15:26:13 2020

@author: Pavel
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
data = pd.read_csv('201503611.csv')

# Grouping data
X = data.drop(columns=['casos', 'nuevos_casos','muertes','nuevas_muertes']) # For all days will be x-axis
y_cases = data['casos']
y_deaths = data['muertes']
days = [320, 325, 330, 335, 340, 345, 360, 365,370,375,380,385,390,395,400,405,410]
future_days = pd.DataFrame(days)
# Training data
X_train, X_test, y_cases_train, y_cases_test = train_test_split(X, y_cases,shuffle=False)
_, _, y_deaths_train, y_deaths_test = train_test_split(X, y_deaths,shuffle=False)

mlp = MLPRegressor(
            hidden_layer_sizes=(100,100,100),
            solver='lbfgs',
            activation='relu',
            max_iter=70000).fit(X,y_cases)

mlp2 = MLPRegressor( 
            hidden_layer_sizes=(100,100,100),
            solver='lbfgs',
            activation='relu',
            max_iter=70000).fit(X,y_deaths)

training_cases = mlp.predict(X)
training_deaths = mlp2.predict(X)

predict_cases = mlp.predict(future_days)
predict_deaths = mlp2.predict(future_days)

fig = plt.figure(figsize=(10, 10))
plot1 = fig.add_subplot(111)
plot1.set_title('Casos')
plot1.set_ylabel('Casos')
plot1.set_xlabel('Dia')
plot1.plot(X, y_cases)
plot1.plot(X, training_cases)
plot1.plot(days, predict_cases)
plt.show()


fig3 = plt.figure()
plot3 = fig3.add_subplot(111)
plot3.set_title('Muertes')
plot3.set_ylabel('Muertos')
plot3.set_xlabel('Dia')
plot3.plot(X, y_deaths)
plot3.plot(X, training_deaths)
plot3.plot(days, predict_deaths)
plt.show()



