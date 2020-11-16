# @author Andrés Ricardo Ismael Guzmán
# 201010425

import numpy
import matplotlib.pyplot as plot
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# convertir un arreglo en una matriz de la forma
# x = t; y = t + 1
# es decir si tenemos una lista de variables x, x1, x2, x3...
# la matriz queda de la siguientes manera [x,x1], [x1,x2], [x2,x3]...
def createDS(ds, look_back=1):
	dX, dY = [], []
	for i in range(len(ds)-look_back-1):
		a = ds[i:(i+look_back), 0]
		dX.append(a)
		dY.append(ds[i + look_back, 0])
	return numpy.array(dX), numpy.array(dY)


def run():
    #dejar fijo este valor para poder reproducir los valores
    numpy.random.seed(7)

    # cargar el ds
    # como los datos de fechas no nos importan sólo tomamos la columna 2
    # si se desea hacer un fit de las muertes tomar columna 3
    df = read_csv('../data/201010425.csv', usecols=[2], engine='python')
    ds = df.values
    ds = ds.astype('float32')

    # normalizar el ds
    # se usa MinMax porque LSTM es sensible a la escala
    scaler = MinMaxScaler(feature_range=(0, 1))
    ds = scaler.fit_transform(ds)

    # dividir el arreglo para entrenarlo y para reforzarlo
    train_set_size = int(len(ds) * 0.66)
    train, test = ds[0:train_set_size,:], ds[train_set_size:len(ds),:]
    
    # convertir la data en X=t, Y=t+1
    look_back = 1
    train_set_x, train_set_y = createDS(train, look_back)
    test_set_x, test_set_y = createDS(test, look_back)
    
    # Cambiar la entrada para ser [samples, time steps, features]
    train_set_x = numpy.reshape(train_set_x, (train_set_x.shape[0], 1, train_set_x.shape[1]))
    test_set_x = numpy.reshape(test_set_x, (test_set_x.shape[0], 1, test_set_x.shape[1]))
    print(test_set_x)
    # crear y modelar la red LSTM 
    # se setea el número de epochs de 100
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_set_x, train_set_y, epochs=100, batch_size=1, verbose=2)
    
    # predicciones
    train_prediction_model = model.predict(train_set_x)
    test_prediction_model = model.predict(test_set_x)
    
    # invertir las predicciones
    train_prediction_model = scaler.inverse_transform(train_prediction_model)
    train_set_y = scaler.inverse_transform([train_set_y])
    test_prediction_model = scaler.inverse_transform(test_prediction_model)
    test_set_y = scaler.inverse_transform([test_set_y])
    # calcular error
    trainScore = math.sqrt(mean_squared_error(train_set_y[0], train_prediction_model[:,0]))
    print('Score del entrenamiento: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(test_set_y[0], test_prediction_model[:,0]))
    print('Score de la prueba: %.2f RMSE' % (testScore))
    
    # cambiar las predicciones para poderlas graficar
    train_prediction_plot = numpy.empty_like(ds)
    train_prediction_plot[:, :] = numpy.nan
    train_prediction_plot[look_back:len(train_prediction_model)+look_back, :] = train_prediction_model
    # cambiar las predicciones de prueba para graficar
    test_prediction_plot = numpy.empty_like(ds)
    test_prediction_plot[:, :] = numpy.nan
    test_prediction_plot[len(train_prediction_model)+(look_back*2)+1:len(ds)-1, :] = test_prediction_model
    # plot baseline and predictions
    plot.plot(scaler.inverse_transform(ds))
    plot.plot(train_prediction_plot)
    plot.plot(test_prediction_plot)
    plot.show()
    
    model.save('lstm_model.h5')

def makeprediction():
    model = load_model('lstm_model.h5')
    # el arreglo 
    #Xnew = array([[0.29466096, 0.30317302]])
    #yhat = model.predict(Xnew), verbose=0)
    #print(yhat)

#correr
run()
#makeprediction()
