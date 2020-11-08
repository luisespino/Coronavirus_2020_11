from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import numpy as np
import random

#----------------------------------------------------------------------------------------#
# Step 1: training data

X = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
Y = [0,1,2,3,6,6,8,9,13,17,19,20,21,24,25,32,34,36,39,46,47,50,61,70,74,80,87,126,139,153,156]

X = np.asarray(X)
Y = np.asarray(Y)

X = X[:,np.newaxis]
Y = Y[:,np.newaxis]

plt.scatter(X,Y)

#plt.show()



#----------------------------------------------------------------------------------------#
# Step 2: data preparation

nb_degree = 3

polynomial_features = PolynomialFeatures(degree = nb_degree)
#print(X)
X_TRANSF = polynomial_features.fit_transform(X)

#----------------------------------------------------------------------------------------#
# Step 3: define and train a model

model = LinearRegression()

model.fit(X_TRANSF, Y)

#----------------------------------------------------------------------------------------#
# Step 4: calculate bias and variance

Y_NEW = model.predict(X_TRANSF)

rmse = np.sqrt(mean_squared_error(Y,Y_NEW))
r2 = r2_score(Y,Y_NEW)

print('RMSE: ', rmse)
print('R2: ', r2)

#----------------------------------------------------------------------------------------#
# Step 5: prediction

x_new_min = 0.0
x_new_max = 50.0

X_NEW = np.linspace(x_new_min, x_new_max, 50)
X_NEW = X_NEW[:,np.newaxis]

X_NEW_TRANSF = polynomial_features.fit_transform(X_NEW)

Y_NEW = model.predict(X_NEW_TRANSF)

plt.plot(X_NEW, Y_NEW, color='coral', linewidth=3)

plt.grid()
plt.xlim(x_new_min,x_new_max)
plt.ylim(0,1000)

title = 'Degree = {}; RMSE = {}; R2 = {}'.format(nb_degree, round(rmse,2), round(r2,2))

plt.title("Polynomial Linear Regression using scikit-learn and python 3 \n " + title,
          fontsize=10)
plt.xlabel('x')
plt.ylabel('y')

plt.savefig("polynomial_linear_regression.png", bbox_inches='tight')
plt.show()
