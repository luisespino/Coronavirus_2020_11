import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd  

data = pd.read_csv('201403624_data.csv')  
X = data.iloc[:, 0].values.reshape(-1, 1)  
Y = data.iloc[:, 2].values.reshape(-1, 1)  
linear_regressor = LinearRegression()  
linear_regressor.fit(X, Y)  
Y_pred = linear_regressor.predict(X)  

x_new_min = 0.0 
x_new_max = 250.0
plt.xlim(x_new_min,x_new_max)  
plt.ylim(0,100)  

title = 'Number of deaths in Guatemala\n'+'Trained Model : Y = ' + str(linear_regressor.coef_[0][0]) + 'X+' + str(linear_regressor.intercept_[0])
plt.title("Polynomial Linear Regression using scikit-learn and python3 \n" + title, fontsize=10)
plt.xlabel('Days')
plt.ylabel('Total of deaths')
plt.scatter(X, Y)
plt.plot(X, Y_pred, color='cyan')
plt.legend(('Linear Regression','Data'), loc='upper right')
plt.savefig("201403624_img1.png", bbox_inches='tight')

plt.show()


