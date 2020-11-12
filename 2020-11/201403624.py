import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd  # To read data

data = pd.read_csv('201403624_data.csv')  # load data set
X = data.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
Y = data.iloc[:, 2].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()



X = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209]
Y = [0,1,1,2,6,6,9,12,17,19,20,21,24,25,28,34,34,36,38,39,47,50,61,61,70,77,87,95,126,137,155,156,167,180,196,214,235,257,289,294,316,384,430,473,500,530,530,557,599,644,688,703,730,763,798,832,900,967,1052,1114,1199,1342,1518,1643,1763,1763,1912,2133,2265,2512,2743,3054,3424,3760,3954,4145,4348,4607,4739,5087,5336,5586,5760,6154,6485,6792,7055,7502,7866,8221,8561,8982,9491,9845,10272,10706,11251,11868,12509,12755,13145,13769,14540,14819,15619,15828,16397,16930,17409,18096,19011,20072,21293,22501,23248,23972,24787,25411,26658,27619,28598,29355,29742,30872,32074,32939,33809,38042,38677,39039,40229,41135,42192,43283,44492,45053,45309,46451,47605,48826,49789,50979,51306,51542,52365,53509,54339,55270,56189,56605,56987,57966,59089,60284,61428,62313,62562,62944,63847,64881,65983,66941,67856,68188,68533,69651,70714,71856,72921,73679,73912,74074,74893,75644,76358,77040,77481,77683,77828,78721,79622,80306,81009,81658,81909,82172,82684,82924,83664,84344,85152,85444,85681,86623,87442,87933,88878,89702,90092,90263,90968,91746,92409,93090,93748,93963,94182,94870,95704,96480]

X = np.asarray(X)
Y = np.asarray(Y)

X = X[:,np.newaxis]
Y = Y[:,np.newaxis]


#prediccion
nb_degree = 5
polynomial_features = PolynomialFeatures(degree = nb_degree) 
X_TRANSF = polynomial_features.fit_transform(X)  

model = LinearRegression() 
model.fit(X_TRANSF, Y)  

Y_NEW = model.predict(X_TRANSF)  
rmse = np.sqrt(mean_squared_error(Y,Y_NEW)) 
r2 = r2_score(Y,Y_NEW)

x_new_min = 0.0 
x_new_max = 500.0

X_NEW = np.linspace(x_new_min, x_new_max, 50) 
X_NEW = X_NEW[:,np.newaxis]  

X_NEW_TRANSF = polynomial_features.fit_transform(X_NEW)  
Y_NEW = model.predict(X_NEW_TRANSF)  

#plt.plot(X_NEW, Y_NEW, color='coral', linewidth=3)  

plt.grid()  
plt.xlim(x_new_min,x_new_max)  
plt.ylim(0,150000)  

title = 'Degree= {}; RMSE={}; R2={}'.format(nb_degree, round(rmse,2), round(r2,2))
plt.title("Polynomial Linear Regression using scikit-learn and python3 \n" + title, fontsize=10)

plt.xlabel('dias')
plt.ylabel('total contagios')
plt.savefig("polynomial_linear_regresion.png", bbox_inches='tight')


kmeans = KMeans(n_clusters=3)


kmeans.fit(X_NEW)
#print("Clusters:\n ",kmeans.cluster_centers_)
plt.scatter(X_NEW,Y_NEW, c=kmeans.labels_, cmap='rainbow')
valores_prueba=np.array([[2500],[2000],[14000]])
#print("Predict:\n ",kmeans.predict(valores_prueba))
plt.scatter(valores_prueba[:,0],[210,211,212], color='Black')
#plt.scatter(X_NEW,Y_NEW, color='Black')

#plt.show()

