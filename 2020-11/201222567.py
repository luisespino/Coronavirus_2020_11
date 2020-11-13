from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as graph
import numpy as np

# --- JAPANS COVID CASE POLYNOMIAL/LINEAR REGRESSION
# --- author KEVIN CRISTOPHER JUVENTINO ALVARADO MALDONADO 201222567
# --- data gathered from  https://ourworldindata.org/covid-cases

# =====================================================================================================================================================
# cases repesents our Y axis
# days represent our X axis
# Confirmed cases from  October 1st to Noviember 10th of 2020
cases = [83010, 84215, 84768, 85339, 85739, 86047, 86543, 87020, 87639, 88233, 88912, 89347, 89673, 90140, 90710, 91431, 92063, 92656, 93127, 93480, 93933,
         94524, 95138, 95835, 96534, 97074, 97498, 98116, 98852, 99622, 100392, 101146, 101813, 102281, 102900, 103838, 104782, 105914, 107086, 108084, 108983]

# October 1st
days = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]

# November 1st                                         10th of November
# This could be done easily with pandas and in a much more efficient manner, but it was done this way
# in order to make the code and logic easier to understand while reading it
# =====================================================================================================================================================

# Conversions needed by Scikit library
cases = np.asarray(cases)
days = np.asarray(days)

cases = cases[:, np.newaxis]
days = days[:, np.newaxis]


# =====================================================================================================================================================
# we stablish the basic configuration of our prediction
poly = PolynomialFeatures(degree=4)

# adjust the days to fit for the X axis
days_poly = poly.fit_transform(days)

# The model is defined and correlates the X and Y axis
model = LinearRegression()
model.fit(days_poly, cases)

# A predition is cast mathematically using a polynomial regression
prediction = model.predict(days_poly)
new_x, new_y = zip(*sorted(zip(days, prediction)))

# the prediction graph is created
graph.figure(figsize=(10, 5))
graph.scatter(
    days,
    cases,
    c='green', marker='x'
)
graph.plot(
    new_x, new_y,
    c='blue'
)
graph.title("Japans confirmed cases vs days from october 1st to november 10th")
graph.xlabel('Day')
graph.ylabel('Confirmed Cases')
graph.show()
