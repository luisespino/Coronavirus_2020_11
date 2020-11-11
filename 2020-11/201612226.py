from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt
import numpy as np
import random

#===============================================================================================#
# Number of cases per day of covid 19 in the US for 218 days

cases = [
    1,0,1,0,3,0,0,0,0,2,1,0,3,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,2,0,0,0,0,0,1,0,8,6,23,25,
    20,66,47,64,147,225,290,278,414,267,338,1237,755,2797,3419,4777,3528,5836,8821,10934,
    10115,13987,16916,17965,19332,18251,22635,22562,27043,26135,34864,30683,26065,43438,
    21597,31534,31705,33251,33288,29145,24156,26385,27158,29164,29002,29916,25995,29468,
    26490,25858,37144,29873,33161,29256,23371,23901,25512,31787,30369,29794,29763,19138,
    22303,23366,30861,25996,26660,23792,18106,21467,20869,27191,22977,31967,13284,24481,
    23405,22860,20522,24268,26229,15342,24958,16429,19680,21304,18123,23553,26177,14790,
    24955,14676,20555,29034,29214,17919,17598,17376,20486,21744,22317,25468,21957,18577,
    28392,22834,27828,32218,32411,27616,26657,34313,37667,40588,44602,44703,41390,35664,
    43644,54357,52730,57718,52228,44361,46329,50304,64771,59260,66281,62918,60469,58858,
    60971,67404,72045,74710,67574,63201,57777,63028,70106,72219,74818,64582,61795,54448,
    59862,65935,68042,68605,58947,47576,49716,49988,53685,55836,62042,54590,48690,40522,
    55540,56307,52799,56729,54686,41893,38986,39318,46500,44864,46754,45265,38679,33076,
    37086,46393
    ]

days = list(range(len(cases)))

print(len(days))

days = np.asarray(days)
cases = np.asarray(cases)

days = days[:, np.newaxis]
cases = cases[:, np.newaxis]


plt.scatter(days, cases)
plt.show()


xseq = np.linspace(days.min(), days.max(), 300).reshape(-1,1)
regr = make_pipeline(PolynomialFeatures(12), LinearRegression())
regr.fit(days, cases)

plt.scatter(days, cases)
plt.plot(xseq, regr.predict(xseq), color = "red")
plt.show()

#===============================================================================================#
# Ref
# https://espanol.cdc.gov/coronavirus/2019-ncov/cases-updates/previouscases.html