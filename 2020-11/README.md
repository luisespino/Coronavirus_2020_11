# Technical Report 2020-11

Universidad de San Carlos de Guatemala

## Title (depends on what the majority write)

### Authors

Espino Barrios, Luis Fernando. (professor)

Alvarado Kevin, Cardona Berny, Galicia Nery, Giron Gary, Guarchaj Ottoniel, Hernandez Fernando Antonio, Hevia Eduardo, Lemus Yoselin, Lizama Luis, Mejía Kevin, Melgar James, Morales Mario, Ordoñez Bryan, Solares Cesar, Vega Daniel, Veliz Jorge, Villatoro Kherson. (students)

### Resume
In the current pandemic that the world is experiencing, COVID-19 has had great variations and tends to be misleading due to the new waves derived from social relaxation, today, there are several
factors that make comparisons between countries difficult, there are differences from one nation to another, as well as undetected or reported cases of the disease, in addition there are different types of tests, detection strategies, but this is limited by the different definitions and forms of classification, such as mild or asymptomatic cases that are not regularly accounted for, because they are not reported, so the quality of care can also be determining, even so it is important to have models with which it is possible to predict behavior of the disease in such a way that preventive measures can be taken at the population level to avoid contagion, as well as taking greater care with people at high risk, in said processes of determining characteristics and factors, different characteristics of the patients in the different analyzes carried out throughout the following report, age, sex, nationality, among others.
Next it will be about how the covid-19 pandemic has evolved since the first day it arrived in Guatemala and some countries in America, such as Honduras, Costa Rica, the United States, and even a comparison with countries on the other side of the world such as China. , addressing questions about the mortality rate, number of people infected per day, percentage of deaths according to active cases.
All the data collected and displayed have been extracted from the public databases of each country that track this information. With this information and using the Python programming language and its Sckit-learn library, it is possible to analyze and predict the data that will be shown below.

### Table of Contents

- [Covid-19 infection trend in Guatemala](#covid-19-infection-trend-in-guatemala)
- [Prediction of infecteds in Costa Rica](#prediction-of-infecteds-in-costa-rica)
- [Rise in COVID cases base on 4 values for country](#rise-in-covid-cases-base-on-4-values-for-country)
- [Mortality prediction due to COVID - 19 in the department of Guatemala](#Mortality-prediction-due-to-COVID-19-in-the-department-of-Guatemala)
- [References](#References)

### Covid-19 infection trend in Guatemala

The trend of infections is carried out through two perspectives: one having 30 days of information on infections in Guatemala at the beginning of the pandemic and the trend will be shown at different times.

According to the file [20080862.py](https://github.com/LuisEspino/CoronavirusML/blob/main/2020-11/src/20080862.py), the first execution was configured to predict infections at 50 days, the official number was 644 infected, although later the Guatemalan Ministry of Health corrected the number of infections by increasing them. So the prediction is approximately correct.

<p align="center">
<img src="https://user-images.githubusercontent.com/66042898/98481436-d327c080-21bf-11eb-852a-39c27f8c5cf0.jpg" width="400">
</p>

The second execution is made for 200 days, according to official data there were 90,968 infected, which indicates that the prediction is correct.

<p align="center">
<img src="https://user-images.githubusercontent.com/66042898/98488568-a4294300-21ef-11eb-95f3-90f74ba95960.jpg" width="400">
</p>

As of November, Guatemala is reducing the number of infections, so the prediction may vary. However, depending on what is happening in Europe there could be a second wave of infections in December, adjustments would have to be made to the model so that the prediction remains correct.

For the department of Guatemala, it is projected that by the end of November, the curve of confirmed cases will decrease.

### Prediction of infecteds in Costa Rica

In the case of Costa Rica, it had a different behavior since the growth curve of the infected behaved rather in the way of a polynomial of degree 4, that is, it can stabilize and begin a decline.

The degree of the polynomial has an R2 of 0.9989, that is, the data does fit the model. Also with grade 5 R2 is even closer to 0.9998, but the behavior does not correspond to a pandemic, because it goes down too quickly, without stabilizing.

| Polynomial | R2     | RMSE    |
| ---------- | ------ | ------- |
| Grade 4    | 0.9989 | 1184.62 |
| Grade 5    | 0.9998 | 461.72  |

It can be seen in the table that the mean square error RMSE is a lower value in the polynomial of degree 5, this means, that it is a better fit to the graph, also R2 is closer to 1, however, the behavior from a pandemic it cannot be taken down immediately, but the cases are decreasing little by little and the graph is gradually flattening out.

With the above, it can be determined that the degree of the polynomial must be 4 and discard the degree of 5.

<p align="center">
<img src="https://user-images.githubusercontent.com/37234131/99012300-a94e0100-2513-11eb-8a67-bcd93c686047.png" width="400">
</p>

According to the graph, we can observe that indeed, in the next 50 days, the advance of the infected will continue but could begin to stabilize, which is encouraging for the neighboring country.

| Day  | Date      | Infecteds | Data       |
| ---- | ----------| --------- | ---------- |
| 1    | 06/03/20  | 1         | Real       |
| 248  | 08/11/20  | 116,363   | Real       |
| 300  | 31/12/20  | 160,000   | Prediction |

With the above it is observed that by day 300, the number of infected will be close to 160,000 people, that is, it will continue to increase. That is, Costa Rica will close the year with about 160,000 infected.

The data used for the elaboration of the graph was obtained from [COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University](https://github.com/CSSEGISandData/COVID-19) In the case of Costa Rica, the first day for the graph is taken on March 5, 2020 with zero infected, and the final data of the graph is for November 8, 2020 with 116,363 infected, for a total of 249 days sample history.

The code used to generate the graph is [201503821.py](https://github.com/LuisEspino/CoronavirusML/blob/main/2020-11/src/201503821.py), which is based on a linear regression, with a polynomial of degree 4.

#### Comparative analysis of coronavirus cases between Costa Rica and Guatemala

Costa Rica has 5 million inhabitants, they have as of today, November 13, 2020, 120,939 cases of coronavirus, while Guatemala on the same day today with a population of 17 million inhabitants, have 113,543 cases of coronavirus. In other words, Costa Rica has more cases, when they are practically a third of the inhabitants that Guatemala has, this indicates that the sanitary measures taken by each of the people and the responsibility of each of them has made the difference between both countries to avoid contagions.

|                  | Guatemala  | Costa Rica |
| ---------------- | ---------- | ---------- |
| Population       | 17,250,000 | 4,999,000  |
| Infecteds        | 113,543    | 120,939    |
| Percentage       | 0.66%      | 2.41%      |

It is evident that Costa Rica has more cases of infected compared to Guatemala, which indicates that Guatemala has contained the disease in a better way and the restrictive measures have really helped.

#### Explanation of the code used for the prediction of infected in Costa Rica

- [201503821.py](https://github.com/LuisEspino/CoronavirusML/blob/main/2020-11/src/201503821.py)

The code was developed in Python using the Scikit Learn library, in this case a linear regression model was used, through a polynomial function of degree 4. The idea is to find a trend of the currently existing points and fit to test and error one degree for the polynomial, that is, this degree will coincide with the input data, once done it is possible to extend the data over time and make a prediction of the infected.

For this case, first a grade 5 was used, and the fit was very good, but when interpreting that it is a pandemic, the behavior was not correct, therefore another grade was sought, in this case 4, which also presents a good fit, and its behavior in more similar to that of pandemics.

### Mortality prediction due to COVID - 19 in the department of Guatemala

#### Resume

As days go by we will be able to determine a more accurate solution, by discovering the determined behavior, for the development corelated to the the excecuted solution. The mortality of this pandemic, which started to spread world-wide since december of 2019, prediction based on a linear regression, indicates that the outcome of whether there is a second wave derived from social relaxation related to the current decrease in social restrictions and curfews which were rendered without effect since the past month of october.

The solution of a development corresponding to the prediction that can determine the behavior that will continue to have as the days go by was executed, the mortality of the pandemic that affects the country and has spread worldwide since December of last year, this based in a linear regression, for which the results of said prediction are committed to whether there is a second wave derived from the social relaxation that is currently being experienced based on the drop in restrictions and curfews that have been without effect since the previous month .

Using linear regression, a projection of deaths was made from the deaths of the first 200 days since the pandemic began.

The data on the deceased persons were downloaded from the portal of the Ministry of Public Health and Social Assistance of Gautemala. These range from March 13 to November 8, 2020.

The pandemic that we are experiencing today has greatly changed the way people live and although the social fear that it represented has already diminished, it is still a threat to society, especially for those people who are already over 60 years old, have obesity , diabetes, heart disease and so on, that make them more vulnerable, that they should be cared for even more, having more protection measures because in any case they become infected can be a great risk to their life.
The reason why the prediction of deaths in the department of Guatemala is relevant is derived from the fact that with it the death rate can be calculated to which the results that are being expressed in the software are added, in such a way that can predict the amount and speed with which the virus is taking the lives of the most affected people, so that entities such as the government can take action and respond immediately or in advance preferably for any eventuality or change in the behavior currently registered, in such a way that it can spread on a smaller scale as well as control the cases of deaths due to COVID 19 which are presented daily with the respective values ​​that are recorded as there are changes in the information.

As can be seen in the graph below, the level of contagion worldwide is high and despite being Central America a territory with populations not as dense as other parts of the world, it is possible to identify that there is a great increase and impact of the pandemic:
<p align="center">
<img src="https://user-images.githubusercontent.com/20587564/99157783-b264dc80-2691-11eb-888f-4125ad755628.jpg" width="400">
</p>

#### Background
A clear indication of any infectious type disease, especially those that are caused by new pathogens such as SARS-CoV-2, tends to be severity since as a last resort it is measured by its killing capacity, this rate It can help to understand how serious this infectious virus can be, it even gives a vision of those who may be individuals at risk for the disease, either due to advanced age, or other previous health problems that can increase the severity of the virus, in such a way that if you have an estimate of such information, you can have a health assessment of care to prevent the greatest number of cases of death.

Currently there are several models by which the level of mortality in people, who have a tragic outcome, can be measured, for these models one of the two best known measures is surely used:
- The infection fatality ratio (known as IFR) that estimates the proportion of deaths among all people who have been infected.
- The case fatality ratio (known as CFR) which it measures is the proportional relationship between deaths and confirmed cases.

If the IFR method is followed for the measurement with accuracy, it is necessary to take into account that you must have a thorough knowledge of the infections and deaths that have been caused and derived from this virus. As a result, in the early phases of an epidemic, mostly a pandemic, the estimates of fatality motives have been based on the cases that have been detected through observation and they have been calculated through statistical methods, which leads to having estimates of the CFR rate that can range from less than 0.1% to even 25%.

In Covid-19, as with many other infectious diseases, the level with which it is spread is surely underestimated because it is not possible to identify a sample that is considerable from the population of infections in people, it may be because it is automatic Or it could be because they only have signs of mild symptoms, even that they are asymptomatic, therefore, because there is no seriousness, they do not go to the doctor or health centers. There may be groups of people who are not cared for or only partially, due to financial resources, means or other circumstances that prevent them from having the same opportunities to access medical care or to be tested for contagion. The detection of cases is closely related during the epidemic, since the ability to take a test is limited, mostly happening in complex and serious cases such as high-risk people such as the elderly or previous diseases, so it is also possible that the cases were misdiagnosed and some other situation or disease with a similar medical presentation is attributed to them, such as in this case a cold, lung infection, or respiratory disease.

The greatest difference found between groups of people and countries according to mortality is an indirect index of the relative risk that this population has of death that helps us to guide and determine decisions regarding the allocation of resources that should be used for this purpose, during the current COVID-19 pandemic, the main objective of the measurements and this analysis and prediction of deaths for the department of Guatemala, seeks to help verify the CFR at the department level, and even direct the investigator to the IFR to a higher accuracy, taking into account biases and that these data and their validity due to the fact that it is a linear regression is compressed to the emergence of a new wave.

##### Terminology note provided by the World Health Organization:
"The acronym CFR, applied to the measurement of the number of deaths among all people with a disease, usually means' rate
of case fatality ”, although properly speaking this expression is incorrect, since“ rate ”implies a time component
which is absent in the case of the CFR. Some authors have tried to rectify this inconsistency using the expressions
«Case fatality ratio» or «case case fatality ratio» (contrary to the proportion, in the ratio the numerator
it does not have to be a subset of the denominator). The expression «risk of case fatality», used with less
frequency is only correct if the duration of clinical disease is known. In this document we will use the expression
«Case fatality ratio». "


### Mortality prediction due to COVID - 19 in Honduras

The prediction was made on the 293rd day and deals with the number of deaths predicted for the 350th day after the COVID-19 in Honduras. According to the file [201313819.py](https://github.com/LuisEspino/CoronavirusML/blob/main/2020-11/src/201313819.py), it can be seen that the death curve has not been flattened and if the necessary measures are not taken, it may follow the flow of this prediction and on day 350 the mortality figure due to the virus would reach 500,000.

<p align="center">
<img src="https://user-images.githubusercontent.com/6562969/98753311-a6bfa000-2389-11eb-9e90-9cd0d67d7794.png" width="400">
</p>

## Trend of number of infected per day in Guatemala

The analysis is carried out by grouping the number of infected per day according to the number of infected, the day of the pandemic and the current day of the year.

According to the file 201212596.py, It can be seen in the image that 3 groups were created in which the number of infected per day and the days of the current year in which those infected were presented are shown.

With these data we can categorize based on the current day of the year and the number of infected of the day to which group it belongs and obtain an estimate of infections for that day.

<p align="center">
<img src="https://user-images.githubusercontent.com/13458088/99028380-36ee1880-2535-11eb-8208-321e1d860bd4.png" width="400">
</p>

### Centroid information

Grouping the information in 3 clusters, the k-means algorithm gave us the following information using 3 centroids:

- Centroid 1: 89 infections, 72 days of pandemic, 144 current days of the year.
- Centroid 2: 1067 infections, 120 days of pandemic, 192 current days of the year.
- Centroid 3: 558 infections, 163 pandemic days, 235 current days of the year.

### Data analysis
- Given the data shown for centroid 1, it can be concluded approximately that during the 72 days of the pandemic there were an average of 89 cases per day. 
- For centroid 2, it can be concluded approximately that within 120 days of the pandemic there were an average of 1067 cases per day. 
- For centroid 3, it can be concluded approximately that within 163 days of the pandemic there were an average of 558 cases per day.

--- 

### Number of cases per day of covid 19 in the US for 218 days

The linear correlation graph using the polynomial characteristics shows us the behavior of the cases that have been registered in the US, in the first 218 days and we can clearly see the second wave of infections that the country is going through.

<p align="center">
<img src="https://user-images.githubusercontent.com/37234131/99012033-17de8f00-2513-11eb-9ed3-74022bee3211.PNG" width="400">
</p>

The graph with the data analyzed during this period of time is shown, as well as the graph with the appropriate degree to model the behavior of the data.

<p align="center">
<img src="https://user-images.githubusercontent.com/37234131/99012125-45c3d380-2513-11eb-9b7b-dfc598ee209c.PNG" width="400">
</p>

For the analysis of cases in the United States, the following page was taken as a reference, from which the data of the first 218 days were taken. [COVID-19 in the US for 218 days](https://espanol.cdc.gov/coronavirus/2019-ncov/cases-updates/previouscases.html)

The data used and the graphs presented for the analysis of cases of covid 19 in the usa can be found here: [201612226.py](https://github.com/LuisEspino/CoronavirusML/blob/main/2020-11/src/201612226.py)


During the last days the situation of the month the situation in the country focused more on electoral issues, but as we can see the data indicates a new outbreak of massive contagion in the country that alarms many again, including specialists who warn that this It will be a new outbreak with very fast infections, an example of this is a total of 128,000 infections which breaks the new record.

Of the remaining time that Trump has left as president, it is known that he will not change his plan to combat covid 19 which does not have a national closure contemplated, in fact he is hopeful in the supposed vaccine that is expected this list in April 2021.

We can see that the situation is delicate in the country and that the infections will come as a new wave to the country. [rapid growth of COVID-19 in the United States](https://www.infobae.com/america/eeuu/2020/11/13/advierten-por-el-rapido-crecimiento-del-covid-19-en-estados-unidos-en-medio-de-la-transicion-de-poder-entre-joe-bien-y-donald-trump/)

---


### Analysis of the number of deaths from coronavirus in Guatemala

Simple linear regression is the most used technique, it is a way that allows modeling a relationship between two sets of variables. The result is an equation that can be used to make projections or estimates on the data.

This model is considered a predictor x and a dependent variable or response Y. Suppose that the true relationship between Y and x is a straight line and that observation Y at each level x is a random variable.

Analysis of the number of deaths from coronavirus in Guatemala through a linear regression, which says that when the reported deaths are far from what the actual behavior of said event should be, the analysis was carried out from day 1 to be reported the first case. Until day 226, the day on which 548 cases and 17 deaths were reported, all this always remembering the importance of the data regarding the pandemic and that they are not only numbers but also people who fall victims of this pandemic every day, always remembering the necessary prevention measures so that these types of graphs do not take the form they are taking due to so many deaths.

The training result model is as follows:

Y = 0.0805811339388978X + 7.712448377581117

<p align="center">
<img src="https://user-images.githubusercontent.com/15852159/99107552-a7378100-25ab-11eb-8a23-c19c7d36b849.png" width="400">
</p>

The data used and the graphs presented for the analysis of cases of covid 19 in the usa can be found here: [201403624.py](https://github.com/LuisEspino/CoronavirusML/blob/main/2020-11/src/201403624.py).

---



### Covid-19 in Guatemala, cases, predictions for 2021

In Guatemala it was predicted that at the end of this year we were going to see the peak of contagions, We have been in November 12 days and cases are seen in a small increase due to the opening of more shops, sales on the street and not the restriction of schedule.

According to the information given by the Ministry of Health and the reports that are exposed to people. It is said that there will be more contagion but the economy cannot be stop, thanks to this, we can predict that not only will a peak come but the quarantine that we already have since March 13 of this year will be extended. The beak that is said to be 2 months, can be extended up to 4 or 5 months.

In order to flatten the curve that we already have throughout all this, people will have to take care of themselves, protect themselves and take care of themselves otherwise, otherwise we will not be able to finish the fight against this virus, let alone finish quarantine.

In the graph below you can see in red line how the first curve that is already we had before is finished, but the other prediction is that cases will increase over time, by 2021 it is expected to have more contagions.

In the predictions given, it will be observed that whenever it is increasing, unless people take care of themselves, the curve will be able to go down in Guatemala.

<p align="center">
<img src="https://user-images.githubusercontent.com/6568351/98974867-02e60980-24db-11eb-8bc6-95284beba521.png" width="400">
</p>

The degree is 6, which is used to control the number of features added so that the prediction is correct and specified.

### Percentage of men infected by covid-19 in Guatemala since the first active case

The graphic below shows up how infected cases in mens are increasing and decreasing during the days, we can see how infected cases are shown by blue dots in the graph,
futhermore, we can fit a trend line to these cases using a polynomial function of degree 4, in this way, we have a data correlation of 0.69.
With these related data we could predict active cases in Guatemalan men by day 350 of the pandemic

<p align="center">
<img src="https://user-images.githubusercontent.com/20606917/99028199-ca731980-2534-11eb-9b42-be27b8432845.png" width="400">
</p>

### Comparative between Guatemala, Central America and Japan

As we can apretiate in the graphs above, we notice a trend between Guatemala and Countries in Central when it comes to covid stats, which is somehow expected
since we can find alot of cultural similarities between this countries, but now we will compare them to Japan.
First we need to establish why Japan was the target, there are three main reasons:

1. They started doing quarentine like the rest of the world, then they decided they were just going to have to learn to "live with it" and resumed their daily duties,
   with some added restrictions of course, but this was done at an early stage; meanwhile in Guatemala the quarentine lasted for longer time, and restrictions were more severe.
2. They way they have dealt with the pandemic has been outstanding, reflecting in the lowest numbers per capita world wide.
3. Their social standards, which are regarded to be among the best in the world.

So we start by reviewing this Graph created using the file  [201222567.py], where we seek a model beginning from start of the second wave, october 1st currently with 83010 confirmed cases, and aimed towards predicting using a polynomial regression model, the number of confirmed cases by november 30, the end of the month, since november is the
scope of this research.

<p align="center">
<img src="https://user-images.githubusercontent.com/10952236/99139973-299f5f80-2603-11eb-9504-512953798b69.png" width="400">
</p>

As we can see, the numbers seem to similar to Guatemala, but we must also take into consideration that Japans population is 126.5 million as of 2018 and Guatemalas population
is 17.25 million as of 2018 which is a staggering difference, this reflects in the behavior shown in the graphs, Japans seems to be more steady, while most of
Central America is quite the oposite.
So now we must ask ourselves which factors create this difference, and the one that stands out the most is the culture, considering our current situation
perhaps the best way to mitigate the impact of the innevitable second wave might be benchamark the succesful way in which other countrys are fighting this, and we can
clearly see that Japan is a great example as we can see in this predicted outcome:

## Average deaths from confirmed cases and age of covid 19 in Guatemala

Taking into account the number of deaths, the number of confirmed cases by average age taking into account the years from ten to ten starting at zero and ending
In one hundred, that is, for the first group of data the average age would be five, for the second fifteen and for the last it would be ninety-five. We proceeded to the analysis of
these data using artificial intelligence where by means of the k-means algorithm from the sklearn library using the following data set.

| Number of deaths | Confirmed cases | Average age |
| ---------------- | --------------- | ----------- |
| 31               | 3045            | 5           |
| 21               | 5790            | 15          |
| 107              | 28883           | 25          |
| 209              | 28398           | 35          |
| 479              | 18996           | 45          |
| 757              | 12701           | 55          |
| 1032             | 7673            | 65          |
| 575              | 2317            | 75          |
| 241              | 827             | 85          |
| 41               | 134             | 95          |

Grouping the data by 2 clusters, the result of the k-means algorithm indicates the following:

### Centroid 1

_where each data refers to the mean of the grouped data_

| Number of deaths | Confirmed cases | Average age |
| ---------------- | --------------- | ----------- |
| 385              | 4641            | 56          |

### Centroide 2

_where each data refers to the mean of the grouped data_

| Number of deaths | Confirmed cases | Average age |
| ---------------- | --------------- | ----------- |
| 265              | 25425           | 35          |

### Graph of the previous data

![Grafica de clusters ](https://user-images.githubusercontent.com/12839670/99020367-0ea9ee00-2524-11eb-9323-fff1fd87ca90.JPG)

The graph only shows the number of deaths in two dimensions on the "X" axis and the number of confirmed cases on the "Y" axis.

### Analysis of data

From the data shown for centroid 1, it can be said that for every 464 Data analysis1 positive cases with people with an average age of 56 years, 385 will die.

From the data shown for centroid 2, it can be said that for every 25,425 positive cases with people with an average age of 35 years, 265 will die.

What seeing it in a fast way and corobrating what is indicated in a global way, older people are more susceptible to dying, since out of 4641 cases the
8% while of the cases of the youngest people of 25425 cases 1% dies.

With the estimate given above, it is possible to predict the number of deaths that there will be in a hospital, city or any place in general since we can deduce the probability that a person has of dying according to their age and the number of deaths that there will be in a population.

### How to predict

The correct way to find out if a person belongs to one set of data or another is through their age, if their age is closer to 56 years than to 35 years it means that they belong to that set of information, otherwise If your age is closer to 35 years than 56 years, it belongs to the second set of information.

### China, the city where everything started. How are they now?

China is the city where the pandemic started. This year, they experimented a second wave of Covid 19. The objective of this analysis is to determine what was the behavior of the second wave of Covid in China and how long it took to overcome. The analysis begins on January 22, 2020. As can be seen in the following graph, the highest peak of infections was reached in February, with a maximum of 15136 cases of contagion, which represented an anomaly compared to the values recorded in the previous days. This day marked a clear limit in the case of infections, in the following days the policies
taken by the Chinese government were stricter and a began a decrease in daily cases. By March 10, the days in which more than 100 cases were registered were very few. We can say with complete certainty, based on the following graph, that the second wave lasted approximately two months and its peak was one month after the first contagions began to register. Clearly, this decrease in cases was due to stricter health policies and the collaboration of all citizens.

<p align="center">
<img src="https://user-images.githubusercontent.com/34200816/99139374-9283d900-25fd-11eb-9cc9-955d600e031a.png" width="400">
</p>

The data used for this analysis can be found in [this link](https://github.com/LuisEspino/CoronavirusML/blob/main/2020-11/data/201602656.py)

According to the file [20080862.py](https://github.com/LuisEspino/CoronavirusML/blob/main/2020-11/src/20080862.py), the first execution was configured to predict infections at 50 days, the official number was 644 infected, although later the Guatemalan Ministry of Health corrected the number of infections by increasing them. So the prediction is approximately correct.

### Clusters: Deceased according to regions of Guatemala - Covid 19

According to the file [201403819.py](https://github.com/LuisEspino/CoronavirusML/blob/main/2020-11/201403819.py), there is an anomalous region, since this is where the largest number of deaths.

The x-axis shown in the graph represents the regions in which Guatemala is divided.

The y-axis represents the number of deaths.

There is a cluster divided into 3 classes.

<p align="center">
<img src="https://user-images.githubusercontent.com/67341591/99140684-784ff800-2609-11eb-8135-519d8d31dc0c.png" width="400">
</p>

The sources for the data used are at:
- [Regions](https://aprende.guatemala.com/historia/geografia/regiones-de-guatemala/)
- [Number of deceased](https://tablerocovid.mspas.gob.gt/)

# Percentage of Deaths vs. total cases in Central America

This study looks at the percentage of Covid-19 cases in total divided by the number of deaths in each Central American country (Guatemala, Honduras, Nicaragua and El Salvador), each country has a different percentage for example:

In El Salvador the month of July had a number of cases of 18096 and a number of deaths of 773 when making the division gives us a total of 23.4%, making the division of each month the average is 26.2% and with this average we classify each month as above average or below average, in this case the month of July is classified as (No) which means that it is not above average.

This analysis has the purpose of being able to enter the data from other months and that this through skelearn does not say if there is an improvement in the number of deaths or if, on the contrary, the country had a greater amount than the average.

Each country has a different average, which is the following:

|     country      | average | 
| ---------------- | ------- |
| Guatemala        | 36%     | 
| El Salvador      | 26.2%   | 
| Nicaragua        | 32%     | 
| Honduras         | 32.9%   | 

For this study we used the months of July, August, September and October.

### Rise in COVID cases base on 4 values for country

Based on the theory that the increase in COVID cases depends on the days of initial quarantine, travel restrictions, the percentage of investment in public spending on health, and the population density, I set myself the task of knowing if an increase could occur in Guatemala. in the cases with a model trained with the Bayes algorithm in the file [201504394.py](https://github.com/LuisEspino/CoronavirusML/blob/main/2020-11/src/201504394.py).

The data collected can be found in this repository at [201504394.xlsx](https://github.com/LuisEspino/CoronavirusML/blob/main/2020-11/data/201504394.xlsx). The sources for the data used to train the model are at:
- [Initial quarantine days](https://es.wikipedia.org/wiki/Cuarentena_por_la_pandemia_de_COVID-19)
- [Travel restrictions](https://www.kayak.com.gt/restricciones-de-viaje)
- [Investment in health](https://datosmacro.expansion.com/estado/gasto/salud)
- [Population density](https://datosmacro.expansion.com/demografia/poblacion)
- [Increase in cases](https://news.google.com/covid19/map?hl=es-419&gl=US&ceid=US%3Aes-419)



# Behavioral rate of active cases in relation to the number of deaths in some Caribbean countries

This approach is a little different from the others, since on this occasion we connected in seeing the behavior of active cases with respect to the number of deaths in some Caribbean countries, such as the following that we will mention below.

This time, we will use the Gaussian algorithm to predict whether or not the data we will send as parameters (country, number of active cases, number of deaths) will show an improvement in the country. This would be another way of alerting countries to the behavior of the virus in their region, where they can know whether they have been improving or getting worse in order to take the necessary measures, this with respect to the rate of active cases versus deaths for each month.
For this specific case we use data from the cases registered in:
https://www.worldometers.info/coronavirus/#countries

The months we take into account are from June, July, August, September and October, for the countries of Jamaica, Haiti and Cuba.

For example: In Jamaica in the month of June there are 266 active cases with 9 deaths per covid registered. Therefore, the rate we manage by dividing the 266 active cases into 9 deaths, which gives us a value of 29.55 active cases/deaths. (266/9=29.55). And in Cuba we have recorded that by the month of June there were 174 active cases and 83 deaths, which gives a value of 2.09. So here we can say that the lower the number the more alarming the cases are, because most of the active cases are ending in death.


| COUNTRY          | AVERAGE | 
| ---------------- | ------- |
| Jamaica          | 34.3%   | 
| Haiti      	   | 31.6%   | 
| Cuba        	   | 3.1%    |  

In the table we can see that Haiti has a lower average, this is due to the fact that in the data taken from the reference, we see a drastic change in the counts of active cases, between the month of June, July and August. This makes the case look alarming there, just as in the rest most active cases end in death. Jamaica, on the other hand, has a higher average number of active cases, since there are more cases registered in that country and only a small number of them end in death.

### Confirmed cases by Latin American countries
The trend of coronavirus infections in Latin America is very important to know its impact.
The original dataset is converted into a regression dataset using multilayer perceptron artificial neural networks. These results describe a model capable of predicting the number of confirmed cases on a date and country in Latin America.

#### Dataset
Dataset used in the model is obtained from a [publicly available repository](https://github.com/CSSEGISandData/COVID-19) operated by the Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE). It contains the data which describe the number of cases defined by province/state, country/region, latitude and longitude for each day since the start of the COVID-19 infections (01-22-2020). It use only the time series confirmed cases global. The dataset that use the model (obtained by the original dataset) contain the data for latin american countries.

#### Model
According to the file [201602822.py](https://github.com/LuisEspino/CoronavirusML/blob/main/2020-11/src/201602822.py), it use the scikit-learn library with pickle, matplotlib, pandas and numpy packages.
The model use the ReLU activation function and the L-BFGS solver. Consist of four hidden layers and 40 total hidden reurons distributed equally among layers. This model takes the latitude, longitude and days since first case as input data and the number of confirmed cases as output data.

#### Results
The figure shows the comparison of real data and the data predicted by model. It presents a number of cases in a given group (location and date).
<p align="center">
<img src="https://user-images.githubusercontent.com/34287415/99181391-d1945600-26f3-11eb-9b75-4491e42bc3b6.png" width="400">
</p>

The maximum of each daily count is plotted.
<p align="center">
<img src="https://user-images.githubusercontent.com/34287415/99180350-1d8ecd00-26eb-11eb-9847-d8bdeae445bc.png" width="400">
</p>

# Percentage of Deaths vs. total cases in Central America

This study looks at the percentage of Covid-19 cases in total divided by the number of deaths in each Central American country (Guatemala, Honduras, Nicaragua and El Salvador), each country has a different percentage for example:

In El Salvador the month of July had a number of cases of 18096 and a number of deaths of 773 when making the division gives us a total of 23.4%, making the division of each month the average is 26.2% and with this average we classify each month as above average or below average, in this case the month of July is classified as (No) which means that it is not above average.

This analysis has the purpose of being able to enter the data from other months and that this through skelearn does not say if there is an improvement in the number of deaths or if, on the contrary, the country had a greater amount than the average.

Each country has a different average, which is the following:

|     country      | average | 
| ---------------- | ------- |
| Guatemala        | 36%     | 
| El Salvador      | 26.2%   | 
| Nicaragua        | 32%     | 
| Honduras         | 32.9%   | 

For this study we used the months of July, August, September and October.

### Rise in COVID cases base on 4 values for country

Based on the theory that the increase in COVID cases depends on the days of initial quarantine, travel restrictions, the percentage of investment in public spending on health, and the population density, I set myself the task of knowing if an increase could occur in Guatemala. in the cases with a model trained with the Bayes algorithm in the file [201504394.py](https://github.com/LuisEspino/CoronavirusML/blob/main/2020-11/src/201504394.py).

The data collected can be found in this repository at [201504394.xlsx](https://github.com/LuisEspino/CoronavirusML/blob/main/2020-11/data/201504394.xlsx). The sources for the data used to train the model are at:
- [Initial quarantine days](https://es.wikipedia.org/wiki/Cuarentena_por_la_pandemia_de_COVID-19)
- [Travel restrictions](https://www.kayak.com.gt/restricciones-de-viaje)
- [Investment in health](https://datosmacro.expansion.com/estado/gasto/salud)
- [Population density](https://datosmacro.expansion.com/demografia/poblacion)
- [Increase in cases](https://news.google.com/covid19/map?hl=es-419&gl=US&ceid=US%3Aes-419)



# Behavioral rate of active cases in relation to the number of deaths in some Caribbean countries

This approach is a little different from the others, since on this occasion we connected in seeing the behavior of active cases with respect to the number of deaths in some Caribbean countries, such as the following that we will mention below.

This time, we will use the Gaussian algorithm to predict whether or not the data we will send as parameters (country, number of active cases, number of deaths) will show an improvement in the country. This would be another way of alerting countries to the behavior of the virus in their region, where they can know whether they have been improving or getting worse in order to take the necessary measures, this with respect to the rate of active cases versus deaths for each month.
For this specific case we use data from the cases registered in:
https://www.worldometers.info/coronavirus/#countries

The months we take into account are from June, July, August, September and October, for the countries of Jamaica, Haiti and Cuba.

For example: In Jamaica in the month of June there are 266 active cases with 9 deaths per covid registered. Therefore, the rate we manage by dividing the 266 active cases into 9 deaths, which gives us a value of 29.55 active cases/deaths. (266/9=29.55). And in Cuba we have recorded that by the month of June there were 174 active cases and 83 deaths, which gives a value of 2.09. So here we can say that the lower the number the more alarming the cases are, because most of the active cases are ending in death.


| COUNTRY          | AVERAGE | 
| ---------------- | ------- |
| Jamaica          | 34.3%   | 
| Haiti      	   | 31.6%   | 
| Cuba        	   | 3.1%    |  

In the table we can see that Haiti has a lower average, this is due to the fact that in the data taken from the reference, we see a drastic change in the counts of active cases, between the month of June, July and August. This makes the case look alarming there, just as in the rest most active cases end in death. Jamaica, on the other hand, has a higher average number of active cases, since there are more cases registered in that country and only a small number of them end in death.

### Confirmed cases by Latin American countries
The trend of coronavirus infections in Latin America is very important to know its impact.
The original dataset is converted into a regression dataset using multilayer perceptron artificial neural networks. These results describe a model capable of predicting the number of confirmed cases on a date and country in Latin America.

#### Dataset
Dataset used in the model is obtained from a [publicly available repository](https://github.com/CSSEGISandData/COVID-19) operated by the Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE). It contains the data which describe the number of cases defined by province/state, country/region, latitude and longitude for each day since the start of the COVID-19 infections (01-22-2020). It use only the time series confirmed cases global. The dataset that use the model (obtained by the original dataset) contain the data for latin american countries.

#### Model
According to the file [201602822.py](https://github.com/LuisEspino/CoronavirusML/blob/main/2020-11/src/201602822.py), it use the scikit-learn library with pickle, matplotlib, pandas and numpy packages.
The model use the ReLU activation function and the L-BFGS solver. Consist of four hidden layers and 40 total hidden reurons distributed equally among layers. This model takes the latitude, longitude and days since first case as input data and the number of confirmed cases as output data.

#### Results
The figure shows the comparison of real data and the data predicted by model. It presents a number of cases in a given group (location and date).
<p align="center">
<img src="https://user-images.githubusercontent.com/34287415/99181391-d1945600-26f3-11eb-9b75-4491e42bc3b6.png" width="400">
</p>

The maximum of each daily count is plotted.
<p align="center">
<img src="https://user-images.githubusercontent.com/34287415/99180350-1d8ecd00-26eb-11eb-9847-d8bdeae445bc.png" width="400">
</p>


# Behavior and classification of people infected by COVID-19 by municipality in Guatemala

The following prediction, rather than a prediction, is focused on classifying into "groups" the most infected sectors within the departments of Guatemala, in this case, its municipalities.

## Preliminaries
Since last March 13, when the first COVID-19 contagion was detected in Guatemala, the pandemic has been growing steadily in the country until the second week of April, when there was a peak of 39 cases in a single day , and then continue with the marked trend of a score of daily cases. However, since the second week of May, infections began to increase exponentially, from 68 daily cases on May 8 to 370 new cases in a single day on May 24. In June the cases already exceeded a thousand cases a day. And in July, with a new dashboard, the numbers have become even less reliable.

The president, to evaluate his decisions focused on the reopening of the country before the population, paradoxically showed the data of a country with more municipalities on maximum alert, with a red light for more than half of the country.

These indicators come from the COVID19 dashboard that the Ministry of Health and Social Assistance (MSPAS) feeds daily. However, what the president ignored in his presentation was that, although the number of daily infections has decreased compared to previous weeks, the deceased are a figure that is increasing. Deaths from coronavirus have become the biggest challenge in the region.

## Analysis
Based on the foregoing, it can be determined that for the information provided by the country's high command to have logic to be able to make the traffic light classification, they had to group the municipalities based on their infected.

<p align="center">
<img src="https://user-images.githubusercontent.com/6531771/99190774-1ccb5a80-272e-11eb-8f6b-649e7d43a3d0.png" width="400">
</p>

This can clearly be done with a prediction using a machine learning algorithm to create the necessary clusters (in this case, the lights of a traffic light) to be able to identify the measures that a municipality can count on depending on its number of infected .

This prediction has the advantage that it is not only based on the data of the day but can also work preventively in order to find a pattern in the data in the future.

## Guatemala Goverment Alert Level System - Bayes Naive Classification

The government of Guatemala manages the restrictions that are applied in each municipality and economic activity according to a system called "Semaforo". This system is based on 3 factors which are: Rate per 100k inhabitants, Percentage of positive tests and Rate of tests per 1k inhabitants. So, depending on this 3 factor the goverment set the alert lavel and take measures.


For example:

| Rate per 100k inhabitans | % Positive tests | Rate of tests per 1k inhabitants | Alert Level |
| ------------------------ | ---------------- | -------------------------------- | ----------- |
| 14.65                    | 13.56            | 0.05                             | 5           |


Using the data provided by the goverment on [this site](https://covid19.gob.gt/semaforo.html), We make a clasification model based on Naive Bayes (GaussianNB) algorithm to classify the alert level for future results of the 3 factors. 

The alert level is in a scale from 1 to 10, according to the data, alert level can take levels in step of .5 meaning that can be levels like 4.5 or 9.5, thats why in we use LabelEncoder from Sklearn to make this values unique and have a more consistent scale.

In our tests we get the following values:

| Rate per 100k inhabitans | % Positive tests | Rate of tests per 1k inhabitants | Alert Level |
| ------------------------ | ---------------- | -------------------------------- | ----------- |
| 119.44                   | 11.50            | 0.74                             | 7           |
| 25.65                    | 11.20            | 0.16                             | 6           |
| 50.0                     | 5.0              | 0.20                             | 7           |

| Precision |
| --------- |
| 0.6823529 |

The first result row was an already known value (7) , this one was for control, the model have a precision on 68.23529%, we are aware that this model can be more precise but due to the actual data inconsitency on data capture we can not asure more precision.

Source Code [200714832.py]
Data File [200714832.xlsx]

### References

- Supervised learning — scikit-learn 0.23.2 documentation. (s. f.). Scikit Learn. Retrieved November 08, 2020, https://scikit-learn.org/stable/supervised_learning.html#supervised-learning
- C. (s. f.). CSSEGISandData/COVID-19. GitHub. Retrieved November 08, 2020, https://github.com/CSSEGISandData/COVID-19
