# Technical Report 2020-11

Universidad de San Carlos de Guatemala

## Title (depends on what the majority write)

### Authors

Espino Barrios, Luis Fernando. (professor)

Cardona Berny, Galicia Nery, Giron Gary, Mejía Kevin, Solares Cesar, Bryan Ordoñez, Villatoro Jerson, Veliz Jorge, Fernando Antonio Hernandez, Vega Daniel, Hevia Eduardo. (students)

### Resume

...

### Covid-19 infection trend in Guatemala

The trend of infections is carried out through two perspectives: one having 30 days of information on infections in Guatemala at the beginning of the pandemic and the trend will be shown at different times.

According to the file [20080862.py](https://github.com/LuisEspino/CoronavirusML/blob/main/2020-11/20080862.py), the first execution was configured to predict infections at 50 days, the official number was 644 infected, although later the Guatemalan Ministry of Health corrected the number of infections by increasing them. So the prediction is approximately correct.

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

The degree of the polynomial has an R2 of 0.998, that is, the data does fit the model. Also with grade 5 R2 is even closer to 0.999, but the behavior does not correspond to a pandemic, because it goes down too quickly, without stabilizing.

<p align="center">
<img src="https://user-images.githubusercontent.com/37234131/99012300-a94e0100-2513-11eb-8a67-bcd93c686047.png" width="400">
</p>

According to the graph, we can observe that indeed, in the next 50 days, the advance of the infected will continue but could begin to stabilize, which is encouraging for the neighboring country.

With the above it is observed that by day 300, the number of infected will be close to 160,000 people, that is, it will continue to increase.

The data used for the elaboration of the graph was obtained from [COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University](https://github.com/CSSEGISandData/COVID-19) In the case of Costa Rica, the first day for the graph is taken on March 5, 2020 with zero infected, and the final data of the graph is for November 8, 2020 with 116,363 infected, for a total of 249 days sample history.

#### Comparative analysis of coronavirus cases between Costa Rica and Guatemala

Costa Rica has 5 million inhabitants, they have as of today, November 13, 2020, 120,939 cases of coronavirus, while Guatemala on the same day today with a population of 17 million inhabitants, have 113,543 cases of coronavirus. In other words, Costa Rica has more cases, when they are practically a third of the inhabitants that Guatemala has, this indicates that the sanitary measures taken by each of the people and the responsibility of each of them has made the difference between both countries to avoid contagions.

### Mortality prediction due to COVID - 19 in the department of Guatemala

As days go by we will be able to determine a more accurate solution, by discovering the determined behavior, for the development corelated to the the excecuted solution. The mortality of this pandemic, which started to spread world-wide since december of 2019, prediction based on a linear regression, indicates that the outcome of whether there is a second wave derived from social relaxation related to the current decrease in social restrictions and curfews which were rendered without effect since the past month of october.

The solution of a development corresponding to the prediction that can determine the behavior that will continue to have as the days go by was executed, the mortality of the pandemic that affects the country and has spread worldwide since December of last year, this based in a linear regression, for which the results of said prediction are committed to whether there is a second wave derived from the social relaxation that is currently being experienced based on the drop in restrictions and curfews that have been without effect since the previous month .

Using linear regression, a projection of deaths was made from the deaths of the first 200 days since the pandemic began.

The data on the deceased persons were downloaded from the portal of the Ministry of Public Health and Social Assistance of Gautemala. These range from March 13 to November 8, 2020.

The pandemic that we are experiencing today has greatly changed the way people live and although the social fear that it represented has already diminished, it is still a threat to society, especially for those people who are already over 60 years old, have obesity , diabetes, heart disease and so on, that make them more vulnerable, that they should be cared for even more, having more protection measures because in any case they become infected can be a great risk to their life.
The reason why the prediction of deaths in the department of Guatemala is relevant is derived from the fact that with it the death rate can be calculated to which the results that are being expressed in the software are added, in such a way that can predict the amount and speed with which the virus is taking the lives of the most affected people, so that entities such as the government can take action and respond immediately or in advance preferably for any eventuality or change in the behavior currently registered, in such a way that it can spread on a smaller scale as well as control the cases of deaths due to COVID 19 which are presented daily with the respective values ​​that are recorded as there are changes in the information.

### Mortality prediction due to COVID - 19 in Honduras

This prediction was made on the 293rd day and deals with the number of deaths predicted for the 350th day after the COVID-19 in Honduras. According to the graph, it can be seen that on day 350 the mortality figure due to the virus would reach 500,000. It can be seen that the death curve has not been flattened and if the necessary measures are not taken, it may follow the flow of this prediction

<p align="center">
<img src="https://user-images.githubusercontent.com/6562969/98753311-a6bfa000-2389-11eb-9e90-9cd0d67d7794.png" width="400">
</p>

### Trend of number of infected per day in Guatemala

The analysis is carried out by grouping the number of infected per day according to the number of infected, the day of the pandemic and the current day of the year.

It can be seen in the image that 3 groups were created in which the number of infected per day and the days of the current year in which those infected were presented are shown.

With these data we can categorize based on the current day of the year and the number of infected of the day to which group it belongs and obtain an estimate of infections for that day.

<p align="center">
<img src="https://user-images.githubusercontent.com/13458088/99028380-36ee1880-2535-11eb-8208-321e1d860bd4.png" width="400">
</p>


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

The data used and the graphs presented for the analysis of cases of covid 19 in the usa can be found here: [201612226.py](https://github.com/LuisEspino/CoronavirusML/blob/main/2020-11/201612226.py)

---

Analysis of the number of deaths from coronavirus in Guatemala by means of a linear regression, which says when the reported deaths are far from what should be the real behavior of said event, the analysis was carried out from day 1 to be reported the first case until day 226, the day in which 548 cases and 17 deaths were reported.

The training result model is as follows:

Y = 0.0805811339388978X + 7.712448377581117

<p align="center">
<img src="https://user-images.githubusercontent.com/15852159/99107552-a7378100-25ab-11eb-8a23-c19c7d36b849.png" width="400">
</p>

### Covid-19 in Guatemala, cases, predictions for 2021

In Guatemala it was predicted that at the end of this year we were going to see the peak of contagions, We have been in November 12 days and cases are seen in a small increase due to the opening of more shops, sales on the street and not the restriction of schedule.

According to the information given by the Ministry of Health and the reports that are exposed to people. It is said that there will be more contagion but the economy cannot be stop, thanks to this, we can predict that not only will a peak come but the quarantine that we already have since March 13 of this year will be extended. The beak that is said to be 2 months, can be extended up to 4 or 5 months.

In order to flatten the curve that we already have throughout all this, people will have to take care of themselves, protect themselves and take care of themselves otherwise, otherwise we will not be able to finish the fight against this virus, let alone finish quarantine.

In the graph below you can see in red line how the first curve that is already we had before is finished, but the other prediction is that cases will increase over time, by 2021 it is expected to have more contagions.

In the predictions given, it will be observed that whenever it is increasing, unless people take care of themselves, the curve will be able to go down in Guatemala.

<p align="center">
<img src="https://user-images.githubusercontent.com/6568351/98974867-02e60980-24db-11eb-8bc6-95284beba521.png" width="400">
</p>

### Percentage of men infected by covid-19 in Guatemala since the first active case

The graphic below shows up how infected cases in mens are increasing and decreasing during the days, we can see how infected cases are shown by blue dots in the graph,
futhermore, we can fit a trend line to these cases using a polynomial function of degree 4, in this way, we have a data correlation of 0.69.
With these related data we could predict active cases in Guatemalan men by day 350 of the pandemic

<p align="center">
<img src="https://user-images.githubusercontent.com/20606917/99028199-ca731980-2534-11eb-9b42-be27b8432845.png" width="400">
</p>

### Comparative between Guatemala/Central america and Japan

As we can apretiate in the graphs above, we can see a similarities between Guatemala and Countries in Central when it comes to covid stats, which is somehow expected
since we can find alot of cultural similarities between this countries, but now we will compare them to Japan.
First we need to establish why Japan was the target, there are three main reasons:

1. They started to do quarentine like the rest of the world, then they decided they were just going to have to learn to "live with it" and resumed their daily duties,
   with some added restrictions of course, but this was done at an early stage; meanwhile in Guatemala the quarentine lasted for longer time, and restrictions were more severe.
2. They way they have dealt with the pandemic has been outstanding, reflecting in the lowest numbers per capita world wide.
3. Their social standards, which are regarded to be among the best in the world.

So we start by reviewing this Graph, where we find a model starting since the beggining of the second wave, october 1st, and aimed towards predicting, using a polynomial regression model. the number of confirmed cases by november 30, the end of the month

--- graph

As we can see, the numbers seem to similar to Guatemala, but we must also take into consideration that Japans population is 126.5 million as of 2018 and Guatemalas population
is 17.25 million as of 2018 which is a staggering difference, this reflects in the behavior shown in the graphs, Japans seems to be more steady, while most of
Central America is quite the oposite.
So now we must ask ourselves which factors create this difference, and the one that stands out the most is the culture, considering our current situation
perhaps the best way to mitigate the impact of the innevitable second wave might be benchamark the succesful way in which other countrys are fighting this, and we can
clearly see that Japan is a great example.

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

### References

... more articles from students (the order of the articles will be defined by the professor)
...
