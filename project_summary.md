# Bike Sharing Demand: A Regression Project

The goal for this week’s project is to build and train a regression model on the Capital Bike Share (Washington, D.C.) Kaggle data set, in order to **predict demand for bicycle rentals at any given hour**, based on time and weather, e.g.:

*“Given the forecasted weather conditions, how many bicycles can we expect to be rented out (city-wide) this Saturday at 2pm?”*

## The Data Set

The data set contains hourly data from 2011-01-01 until 2012-12-19 of the following 11 features:

- Datetime 

- season (1 = spring, 2 = summer, 3 = fall, 4 = winter)
- holiday (binary 0,1)
- workingday (binary 0,1)
- weather (1: Clear, Few clouds, Partly cloudy, Partly cloudy 
  2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist 
  3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds 
  4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog )
- temp (metric in °C)
- atemp (metric in °C)
- humidity (metric in %)
- windspeed (metric in m/s)
- count of casual users
- count of registered users
- total count

## Data Exploration

The demand for bikes depends on many factors. There can be seen a correlation between the count of bikes and the hour of the day or if it is a weekday or not. The first figure below shows the demand on a weekday with two peeks at rush-hour the second figure shows the demand on a saturday with a peak in the early afternoon.

![](/Users/brittapuyn/Documents/SPICED/spiced_projects/nlpepper-student-code/03_week/project/2012-01-06.jpg)

![2012-01-07](/Users/brittapuyn/Documents/SPICED/spiced_projects/nlpepper-student-code/03_week/project/2012-01-07.jpg)

There are also correlations between the demand and the the season of the year and the weather category:

![season_count](/Users/brittapuyn/Documents/SPICED/spiced_projects/nlpepper-student-code/03_week/project/output/season_count.jpg)

The demand in spring is the lowest and in summer is the highest demand for bikes

![weather_count](/Users/brittapuyn/Documents/SPICED/spiced_projects/nlpepper-student-code/03_week/project/output/weather_count.jpg)

If the weather conditions are bad, the demand for bikes descreases significantly.

![temperature_count](/Users/brittapuyn/Documents/SPICED/spiced_projects/nlpepper-student-code/03_week/project/output/temperature_count.jpg)

The demand for bikes increases with higher temperatures. If the temperatures are too high (> 30°C) a small decrease of bike demand can be recognized.

![windspeed_count](/Users/brittapuyn/Documents/SPICED/spiced_projects/nlpepper-student-code/03_week/project/output/windspeed_count.jpg)

The daily bike count decreases with increasing windspeed.

![humidity_count](/Users/brittapuyn/Documents/SPICED/spiced_projects/nlpepper-student-code/03_week/project/output/humidity_count.jpg) 

Also a decrease of bike damand for high relative humidity can be recognized.

## Results

### 1. Poisson Regression 

My first try was a Poisson regression model, which I tuned a lot for reaching a moderate score at Kaggle. I used almost every features from the dataset as input features and the total bike count as output:

`X = df[['datetime','temp','weather','season','holiday','workingday','windspeed']]`

`y = df['count']`

#### Feature Engineering Pipeline:

1. splitting datetime into columns: `['year', 'month', 'weekday','hour']`
2. Column transforming:
   1. One-Hot-Encoding: `['weekday','season','weather']`
   2. Binning `['hour']`
   3. scaling `['temp','windspeed]`
   4. Passthrough `['holiday']`
3. Polynomial Features `degree=2`

**Number of Features: 945**

#### Hyperparameter 

I executed a GridSearchCV for optimizing following hyper parameter (see optimum result):

- n_bins (`24`)
- binning_strategy (`uniform`)
- polynomial degree (`2`)
- alpha (`0.01`)
- max_iter (`1000`)

**Kaggle Score = 0.45584 (mean sqared log error)**

The picture below shows the comparison between the test data and the predicted results of the training dataset.

![poissonreg_res](/Users/brittapuyn/Documents/SPICED/spiced_projects/nlpepper-student-code/03_week/project/output/poissonreg_res.jpg)



### 2. Random Forest Regressor

Secondly I tried a RandomForest Regressor for comparison. I started with very low effort on the feature engineering (only splitting the datetimecolumn) and got already very reasonable results with a very low number of total features.

I used the same input and output features as in the poisson regression model:

`X = df[['datetime','temp','weather','season','holiday','workingday','windspeed']]`

`y = df['count']`

#### Feature Engineering Pipeline:

1. splitting datetime into columns: `['year', 'month', 'weekday','hour']`
2. Column transforming:
   1. Passthrough  all features

**Number of Features: 10**!!!

#### Hyperparamter

- `max_depth = 12`
- `n_estimators = 100`

**Kaggle Score = 0.48412 (mean sqared log error)**

![randomforestreg_res](/Users/brittapuyn/Documents/SPICED/spiced_projects/nlpepper-student-code/03_week/project/output/randomforestreg_res.jpg)

## Conclusion:

Using a linear regression model on the dataset takes a lot of feature engineering to get a good performance due to non-linear correlations and interaction effects. The total count of features for using the poisson regression is pretty high (#945) and can maybe be reduced by further feature engineering and optimizing of hyper-parameter.

The RandomForestClassifier reaches a good performance with a very simple model and a low number of features (#10). The model could even be better with further feature engineering and optimizing of hyperparameter. The chosen `max_deph`could already be a reason for overfitting.