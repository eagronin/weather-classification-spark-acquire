# Data Acquisition

This section describes and imports the dataset **daily_weather.csv** to analyze weather patterns in San Diego, CA.  Specifically, we will build a decision tree for predicting low humidity days, which increase the risk of wildfires.

The [next section](https://eagronin.github.io/weather-classification-spark-prepare/) explores and cleans the data.

The file **daily_weather.csv** was downloaded from the Coursera website and saved on Cloudera cloud.

The  is a comma-separated file that contains weather data. This data comes from a weather station located in San Diego, CA. The weather station is equipped with sensors that capture weather-related measurements such as air temperature, air pressure, and relative humidity. Data was collected for a period of three years, from September 2011 to September 2014, to ensure that sufficient data for different seasons and weather conditions is captured.

Sensor measurements from the weather station were captured at one-minute intervals. These measurements were then processed to generate values to describe daily weather. Since this dataset was created to classify low-humidity days vs. non-low-humidity days (that is, days with normal or high humidity), the variables included are weather measurements in the morning, with one measurement, namely relatively humidity, in the afternoon. The idea is to use the morning weather values to predict whether the day will be low-humidity or not based on the afternoon measurement of relatively humidity.

Each row in daily_weather.csv captures weather data for a separate day. Each row consists of the following variables:

| Variable | Description | Unit of Measure |
| --- | --- | --- |
| number | unique number for each row | NA |
| air_pressure_9am | air pressure averaged over a period from 8:50am to 9:10am | hectopascals |
| air_temp_9am | air temperature averaged over a period from 8:50am to 9:10am | degrees Fahrenheit |
| avg_wind_direction_9am | wind direction averaged over a period from 8:50am to 9:10am | degrees, with 0 means coming from the North, and increasing clockwise |
| avg_wind_speed_9am | wind speed averaged over a period from 8:50am to 9:10am | miles per hour |
| max_wind_directon_9am | wind gust direction averaged over a period from 8:50am to 9:10am | degrees, with 0 being North and increasing clockwise |
| max_wind_speed_9am | wind gust speed averaged over a period from 8:50am to 9:10am | miles per hour |
| rain_accumulation_9am | amount of accumulated rain averaged over a period from 8:50am to 9:10am | millimeters |
| rain_duration_9am | amount of time raining averaged over a period from 8:50am to 9:10am | seconds |
| relative_humidity_9am | relative humidity averaged over a period from 8:50am to 9:10am | percent |
| relative_humidity_3pm | relative humidity averaged over a period from 2:50pm to 3:10pm | percent |

The following code imports **daily_weather.csv** from a folder on the cloud:

```
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

df = sqlContext.read.load('file:///home/cloudera/Downloads/big-data-4/daily_weather.csv', 
                          format='com.databricks.spark.csv', 
                          header='true',inferSchema='true')
```

Nest step: [Data Preparation](https://eagronin.github.io/weather-classification-spark-prepare/)

# Data Preparation

This section explores and cleans the data to prepare it for the analysis of weather patterns in San Diego, CA.  Specifically, in the [next section](https://eagronin.github.io/weather-classification-spark-analyze/) we will build a decision tree for predicting low humidity days, which increase the risk of wildfires.

The dataset is described and imported in the [previous section](https://eagronin.github.io/weather-classification-spark-acquire/)

The analysis is described in the [next section](https://eagronin.github.io/weather-classification-spark-analyze/)

## Data Exploration
The **daily_weather.csv** dataset has 11 features and 1,095 rows (or samples) as can be verified using `len(df.columns)` and `df.count()` commands, respectively.

To explore the dataset, we output feature names using `df.columns`, the data type for each feature using `df.printSchema()` and summary statistics using `df.describe().toPandas().transpose()`.  The output is as follows:

Feature names

```
['number',
 'air_pressure_9am',
 'air_temp_9am',
 'avg_wind_direction_9am',
 'avg_wind_speed_9am',
 'max_wind_direction_9am',
 'max_wind_speed_9am',
 'rain_accumulation_9am',
 'rain_duration_9am',
 'relative_humidity_9am',
 'relative_humidity_3pm']
```

Data type for each feature:

```
root
 |-- number: integer (nullable = true)
 |-- air_pressure_9am: double (nullable = true)
 |-- air_temp_9am: double (nullable = true)
 |-- avg_wind_direction_9am: double (nullable = true)
 |-- avg_wind_speed_9am: double (nullable = true)
 |-- max_wind_direction_9am: double (nullable = true)
 |-- max_wind_speed_9am: double (nullable = true)
 |-- rain_accumulation_9am: double (nullable = true)
 |-- rain_duration_9am: double (nullable = true)
 |-- relative_humidity_9am: double (nullable = true)
 |-- relative_humidity_3pm: double (nullable = true)
```

Summary statistics

| summary |	count |	mean |	stddev |	min |	max |
| --- | ---| --- | --- | ---| --- |
| number |	1095 |	547.0 |	316.24 |	0 |	1094 |
| air_pressure_9am |	1092 |	918.88 |	3.18 |	907.99 |	929.32 |
| air_temp_9am |	1090 |	64.93 |	11.17 |	36.75 |	98.90 |
| avg_wind_direction_9am |	1091 |	142.23 |	69.13 |	15.50 |	343.4 |
| avg_wind_speed_9am |	1092 |	5.50 |	4.55 |	0.693 |	23.55 |
| max_wind_direction_9am |	1092 |	148.95 |	67.23 |	28.89 |	312.19 |
| max_wind_speed_9am |	1091 |	7.01 |	5.59 |	1.18 |	29.84 |
| rain_accumulation_9am |	1089 |	0.20 |	1.59 |	0.0 |	24.01 |
| rain_duration_9am |	1092 |	294.10 |	1598.07 |	0.0 |	17704.0 |
| relative_humidity_9am |	1095 |	34.24 |	25.47 |	6.09 |	92.62 |
| relative_humidity_3pm |	1095 |	35.34 |	22.52 |	5.30 |	92.25 |

## Handling Missing Values
The summary statistics table indicates that some of the features have less than 1,095 rows (the total number of rows in the dataset).  

For example, air_temp_9am has only 1,090 rows:  

```
df.describe('air_temp_9am').show()
```

|summary|      air_temp_9am|
| --- | --- |
|  count|              1090|
|   mean| 64.93300141287075|
| stddev|11.175514003175877|
|    min|36.752000000000685|
|    max| 98.90599999999992|


This means that five rows in air_temp_9am have missing values.

We can drop all the rows missing a value in any feature as follows:

```
removeAllDF.count()
```

This leaves us with 1,064 rows in dataframe removeAllDF, as can be verified using `removeAllDF.count()`.  

Looking just at the statistics for the air temperature at 9am:

```
removeAllDF.describe('air_temp_9am').show()
```

|summary|      air_temp_9am|
| --- | --- |
|  count|              1064|
|   mean| 65.02260949558739|
| stddev|11.168033449415699|
|    min|36.752000000000685|
|    max| 98.90599999999992|


After the number of observations for air_temp_9am declined from 1,090 to 1,064, the mean and standard deviation of this feature are still close the original values: mean is 64.933 vs. 65.022, and standard deviation is 11.175 vs. 11.168.

Alternatively, we can replace missing values in each feature with the mean value for that feature:

```
from pyspark.sql.functions import avg

imputeDF = df

for x in imputeDF.columns:
    meanValue = removeAllDF.agg(avg(x)).first()[0]
    print(x, meanValue)
    imputeDF = imputeDF.na.fill(meanValue, [x])
```

This code produces the following output of the mean values for each feature:

```
number 545.00
air_pressure_9am 918.90
air_temp_9am 65.02
avg_wind_direction_9am 142.30
avg_wind_speed_9am 5.48
max_wind_direction_9am 148.48
max_wind_speed_9am 6.99
rain_accumulation_9am 0.18
rain_duration_9am 266.39
relative_humidity_9am 34.07
relative_humidity_3pm 35.14
```

The summary statistics for air_temp_9am are now as follows:

```
imputeDF.describe('air_temp_9am').show()
```

|summary|      air_temp_9am|
| --- | --- |
|  count|              1095|
|   mean| 64.93341058219822|
| stddev|11.149948199920226|
|    min|36.752000000000685|
|    max| 98.90599999999992|

The number of rows in air_temp_9am is now 1,095 (increased from 1,090) which means that the feature no longer has missing values.


