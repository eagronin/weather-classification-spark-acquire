# Data Acquisition

This section describes and imports the dataset **daily_weather.csv** to analyze weather patterns in San Diego, CA.  Specifically, we will build a decision tree for predicting low humidity days, which increase the risk of wildfires.

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

Then we output feature names, the data type for each feature and summary statistics:

```
df.columns
df.printSchema()
df.describe().toPandas().transpose()
```

The output is as follows:

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

```
| summary |	count |	mean |	stddev |	min |	max |
| --- | ---| --- | --- | ---| --- |
| number |	1095 |	547.0 |	316.24357700987383 |	0 |	1094 |
| air_pressure_9am |	1092 |	918.8825513138097 |	3.1841611803868353 |	907.9900000000024 |	929.3200000000012 |
| air_temp_9am |	1090 |	64.93300141287075 |	11.175514003175877 |	36.752000000000685 |	98.90599999999992 |
| avg_wind_direction_9am |	1091 |	142.23551070057584 |	69.13785928889183 |	15.500000000000046 |	343.4 |
| avg_wind_speed_9am |	1092 |	5.50828424225493 |	4.552813465531715 |	0.69345139999974 |	23.554978199999763 |
| max_wind_direction_9am |	1092 |	148.9535179651692 |	67.23801294602951 |	28.89999999999991 |	312.19999999999993 |
| max_wind_speed_9am |	1091 |	7.019513529175272 |	5.59820917078096 |	1.1855782000000479 |	29.84077959999996 |
| rain_accumulation_9am |	1089 |	0.20307895225211126 |	1.5939521253574904 |	0.0 |	24.01999999999907 |
| rain_duration_9am |	1092 |	294.1080522756142 |	1598.078778660148 |	0.0 |	17704.0 |
| relative_humidity_9am |	1095 |	34.24140205923539 |	25.472066802250044 |	6.090000000001012 |	92.6200000000002 |
| relative_humidity_3pm |	1095 |	35.34472714825902 |	22.52407945358728 |	5.3000000000006855 |	92.2500000000003 |
```



