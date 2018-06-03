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


