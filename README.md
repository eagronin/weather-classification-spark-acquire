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

```python
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

```python
df.describe('air_temp_9am').show()
```

|summary|      air_temp_9am|
| --- | --- |
|  count|              1090|
|   mean| 64.93|
| stddev|11.17|
|    min|36.75|
|    max| 98.90|


This means that five rows in air_temp_9am have missing values.

We can drop all the rows missing a value in any feature as follows:

```python
removeAllDF.count()
```

This leaves us with 1,064 rows in dataframe removeAllDF, as can be verified using `removeAllDF.count()`.  

Looking just at the statistics for the air temperature at 9am:

```python
removeAllDF.describe('air_temp_9am').show()
```

|summary|      air_temp_9am|
| --- | --- |
|  count|              1064|
|   mean| 65.02|
| stddev|11.16|
|    min|36.75|
|    max| 98.90|


After the number of observations for air_temp_9am declined from 1,090 to 1,064, the mean and standard deviation of this feature are still close the original values: mean is 64.933 vs. 65.022, and standard deviation is 11.175 vs. 11.168.

Alternatively, we can replace missing values in each feature with the mean value for that feature:

```python
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

```python
imputeDF.describe('air_temp_9am').show()
```

|summary|      air_temp_9am|
| --- | --- |
|  count|              1095|
|   mean| 64.93|
| stddev|11.14|
|    min|36.75|
|    max| 98.90|

The number of rows in air_temp_9am is now 1,095 (increased from 1,090) which means that the feature no longer has missing values.

In the analysis performed in the next section, we will use the version of the data in which all the missing values have been dropped:

```python
df = removeAllDF
```

Next step: [Analysis](https://eagronin.github.io/weather-classification-spark-analyze/)

# Analysis

This section describes the analysis of weather patterns in San Diego, CA.  Specifically, we build and evaluate the performance of a decision tree for predicting low humidity days.  Such low humidity days increase the risk of wildfires and, therefore, predicting such days is important for providing a timely warning to the residents and appropriate authorities.

Exploration and cleaning of the data are discussed in the [previous section](https://eagronin.github.io/weather-classification-spark-prepare/)

The results are discussed in the [next section](https://eagronin.github.io/weather-classification-spark-report/)

## Training a Decision Tree Classifier

The following code defines a dataframe with the features used for the decision tree classifier.  It then create the target, a categorical variable to denote if the humidity is not low. If the value is less than 25%, then the categorical value is 0, otherwise the categorical value is 1.  Finally, the code aggregate the features used to make predictions into a single column using `VectorAssembler` and partition the data into training and test data: 

```python
featureColumns = ['air_pressure_9am','air_temp_9am','avg_wind_direction_9am','avg_wind_speed_9am',
        'max_wind_direction_9am','max_wind_speed_9am','rain_accumulation_9am',
        'rain_duration_9am']

binarizer = Binarizer(threshold = 24.99999, inputCol = "relative_humidity_3pm", outputCol="label")
binarizedDF = binarizer.transform(df)

assembler = VectorAssembler(inputCols = featureColumns, outputCol = "features")
assembled = assembler.transform(binarizedDF)
(trainingData, testData) = assembled.randomSplit([.7,.3], seed = 13234)
```

Next, we careate and train a decision tree:

```python
dt = DecisionTreeClassifier(labelCol = "label", featuresCol = "features", maxDepth = 5, minInstancesPerNode = 20, impurity = "gini")
pipeline = Pipeline(stages = [dt])
model = pipeline.fit(trainingData)
```

Let's make predictions for the test data and compare the target (or label) with its prediction for the first 20 rows of the test dataset:

```python
predictions = model.transform(testData)
predictions.select("prediction", "label").show(20)
```

| prediction | label |
| --- | --- |
|       1.0|  1.0|
|       1.0|  1.0|
|       1.0|  1.0|
|       1.0|  1.0|
|       1.0|  1.0|
|       1.0|  1.0|
|       1.0|  1.0|
|       0.0|  0.0|
|       0.0|  0.0|
|       1.0|  1.0|
|       1.0|  1.0|
|       1.0|  1.0|
|       0.0|  0.0|
|       1.0|  1.0|
|       0.0|  1.0|
|       1.0|  1.0|
|       1.0|  1.0|
|       1.0|  0.0|
|       1.0|  1.0|
|       0.0|  0.0|

The output shows that out of the first 20 target values 18 values are predicted corretly.  

The following code saves the predictions, which are subsequently used for model evaluation:

```python
predictions.select("prediction", "label").coalesce(1).write.save(path = "file:///home/cloudera/Downloads/big-data-4/predictions",
                                                    format = "com.databricks.spark.csv",
                                                    header = 'true')
```

## Evaluation of a Decision Tree Classifier

The following code evaluates the performance of the decision tree:

```
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

predictions = sqlContext.read.load('file:///home/cloudera/Downloads/big-data-4/predictions', 
                          format='com.databricks.spark.csv', 
                          header='true',inferSchema='true')
evaluator = MulticlassClassificationEvaluator(
    labelCol = "label",predictionCol = "prediction", metricName = "precision")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % (accuracy))
```

The accuracy of the decision tree perormance using test data is 0.75.

Next, we generate and output the confusion matrix.  

The MulticlassMetrics class can be used to generate a confusion matrix of the classifier above. However, unlike MulticlassClassificationEvaluator, MulticlassMetrics works with RDDs of numbers and not DataFrames, so we need to convert our predictions DataFrame into an RDD.

If we use the RDD attribute of predictions, we see this is an RDD of Rows: `predictions.rdd.take(2)` outputs `[Row(prediction=1.0, label=1.0), Row(prediction=1.0, label=1.0)]`.

Instead, we can map the RDD to tuple to get an RDD of numbers: `predictions.rdd.map(tuple).take(2)` outputs `[(1.0, 1.0), (1.0, 1.0)]`.  The following code then generates the confusion matrix for the decision tree classifier:

```
from pyspark.mllib.evaluation import MulticlassMetrics

metrics = MulticlassMetrics(predictions.rdd.map(tuple))
metrics.confusionMatrix().toArray().transpose()
```

This results in the following confusion matrix:

```
array([[ 134.,   53.],
       [  29.,  118.]])
```
