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

# Analysis

This section describes the analysis of weather patterns in San Diego, CA.  Specifically, we build and evaluate the performance of a decision tree for predicting low humidity days.  Such low humidity days increase the risk of wildfires and, therefore, predicting such days is important for providing a timely warning to the residents and appropriate authorities.

Exploration and cleaning of the data are discussed in the [previous section](https://eagronin.github.io/weather-classification-spark-prepare/)

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

```python
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

```python
from pyspark.mllib.evaluation import MulticlassMetrics

metrics = MulticlassMetrics(predictions.rdd.map(tuple))
metrics.confusionMatrix().toArray().transpose()
```

This results in the following confusion matrix:

```
array([[ 134.,   53.],
       [  29.,  118.]])
```

Previous step: [Data Preparation](https://eagronin.github.io/weather-classification-spark-prepare/)
