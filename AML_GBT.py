# Databricks notebook source
import logging
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.classification import LogisticRegression, GBTClassifier, RandomForestClassifier, LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from time import time
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.storagelevel import StorageLevel
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# COMMAND ----------

PYSPARK_CLI = True
if PYSPARK_CLI:
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)

# COMMAND ----------

# File location and type
file_location = "/user/sshah82/AML/HI_Medium_Trans.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

launderingSchema = StructType([
StructField("TimeStamp", StringType(), False),
StructField("From_Bank", IntegerType(), False),
StructField("From_acc", StringType(), False),
StructField("To_Bank", IntegerType(), False),
StructField("To_acc", StringType(), False),
StructField("Amount_Received", FloatType(), False),
StructField("Receiving_Currency", StringType(), False),
StructField("Amount_Paid", FloatType(), False),
StructField("Payment_Currency", StringType(), False),
StructField("Payment_Format", StringType(), False),
StructField("Is_Laundering", IntegerType(), False)
])

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
.schema(launderingSchema) \
.option("header", first_row_is_header) \
.option("sep", delimiter) \
.load(file_location)

print(df)

# COMMAND ----------

#Display the schema
df.printSchema()

# COMMAND ----------

#checking the dataset count
print(df.count())
Laundering_df = df.filter(col("Is_Laundering") == 1)
print(Laundering_df.count())
test_df = df.filter(col("Is_Laundering") == 0)
print(test_df.count())

# COMMAND ----------

#Feature Engineering
indexer1 = StringIndexer(inputCol="Receiving_Currency", outputCol="Receiving_CurrencyIndex")
indexer2 = StringIndexer(inputCol="Payment_Currency", outputCol="Payment_CurrencyIndex")
indexer3 = StringIndexer(inputCol="Payment_Format", outputCol="Payment_FormatIndex")


df = indexer1.fit(df).transform(df)
df = indexer2.fit(df).transform(df)
df = indexer3.fit(df).transform(df)

print(df)

# COMMAND ----------

#Final DataFrame for Model
df3 = df.select("From_Bank", "To_Bank", "Amount_Received", "Receiving_CurrencyIndex", "Amount_Paid", "Payment_CurrencyIndex", "Payment_FormatIndex", col("Is_Laundering").alias("label"))
df3.show(2)

# COMMAND ----------

#Balancing by oversampling minortity class
Laundering_df2 = df3.filter(col("label") == 1)
print(Laundering_df2.count())
test_df2 = df3.filter(col("label") == 0)
print(test_df2.count())
balanced_ratio = test_df2.count() / Laundering_df2.count()
print(balanced_ratio)
oversampled_df2 = Laundering_df2.sample(withReplacement=True, fraction=balanced_ratio)
print(oversampled_df2.count())
df3 = test_df2.union(oversampled_df2)
print(df3.count())

# COMMAND ----------

# Add the VectorAssembler stage to combine features
assembler = VectorAssembler(inputCols = ["From_Bank", "To_Bank", "Amount_Received", "Receiving_CurrencyIndex", "Amount_Paid", "Payment_CurrencyIndex", "Payment_FormatIndex"], outputCol="features")

# COMMAND ----------

splits = df3.randomSplit([0.7,0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")
print ("Training Rows:", train.count(), " Testing Rows:", test.count())

# COMMAND ----------

gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=10)

# COMMAND ----------

# Combine stages into pipeline 
pipeline = Pipeline(stages=[assembler, gbt])

# COMMAND ----------

# Hyperparameter Tuning 
paramGrid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [3, 5]) \
    .addGrid(gbt.maxIter, [10, 20]) \
    .build()

# COMMAND ----------

# Create a TrainValidator
tv = TrainValidationSplit(estimator=pipeline, evaluator=BinaryClassificationEvaluator(), estimatorParamMaps=paramGrid, trainRatio=0.8)

# COMMAND ----------

#Training the model and Calculating its time
import time

# Start time
start_time = time.time() 

tvModel = tv.fit(train)

# End time
end_time = time.time()

print("Model trained!")
# Calculate training time
training_time = end_time - start_time

# Calculate minutes and seconds
minutes = int(training_time // 60)
seconds = int(training_time % 60)

logging.info("Training time: %02d:%02d" % (minutes, seconds))

# COMMAND ----------

prediction = tvModel.transform(test)
predicted = prediction.select("features", "prediction", "probability", "trueLabel")

predicted.show(100, truncate=False)

# COMMAND ----------

tp = float(predicted.filter("prediction == 1.0 AND truelabel == 1").count())
fp = float(predicted.filter("prediction == 1.0 AND truelabel == 0").count())
tn = float(predicted.filter("prediction == 0.0 AND truelabel == 0").count())
fn = float(predicted.filter("prediction == 0.0 AND truelabel == 1").count())
metrics = spark.createDataFrame([
 ("TP", tp),
 ("FP", fp),
 ("TN", tn),
 ("FN", fn),
 ("Precision", tp / (tp + fp)),
 ("Recall", tp / (tp + fn))],["metric", "value"])
 
metrics.show()

logging.info("***********TrainValidator-Results-GBT************")

# COMMAND ----------

evaluator = BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = evaluator.evaluate(prediction)
print("AUC = ", auc)

# COMMAND ----------

# Get the best model from TrainValidationSplit
best_model = tvModel.bestModel
gbtModel = best_model.stages[-1]

# Feature importance
import pandas as pd
featureImp = pd.DataFrame(list(zip(assembler.getInputCols(), gbtModel.featureImportances)), columns=["feature", "importance"]) 
featureImp = featureImp.sort_values(by="importance", ascending=False)

# Show feature importance
print(featureImp)

# COMMAND ----------

# Create a CrossValidator
cv = CrossValidator(estimator=pipeline, evaluator=BinaryClassificationEvaluator(), estimatorParamMaps=paramGrid, numFolds=3)

# COMMAND ----------

import time
# Start time
start_time = time.time()

# Fit the model with cross-validation on the training set
cvModel = cv.fit(train)

# End time
end_time = time.time()

print("Model trained!")

# Calculate training time
training_time = end_time - start_time


# Calculate minutes and seconds
minutes = int(training_time // 60)
seconds = int(training_time % 60)

logging.info("Training time: %02d:%02d" % (minutes, seconds))

# COMMAND ----------

prediction = cvModel.transform(test)
predicted = prediction.select("features", "prediction", "probability", "trueLabel")

predicted.show(100, truncate=False)

# COMMAND ----------

tp = float(predicted.filter("prediction == 1.0 AND truelabel == 1").count())
fp = float(predicted.filter("prediction == 1.0 AND truelabel == 0").count())
tn = float(predicted.filter("prediction == 0.0 AND truelabel == 0").count())
fn = float(predicted.filter("prediction == 0.0 AND truelabel == 1").count())
metrics = spark.createDataFrame([
 ("TP", tp),
 ("FP", fp),
 ("TN", tn),
 ("FN", fn),
 ("Precision", tp / (tp + fp)),
 ("Recall", tp / (tp + fn))],["metric", "value"])
 
metrics.show()

logging.info("***********CrossValidator-Results-GBT************")

# COMMAND ----------

evaluator = BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = evaluator.evaluate(prediction)
print("AUC = ", auc)

# COMMAND ----------

# Get the best model from CrossValidationSplit
best_model = cvModel.bestModel
gbtModel = best_model.stages[-1]

# Feature importance
import pandas as pd
featureImp = pd.DataFrame(list(zip(assembler.getInputCols(), gbtModel.featureImportances)), columns=["feature", "importance"]) 
featureImp = featureImp.sort_values(by="importance", ascending=False)

# Show feature importance
print(featureImp)