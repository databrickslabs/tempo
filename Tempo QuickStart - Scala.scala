// Databricks notebook source
// MAGIC %md
// MAGIC 
// MAGIC # Time Series Data 
// MAGIC 
// MAGIC The UCI ML Dataset repository has dozens of time series datasets. For this simple `tempo` tutorial, we've chosen a Human Activity Recognition [Dataset](https://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition) to show how to analyze hundreds of thousands of time series in parallel.
// MAGIC 
// MAGIC *The Heterogeneity Human Activity Recognition (HHAR) dataset from Smartphones and Smartwatches is a dataset devised to benchmark human activity recognition algorithms (classification, automatic data segmentation, sensor fusion, feature extraction, etc.) in real-world contexts; specifically, the dataset is gathered with a variety of different device models and use-scenarios, in order to reflect sensing heterogeneities to be expected in real deployments.*
// MAGIC 
// MAGIC <img src='https://github.com/databrickslabs/tempo/blob/master/Phone%20Accelerometer.png?raw=true' width=1500>'>

// COMMAND ----------

// MAGIC %run "/Shared/Vertical/Shared/tempo/Load Accelerometer Data - Databricks"

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC Build project using sbt (`sbt package` and attach tempo_2.12-0.1.jar onto the current cluster)

// COMMAND ----------

import org.apache.spark.sql.functions._

val phone_accel_df = spark.read.format("csv").option("header", "true").load("dbfs:/home/tempo/Phones_accelerometer").withColumn("event_ts", (col("Arrival_Time").cast("double")/1000).cast("timestamp")).withColumn("x", col("x").cast("double")).withColumn("y", col("y").cast("double")).withColumn("z", col("z").cast("double")).withColumn("event_ts_dbl", col("event_ts").cast("double"))

display(phone_accel_df)

// COMMAND ----------

// DBTITLE 1,Define the Time Series data frame (TSDF)
import com.databrickslabs.tempo._
val phone_accel_tsdf = TSDF(phone_accel_df, tsColumnName="event_ts", partitionColumnNames = "User")

// COMMAND ----------

// DBTITLE 1,Run a Simple Describe Statement - Note the added Global Attributes and Missing Values %
display(phone_accel_tsdf.describe())

// COMMAND ----------

val watch_accel_df = spark.read.format("csv").option("header", "true").load("dbfs:/home/tempo/Watch_accelerometer").withColumn("event_ts", (col("Arrival_Time").cast("double")/1000).cast("timestamp")).withColumn("x", col("x").cast("double")).withColumn("y", col("y").cast("double")).withColumn("z", col("z").cast("double")).withColumn("event_ts_dbl", col("event_ts").cast("double"))

display(watch_accel_df)

// COMMAND ----------

val watch_accel_tsdf = TSDF(watch_accel_df, tsColumnName="event_ts", partitionColumnNames = "User")
display(watch_accel_tsdf.describe())

// COMMAND ----------

// DBTITLE 1,Compute AS OF Join to Merge Last Observation from Phone Data to Watch Data
// MAGIC %python
// MAGIC joined_df = watch_accel_tsdf.asofJoin(phone_accel_tsdf, right_prefix="watch_accel").df
// MAGIC display(joined_df)

// COMMAND ----------

// DBTITLE 1,Tempo also has a specialized SKEW AS OF Join when partitions are very big
val joined_df = watch_accel_tsdf.asofJoin(phone_accel_tsdf, leftPrefix = "", rightPrefix="watch_accel", tsPartitionVal = 10, fraction = 0.1).df
display(joined_df)

// COMMAND ----------

// DBTITLE 1,VWAP
val vwap_tsdf = phone_accel_tsdf.vwap(frequency="D", volume_col="x", price_col="y")
display(vwap_tsdf.df)

// COMMAND ----------


