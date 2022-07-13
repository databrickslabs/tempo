# Databricks notebook source
# MAGIC %pip install dbl-tempo

# COMMAND ----------

from tempo import *
import dlt
from pyspark.sql.functions import *
from pyspark.sql.types import *

# COMMAND ----------

@dlt.table()
def ts_bronze():
    phone_accel_df = (spark.read.format("csv").option("header", "true").load("dbfs:/home/tempo/Phones_accelerometer").withColumn("event_ts", (col("Arrival_Time").cast("double")/1000).cast("timestamp")).withColumn("x", col("x").cast("double")).withColumn("y", col("y").cast("double")).withColumn("z", col("z").cast("double")).withColumn("event_ts_dbl", col("event_ts").cast("double")))
    return phone_accel_df

# COMMAND ----------

@dlt.table(comment= "Fourier Transform of the Input time series dataframe",
           table_properties={
              'quality':'silver',
              'pipelines.autoOptimize.zOrderCols' : 'freq'
           }
          )
@dlt.expect_or_drop("User_check","User in ('a','c','i')")
def ts_ft():
  phone_accel_df = dlt.read("ts_bronze")
  phone_accel_tsdf = TSDF(phone_accel_df, ts_col="event_ts", partition_cols = ["User"])
  ts_ft_df = phone_accel_tsdf.fourier_transform(timestep=1, valueCol="x").df
  return ts_ft_df

# COMMAND ----------


