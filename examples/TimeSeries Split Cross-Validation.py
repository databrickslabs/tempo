# Databricks notebook source
# MAGIC %md
# MAGIC # Set up dataset

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC wget -P /tmp/finserv/ https://pages.databricks.com/rs/094-YMS-629/images/ASOF_Trades.csv;

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC ls /tmp/finserv/

# COMMAND ----------

# MAGIC %sh head -n 30 /tmp/finserv/ASOF_Trades.csv

# COMMAND ----------

data_dir = "/tmp/finserv"
trade_schema = """
  symbol string,
  event_ts timestamp,
  mod_dt date,
  trade_pr double
"""

trades_df = (spark.read.format("csv")
             .schema(trade_schema)
             .option("header", "true")
             .option("delimiter", ",")
             .load(f"{data_dir}/ASOF_Trades.csv"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Prepare Data
# MAGIC
# MAGIC Aggregate trades into 15-minute aggregates

# COMMAND ----------

import pyspark.sql.functions as sfn

bars_df = (trades_df
           .where(sfn.col("symbol").isNotNull())
           .groupBy(sfn.col("symbol"), 
                    sfn.window("event_ts", "15 minutes"))
           .agg(
             sfn.first_value("trade_pr").alias("open"),
             sfn.min("trade_pr").alias("low"),
             sfn.max("trade_pr").alias("high"),
             sfn.last_value("trade_pr").alias("close"),
             sfn.count("trade_pr").alias("num_trades"))
           .select("symbol", 
                   sfn.col("window.start").alias("event_ts"),
                   "open", "high", "low", "close", "num_trades")
           .orderBy("symbol", "event_ts"))

# COMMAND ----------

display(bars_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Rolling 1.5 hour window
# MAGIC
# MAGIC Build a feature-vector by rolling the 6 previous rows (1.5 hours) into a feature vector to predict current prices.

# COMMAND ----------

from pyspark.sql import Window

six_step_win = Window.partitionBy("symbol").orderBy("event_ts").rowsBetween(-6, -1)

six_step_rolling = (bars_df
                    .withColumn("prev_open", sfn.collect_list("open").over(six_step_win))
                    .withColumn("prev_high", sfn.collect_list("high").over(six_step_win))
                    .withColumn("prev_low", sfn.collect_list("low").over(six_step_win))
                    .withColumn("prev_close", sfn.collect_list("close").over(six_step_win))
                    .withColumn("prev_n", sfn.collect_list("num_trades").over(six_step_win))
                    .where(sfn.array_size("prev_n") >= sfn.lit(6)))

# COMMAND ----------

display(six_step_rolling)

# COMMAND ----------

import pyspark.ml.functions as mlfn

features_df = (six_step_rolling
               .withColumn("features", 
                           mlfn.array_to_vector(sfn.concat("prev_open", "prev_high", 
                                                           "prev_low", "prev_close", 
                                                           "prev_n")))
               .drop("prev_open", "prev_high", "prev_low", "prev_close", "prev_n"))

# COMMAND ----------

display(features_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cross-validate a model
# MAGIC But we need to split based on time...

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

from tempo.ml import TimeSeriesCrossValidator

# parameters
params = ParamGridBuilder().build()

# set up model
target_col = "close"
gbt = GBTRegressor(labelCol=target_col, featuresCol="features")

# set up evaluator
eval = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="rmse")

# set up cross-validator
param_grid = ParamGridBuilder().build()
tscv = TimeSeriesCrossValidator(estimator=gbt, 
                                evaluator=eval,
                                estimatorParamMaps=param_grid,
                                collectSubModels=True,
                                timeSeriesCol="event_ts",
                                seriesIdCols=["symbol"])

# COMMAND ----------

import mlflow
import mlflow.spark

mlflow.spark.autolog()

with mlflow.start_run() as run:
  cvModel = tscv.fit(features_df)
  best_gbt_mdl = cvModel.bestModel
  mlflow.spark.log_model(best_gbt_mdl, "cv_model")

# COMMAND ----------


