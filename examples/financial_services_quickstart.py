# Databricks notebook source
# MAGIC %md ## Scaling Financial Time Series with Tempo
# MAGIC
# MAGIC #### Prerequisites:
# MAGIC
# MAGIC * Use an ML Cluster with DBR version `10.0` or later
# MAGIC
# MAGIC ### Use Cases for Tempo
# MAGIC
# MAGIC * AS OF Joins
# MAGIC * Resampling
# MAGIC * Bar calculations
# MAGIC * Spread Analytics
# MAGIC * Best execution (slippage)
# MAGIC * Market Manipulation
# MAGIC * Order Execution Timeliness
# MAGIC
# MAGIC <img src = 'https://databricks.com/wp-content/uploads/2021/01/image-1.png' width = 1000>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 1. Merging and Scaling Analyses with Delta Lake and Apache Spark

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC wget https://pages.databricks.com/rs/094-YMS-629/images/ASOF_Quotes.csv ; wget https://pages.databricks.com/rs/094-YMS-629/images/ASOF_Trades.csv ;

# COMMAND ----------

# MAGIC %fs cp file:/databricks/driver/ASOF_Quotes.csv /tmp/finserv/ASOF_Quotes.csv

# COMMAND ----------

# MAGIC %fs cp file:/databricks/driver/ASOF_Trades.csv /tmp/finserv/ASOF_Trades.csv

# COMMAND ----------

# DBTITLE 1,Tempo is available on PyPI
# MAGIC %pip install dbl-tempo==0.1.12

# COMMAND ----------

from tempo import *

# COMMAND ----------

# MAGIC %sql drop table if exists tempo.quotes; drop table if exists tempo.trades;

# COMMAND ----------

# DBTITLE 1,Convert Existing CSV Tick Data Sources to Delta Lake for Backtesting and TCA Use Cases
from pyspark.sql.types import *
from pyspark.sql.functions import *

trade_schema = StructType([
    StructField("symbol", StringType()),
    StructField("event_ts", TimestampType()),
    StructField("trade_dt", StringType()),
    StructField("trade_pr", DoubleType()),
    StructField("trade_qt", IntegerType()),
    StructField("date", TimestampType())
])

quote_schema = StructType([
    StructField("symbol", StringType()),
    StructField("event_ts", TimestampType()),
    StructField("trade_dt", StringType()),
    StructField("bid_pr", DoubleType()),
    StructField("ask_pr", DoubleType()),
    StructField("date", TimestampType())
])

spark.read.format("csv").schema(trade_schema).option("header", "true").option("delimiter", ",").load("/tmp/finserv/ASOF_Trades.csv").withColumn("trade_qt", lit(100)).withColumn("date", col("event_ts").cast("date")).write.mode('overwrite').option("overwriteSchema", "true").saveAsTable('tempo.trades')

trades_df = spark.table("tempo.trades")

spark.read.format("csv").schema(quote_schema).option("header", "true").option("delimiter", ",").load("/tmp/finserv/ASOF_Quotes.csv").withColumn("date", col("event_ts").cast("date")).write.mode('overwrite').option("overwriteSchema", "true").saveAsTable('tempo.quotes')

quotes_df = spark.table("tempo.quotes")

# COMMAND ----------

trades_df = spark.table("tempo.trades")
quotes_df = spark.table("tempo.quotes")

# COMMAND ----------

# DBTITLE 1,Define TSDF Time Series Data Structure
from tempo import *

trades_tsdf = TSDF(trades_df, partition_cols = ['date', 'symbol'], ts_col = 'event_ts')
quotes_tsdf = TSDF(quotes_df, partition_cols = ['date', 'symbol'], ts_col = 'event_ts')

# COMMAND ----------

# DBTITLE 1,Use Data Preview Features with TSDF Directly
display(trades_tsdf.df)

# COMMAND ----------

# DBTITLE 1,Resample TSDF for Visualization Purposes
from pyspark.sql.functions import *

portfolio = ['NVIDIA', 'ROKU', 'MU', 'AAPL', 'AMZN', 'FB', 'MSFT', 'INTL']

resampled_sdf = trades_tsdf.resample(freq='min', func='floor')
resampled_pdf = resampled_sdf.df.filter(col('event_ts').cast("date") == "2017-08-31").filter(col("symbol").isNotNull()).filter(col("symbol").isin(portfolio)).toPandas()

import plotly.graph_objs as go
import plotly.express as px
import pandas as pd

# Plotly figure 1
fig = px.line(resampled_pdf, x='event_ts', y='trade_pr',
              color="symbol",
              line_group="symbol", hover_name = "symbol")
fig.update_layout(title='Daily Trade Information' , showlegend=False)

fig.show()

# COMMAND ----------

# DBTITLE 1,AS OF Joins - Get Latest Quote Information As Of Time of Trades
joined_df = trades_tsdf.asofJoin(quotes_tsdf, right_prefix="quote_asof").df

display(joined_df.filter(col("symbol") == 'AMH').filter(col("quote_asof_event_ts").isNotNull()))

# COMMAND ----------

import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)
logging.getLogger("tempo").setLevel(logging.WARNING)

joined_df = trades_tsdf.asofJoin(quotes_tsdf, tsPartitionVal=30,right_prefix="quote_asof").df
display(joined_df)

# COMMAND ----------

# DBTITLE 1,For Delta Optimization, Tempo Offers a Writer
trades_tsdf.write(spark, "tempo.silver_delta_trades")

# COMMAND ----------

# DBTITLE 1,Take Advantage of Analytics Format for Range Queries
# MAGIC %sql
# MAGIC
# MAGIC select * from tempo.silver_delta_trades
# MAGIC where symbol = 'MU'
# MAGIC -- newly computed column for filtering!
# MAGIC and event_time between 103400 and 103500

# COMMAND ----------

# DBTITLE 1,Simple Moving Average with Tempo
moving_avg = trades_tsdf.withRangeStats("trade_pr", rangeBackWindowSecs=600).df
output = moving_avg.select('symbol', 'event_ts', 'trade_pr', 'mean_trade_pr', 'stddev_trade_pr', 'sum_trade_pr', 'min_trade_pr')

display(output.filter(col("symbol") == 'AMD'))

# COMMAND ----------

# DBTITLE 1,Exponential Moving Average with Tempo
ema_trades = trades_tsdf.EMA("trade_pr", window = 50).df
display(ema_trades.filter(col("symbol") == 'AMD'))

# COMMAND ----------

display(moving_avg.withColumn("low", col("min_trade_pr")).withColumn("high", col("max_trade_pr")))

# COMMAND ----------

# DBTITLE 1,Produce Minute Bars Using Tempo
from tempo import *
from pyspark.sql.functions import *

minute_bars = TSDF(spark.table("time_test"), partition_cols=['ticker'], ts_col="ts").calc_bars(freq = '1 minute', func= 'ceil')

display(minute_bars)

# COMMAND ----------

# DBTITLE 1,Save tables for Use in Databricks SQL
minute_bars.df.write.mode('overwrite').option("mergeSchema", "true").saveAsTable("tempo.gold_bars_minute")

# COMMAND ----------

moving_avg.write.mode('overwrite').option("mergeSchema", "true").saveAsTable('tempo.gold_sma_10min')

# COMMAND ----------

display(minute_bars.df)

# COMMAND ----------


