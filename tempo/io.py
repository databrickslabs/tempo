#import tempo.tsdf as ts
import pyspark.sql.functions as f

def write(tsdf, spark, tabName, optimizationCols = None):
  """
  param: tsdf: input TSDF object to write
  param: tabName Delta output table name
  param: optimizationCols list of columns to optimize on (time)
  """
  # hilbert curves more evenly distribute performance for querying multiple columns for Delta tables
  spark.conf.set("spark.databricks.io.skipping.mdc.curve", "hilbert")

  df = tsdf.df
  ts_col = tsdf.ts_col
  partitionCols = tsdf.partitionCols
  if optimizationCols:
     optimizationCols = optimizationCols + ['event_time']
  else:
     optimizationCols = ['event_time']

  useDeltaOpt = True
  try:
      dbutils.fs.ls("/")
  except:
      print('Running in local mode')
      useDeltaOpt = False
      pass

  view_df = df.withColumn("event_dt", f.to_date(f.col(ts_col))) \
      .withColumn("event_time", f.translate(f.split(f.col(ts_col).cast("string"), ' ')[1], ':', '').cast("double"))
  view_df.write.mode("overwrite").partitionBy("event_dt").format('delta').saveAsTable(tabName)

  if useDeltaOpt:
      try:
         spark.sql("optimize {} zorder by {}".format(tabName, "(" + ",".join(partitionCols + optimizationCols) + ")"))
      except:
         print("Delta optimizations attempted on a non-Databricks platform. Switch to use Databricks Runtime to get optimization advantages.")
