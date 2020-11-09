import tempo.tsdf as ts

def write(tsdf, tabName, optimizationCols):
  """
  param: tsdf: input TSDF object to write
  param: tabName Delta output table name
  param: optimizationCols list of columns to optimize on (time)
  """
  # hilbert curves more evenly distribute performance for querying multiple columns for Delta tables
  spark.conf.set("spark.databricks.io.skipping.mdc.curve", "hilbert")

  df = self.df
  ts_col = self.ts_col
  partitionCols = self.partitionCols
  optimizationCols = optimizationCols + ['event_time']
  local = False
  try:
      dbutils.fs.ls("/")
      local = True
  except:
      print('Running in local mode')
      pass

  format = "parquet" if local else "delta"
  view_df = df.withColumn("event_dt", f.to_date(col(ts_col))) \
      .withColumn("event_time", f.translate(f.split(f.col(ts_col).cast("string"), ' ')[1], ':', '').cast("double"))
  view_df.write.mode("overwrite").partitionBy("event_dt").format(format).saveAsTable(tab_name)

  if not local:
      spark.sql("optimize {} zorder by {}".format(tab_name, "(" + ",".join(partitionCols + optimizationCols) + ")"))