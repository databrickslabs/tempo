from pyspark.sql.functions import *


class deltaSource:

    # helper methods for writing existing Spark data frame to Delta Lake for time series optimization
    def __init__(self, df, tab_name="delta_ts", ts_col="EVENT_TS", partitionCols=[], optimizationCols = []):
        self.df = df
        self.ts_col = ts_col
        self.partitionCols = partitionCols
        self.optimizationCols = optimizationCols

    def write(self, tab_name, optimization_cols=[]):
        df = self.df
        ts_col = self.ts_col
        partitionCols = self.partitionCols
        optimizationCols = self.optimizationCols + ['event_time']
        local = False
        try:
           dbutils.fs.ls("/")
           True
        except:
           'Running in local mode'

        format = "parquet" if local else "delta"
        view_df = df.withColumn("event_dt", to_date(col(ts_col))) \
                    .withColumn("event_time", translate(split(col(ts_col).cast("string"), ' ')[1], ':', '').cast("double"))
        view_df.write.mode("overwrite").partitionBy("event_dt").format(format).saveAsTable(tab_name)

        if not local:
           spark.sql("optimize {} zorder by {}".format(tab_name, "(" + ",".join(partitionCols + optimizationCols) + ")"))