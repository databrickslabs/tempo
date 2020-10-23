from pyspark.sql.functions import *


class optimized_df:

    # range partitioner - can we build something like this?
    # hash partitioning on partitioning cols
    # TODO - finish up Delta ingestion
    def __init__(self, df, tab_name="delta_ts", ts_col="EVENT_TS", partition_cols=[], optimization_cols=[]):
        self.df = df
        self.ts_col = ts_col
        __convertDF(self.df, ts_col, optimization_cols)

    def __convertDF(self, df, optimization_cols=[]):
        view_df = df.withColumn("event_dt", to_date(col(ts_col)))
          .withColumn("event_time", translate(split(col(ts_col).cast("string"), ' ')[1], ':', '').cast("double"))
        view_df.write.partitionBy(partition_cols).format("delta").saveAsTable("delta_" + tab_name)
        spark.sql("optimize delta_{} zorder by {}".format(tab_name, "(" + ",".join(optimization_cols) + ")"))