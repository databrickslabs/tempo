import pyspark.sql.functions as fn


# convert event timestamp to string
# convert event timestamp to spit out date and time double
#
def withStringTS(tsdf):
     res = df.withColumn(col(tsdf.ts_col).cast("string")
     return(res)
