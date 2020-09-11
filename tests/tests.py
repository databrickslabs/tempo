from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F
from tempo.tsdf import TSDF

# Create spark session
spark = (SparkSession.builder
         .master("local")
         .getOrCreate())

def test_asof_join():
    from tempo.tsdf import TSDF

    leftSchema = StructType([StructField("symbol", StringType()),
                             StructField("event_ts", StringType()),
                             StructField("trade_pr", FloatType())])

    rightSchema = StructType([StructField("symbol", StringType()),
                              StructField("event_ts", StringType()),
                              StructField("bid_pr", FloatType()),
                              StructField("ask_pr", FloatType())])

    expectedSchema = StructType([StructField("symbol", StringType()),
                                 StructField("EVENT_TS_left", StringType()),
                                 StructField("trade_pr", FloatType()),
                                 StructField("EVENT_TS_right", StringType()),
                                 StructField("bid_pr", FloatType()),
                                 StructField("ask_pr", FloatType())])

    left_data = [["S1", "2020-08-01 00:00:10", 349.21],
                 ["S1", "2020-08-01 00:01:12", 351.32],
                 ["S1", "2020-09-01 00:02:10", 361.1],
                 ["S1", "2020-09-01 00:19:12", 362.1]]

    right_data = [["S1", "2020-08-01 00:00:01",  345.11, 351.12],
                  ["S1", "2020-08-01 00:01:05",  348.10, 353.13],
                  ["S1", "2020-09-01 00:02:01",  358.93, 365.12],
                  ["S1", "2020-09-01 00:15:01",  359.21, 365.31]]

    expected_data = [
        ["S1", "2020-08-01 00:00:10", 349.21, "2020-08-01 00:00:01",  345.11, 351.12],
        ["S1", "2020-08-01 00:01:12", 351.32, "2020-08-01 00:01:05",  348.10, 353.13],
        ["S1", "2020-09-01 00:02:10", 361.1, "2020-09-01 00:02:01",  358.93, 365.12],
        ["S1", "2020-09-01 00:19:12", 362.1, "2020-09-01 00:15:01",  359.21, 365.31]]


    dfLeft = (spark.createDataFrame(left_data,leftSchema)
              .withColumn("event_ts", F.to_timestamp(F.col("event_ts"))))

    dfRight = (spark.createDataFrame(right_data, rightSchema)
               .withColumn("event_ts", F.to_timestamp(F.col("event_ts"))))

    dfExpected = (spark.createDataFrame(expected_data, expectedSchema)
                  .withColumn("EVENT_TS_right", F.to_timestamp(F.col("EVENT_TS_right")))
                  .withColumn("EVENT_TS_left", F.to_timestamp(F.col("EVENT_TS_left"))))

    tsdf_left = TSDF(dfLeft)
    joined_df = tsdf_left.asofJoin(dfRight, partitionCols=["symbol"])

    assert (joined_df.select(sorted(joined_df.columns)).subtract(dfExpected.select(sorted(dfExpected.columns))).count() ==
            dfExpected.select(sorted(dfExpected.columns)).subtract(joined_df.select(sorted(joined_df.columns))).count() == 0)

test_asof_join()