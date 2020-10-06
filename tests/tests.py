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

    tsdf_left = TSDF(dfLeft, ts_col="event_ts", partitionCols=["symbol"])
    joined_df = tsdf_left.asofJoin(dfRight).df

    assert (joined_df.select(sorted(joined_df.columns)).subtract(dfExpected.select(sorted(dfExpected.columns))).count() ==
            dfExpected.select(sorted(dfExpected.columns)).subtract(joined_df.select(sorted(joined_df.columns))).count() == 0)

    

def test_range_stats(debug=False):
    """
    This method tests the lookback stats for one numeric column (trade_pr).
    :param - debug is used for testing only - switch to True to see data frame output
    input parameters to the stats which are unique to this test are 1200 for 20 minute lookback
    """

    schema = StructType([StructField("symbol", StringType()),
                         StructField("event_ts", StringType()),
                         StructField("trade_pr", FloatType())])

    expectedSchema = StructType([StructField("symbol", StringType()),
                                 StructField("event_ts", StringType()),
                                 StructField("mean_trade_pr", FloatType()),
                                 StructField("count_trade_pr", IntegerType()),
                                 StructField("min_trade_pr", FloatType()),
                                 StructField("max_trade_pr", FloatType()),
                                 StructField("sum_trade_pr", FloatType()),
                                 StructField("stddev_trade_pr", FloatType()),
                                 StructField("zscore_trade_pr", FloatType())])

    data = [["S1", "2020-08-01 00:00:10", 349.21],
                              ["S1", "2020-08-01 00:01:12", 351.32],
                              ["S1", "2020-09-01 00:02:10", 361.1],
                              ["S1", "2020-09-01 00:19:12", 362.1]]

    expected_data = [
            ["S1", "2020-08-01 00:00:10", 349.21, 1, 349.21, 349.21, 349.21, None, None],
            ["S1", "2020-08-01 00:01:12", 350.26, 2, 349.21, 351.32, 700.53, 1.49, 0.71],
            ["S1", "2020-09-01 00:02:10", 361.1, 1, 361.1, 361.1, 361.1, None, None],
            ["S1", "2020-09-01 00:19:12", 361.6, 2, 361.1, 362.1, 723.2, 0.71, 0.71]]

    df = (spark.createDataFrame(data, schema)
                         .withColumn("event_ts", F.to_timestamp(F.col("event_ts"))))

    dfExpected = (spark.createDataFrame(expected_data, expectedSchema)
                                     .withColumn("event_ts", F.to_timestamp(F.col("event_ts"))))

    tsdf_left = TSDF(df,ts_col="event_ts")

    # using lookback of 20 minutes
    featured_df = tsdf_left.withRangeStats(partitionCols=["symbol"], rangeBackWindowSecs=1200).df

    # cast to decimal with precision in cents for simplicity
    featured_df = featured_df.select(F.col("symbol"), F.col("event_ts"),
                                      F.col("mean_trade_pr").cast("decimal(5, 2)"),
                                      F.col("count_trade_pr"),
                                      F.col("min_trade_pr").cast("decimal(5,2)"),
                                      F.col("max_trade_pr").cast("decimal(5,2)"),
                                      F.col("sum_trade_pr").cast("decimal(5,2)"),
                                      F.col("stddev_trade_pr").cast("decimal(5,2)"),
                                      F.col("zscore_trade_pr").cast("decimal(5,2)"))

    # cast to decimal with precision in cents for simplicity
    dfExpected = dfExpected.select(F.col("symbol"), F.col("event_ts"),
                                    F.col("mean_trade_pr").cast("decimal(5, 2)"),
                                    F.col("count_trade_pr"),
                                    F.col("min_trade_pr").cast("decimal(5,2)"),
                                    F.col("max_trade_pr").cast("decimal(5,2)"),
                                    F.col("sum_trade_pr").cast("decimal(5,2)"),
                                    F.col("stddev_trade_pr").cast("decimal(5,2)"),
                                    F.col("zscore_trade_pr").cast("decimal(5,2)"))


    if debug:
      print('Test 2 - Range Features Data Frame Output:')
      featured_df.orderBy('event_ts').show(5, False)
      print("-----------")
      dfExpected.orderBy('event_ts').show(5, False)
      featured_df.subtract(dfExpected).show(5, False)


    assert (featured_df.orderBy("event_ts").subtract(dfExpected.orderBy("event_ts")).count() == 0)


test_asof_join()
test_range_stats()