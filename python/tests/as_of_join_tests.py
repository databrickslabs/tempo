import unittest

from pyspark.sql.types import *

from tempo.tsdf import TSDF

from tests.base import SparkTest


class AsOfJoinTest(SparkTest):

    def test_asof_join(self):
        """AS-OF Join with out a time-partition test"""
        leftSchema = StructType([StructField("symbol", StringType()),
                                 StructField("event_ts", StringType()),
                                 StructField("trade_pr", FloatType())])

        rightSchema = StructType([StructField("symbol", StringType()),
                                  StructField("event_ts", StringType()),
                                  StructField("bid_pr", FloatType()),
                                  StructField("ask_pr", FloatType())])

        expectedSchema = StructType([StructField("symbol", StringType()),
                                     StructField("left_event_ts", StringType()),
                                     StructField("left_trade_pr", FloatType()),
                                     StructField("right_event_ts", StringType()),
                                     StructField("right_bid_pr", FloatType()),
                                     StructField("right_ask_pr", FloatType())])
        expectedSchemaNoRightPrefix = StructType([StructField("symbol", StringType()),
                                                  StructField("left_event_ts", StringType()),
                                                  StructField("left_trade_pr", FloatType()),
                                                  StructField("event_ts", StringType()),
                                                  StructField("bid_pr", FloatType()),
                                                  StructField("ask_pr", FloatType())])

        left_data = [["S1", "2020-08-01 00:00:10", 349.21],
                     ["S1", "2020-08-01 00:01:12", 351.32],
                     ["S1", "2020-09-01 00:02:10", 361.1],
                     ["S1", "2020-09-01 00:19:12", 362.1]]

        right_data = [["S1", "2020-08-01 00:00:01", 345.11, 351.12],
                      ["S1", "2020-08-01 00:01:05", 348.10, 353.13],
                      ["S1", "2020-09-01 00:02:01", 358.93, 365.12],
                      ["S1", "2020-09-01 00:15:01", 359.21, 365.31]]

        expected_data = [
            ["S1", "2020-08-01 00:00:10", 349.21, "2020-08-01 00:00:01", 345.11, 351.12],
            ["S1", "2020-08-01 00:01:12", 351.32, "2020-08-01 00:01:05", 348.10, 353.13],
            ["S1", "2020-09-01 00:02:10", 361.1, "2020-09-01 00:02:01", 358.93, 365.12],
            ["S1", "2020-09-01 00:19:12", 362.1, "2020-09-01 00:15:01", 359.21, 365.31]]

        # Construct dataframes
        dfLeft = self.buildTestDF(leftSchema, left_data)
        dfRight = self.buildTestDF(rightSchema, right_data)
        dfExpected = self.buildTestDF(expectedSchema, expected_data, ["left_event_ts", "right_event_ts"])

        # perform the join
        tsdf_left = TSDF(dfLeft, ts_col="event_ts", partition_cols=["symbol"])
        tsdf_right = TSDF(dfRight, ts_col="event_ts", partition_cols=["symbol"])

        joined_df = tsdf_left.asofJoin(tsdf_right, left_prefix="left", right_prefix="right").df
        non_prefix_joined_df = tsdf_left.asofJoin(tsdf_right, left_prefix="left", right_prefix = '').df

        # joined dataframe should equal the expected dataframe
        self.assertDataFramesEqual(joined_df, dfExpected)

        noRightPrefixdfExpected = self.buildTestDF(expectedSchemaNoRightPrefix, expected_data, ["left_event_ts", "event_ts"])

        self.assertDataFramesEqual(non_prefix_joined_df, noRightPrefixdfExpected)

        spark_sql_joined_df = tsdf_left.asofJoin(tsdf_right, left_prefix="left", right_prefix="right").df
        self.assertDataFramesEqual(spark_sql_joined_df, dfExpected)

    def test_asof_join_skip_nulls_disabled(self):
        """AS-OF Join with skip nulls disabled"""
        leftSchema = StructType([StructField("symbol", StringType()),
                                 StructField("event_ts", StringType()),
                                 StructField("trade_pr", FloatType())])

        rightSchema = StructType([StructField("symbol", StringType()),
                                  StructField("event_ts", StringType()),
                                  StructField("bid_pr", FloatType()),
                                  StructField("ask_pr", FloatType())])

        expectedSchema = StructType([StructField("symbol", StringType()),
                                     StructField("left_event_ts", StringType()),
                                     StructField("left_trade_pr", FloatType()),
                                     StructField("right_event_ts", StringType()),
                                     StructField("right_bid_pr", FloatType()),
                                     StructField("right_ask_pr", FloatType())])

        left_data = [["S1", "2020-08-01 00:00:10", 349.21],
                     ["S1", "2020-08-01 00:01:12", 351.32],
                     ["S1", "2020-09-01 00:02:10", 361.1],
                     ["S1", "2020-09-01 00:19:12", 362.1]]

        right_data = [["S1", "2020-08-01 00:00:01", 345.11, 351.12],
                      ["S1", "2020-08-01 00:01:05", None, 353.13],
                      ["S1", "2020-09-01 00:02:01", None, None],
                      ["S1", "2020-09-01 00:15:01", 359.21, 365.31]]

        expected_data_skip_nulls = [
            ["S1", "2020-08-01 00:00:10", 349.21, "2020-08-01 00:00:01", 345.11, 351.12],
            ["S1", "2020-08-01 00:01:12", 351.32, "2020-08-01 00:01:05", 345.11, 353.13],
            ["S1", "2020-09-01 00:02:10", 361.1, "2020-09-01 00:02:01", 345.11, 353.13],
            ["S1", "2020-09-01 00:19:12", 362.1, "2020-09-01 00:15:01", 359.21, 365.31]]

        expected_data_skip_nulls_disabled = [
            ["S1", "2020-08-01 00:00:10", 349.21, "2020-08-01 00:00:01", 345.11, 351.12],
            ["S1", "2020-08-01 00:01:12", 351.32, "2020-08-01 00:01:05", None, 353.13],
            ["S1", "2020-09-01 00:02:10", 361.1, "2020-09-01 00:02:01", None, None],
            ["S1", "2020-09-01 00:19:12", 362.1, "2020-09-01 00:15:01", 359.21, 365.31]]

        # Construct dataframes
        dfLeft = self.buildTestDF(leftSchema, left_data)
        dfRight = self.buildTestDF(rightSchema, right_data)
        dfExpectedSkipNulls = self.buildTestDF(
            expectedSchema, expected_data_skip_nulls, ["left_event_ts", "right_event_ts"]
        )
        dfExpectedSkipNullsDisabled = self.buildTestDF(
            expectedSchema, expected_data_skip_nulls_disabled, ["left_event_ts", "right_event_ts"]
        )

        tsdf_left = TSDF(dfLeft, ts_col="event_ts", partition_cols=["symbol"])
        tsdf_right = TSDF(dfRight, ts_col="event_ts", partition_cols=["symbol"])

        # perform the join with skip nulls enabled (default)
        joined_df = tsdf_left.asofJoin(tsdf_right, left_prefix="left", right_prefix="right").df

        # joined dataframe should equal the expected dataframe with nulls skipped
        self.assertDataFramesEqual(joined_df, dfExpectedSkipNulls)

        # perform the join with skip nulls disabled
        joined_df = tsdf_left.asofJoin(tsdf_right, left_prefix="left", right_prefix="right", skipNulls=False).df

        # joined dataframe should equal the expected dataframe without nulls skipped
        self.assertDataFramesEqual(joined_df, dfExpectedSkipNullsDisabled)

    def test_sequence_number_sort(self):
        """Skew AS-OF Join with Partition Window Test"""
        leftSchema = StructType([StructField("symbol", StringType()),
                                 StructField("event_ts", StringType()),
                                 StructField("trade_pr", FloatType()),
                                 StructField("trade_id", IntegerType())])

        rightSchema = StructType([StructField("symbol", StringType()),
                                  StructField("event_ts", StringType()),
                                  StructField("bid_pr", FloatType()),
                                  StructField("ask_pr", FloatType()),
                                  StructField("seq_nb", LongType())])

        expectedSchema = StructType([StructField("symbol", StringType()),
                                     StructField("event_ts", StringType()),
                                     StructField("trade_pr", FloatType()),
                                     StructField("trade_id", IntegerType()),
                                     StructField("right_event_ts", StringType()),
                                     StructField("right_bid_pr", FloatType()),
                                     StructField("right_ask_pr", FloatType()),
                                     StructField("right_seq_nb", LongType())])

        left_data = [["S1", "2020-08-01 00:00:10", 349.21, 1],
                     ["S1", "2020-08-01 00:00:10", 350.21, 5],
                     ["S1", "2020-08-01 00:01:12", 351.32, 2],
                     ["S1", "2020-09-01 00:02:10", 361.1, 3],
                     ["S1", "2020-09-01 00:19:12", 362.1, 4]]

        right_data = [["S1", "2020-08-01 00:00:01", 345.11, 351.12, 1],
                      ["S1", "2020-08-01 00:00:10", 19.11, 20.12, 1],
                      ["S1", "2020-08-01 00:01:05", 348.10, 1000.13, 3],
                      ["S1", "2020-08-01 00:01:05", 348.10, 100.13, 2],
                      ["S1", "2020-09-01 00:02:01", 358.93, 365.12, 4],
                      ["S1", "2020-09-01 00:15:01", 359.21, 365.31, 5]]

        expected_data = [
            ["S1", "2020-08-01 00:00:10", 349.21, 1, "2020-08-01 00:00:10", 19.11, 20.12, 1],
            ["S1", "2020-08-01 00:00:10", 350.21, 5, "2020-08-01 00:00:10", 19.11, 20.12, 1],
            ["S1", "2020-08-01 00:01:12", 351.32, 2, "2020-08-01 00:01:05", 348.10, 1000.13, 3],
            ["S1", "2020-09-01 00:02:10", 361.1, 3, "2020-09-01 00:02:01", 358.93, 365.12, 4],
            ["S1", "2020-09-01 00:19:12", 362.1, 4, "2020-09-01 00:15:01", 359.21, 365.31, 5]]

        # construct dataframes
        dfLeft = self.buildTestDF(leftSchema, left_data)
        dfRight = self.buildTestDF(rightSchema, right_data)
        dfExpected = self.buildTestDF(expectedSchema, expected_data, ["right_event_ts", "event_ts"])

        # perform the join
        tsdf_left = TSDF(dfLeft, partition_cols=["symbol"])
        tsdf_right = TSDF(dfRight, partition_cols=["symbol"], sequence_col="seq_nb")
        joined_df = tsdf_left.asofJoin(tsdf_right, right_prefix='right').df

        # joined dataframe should equal the expected dataframe
        self.assertDataFramesEqual(joined_df, dfExpected)

    def test_partitioned_asof_join(self):
        """AS-OF Join with a time-partition"""
        leftSchema = StructType([StructField("symbol", StringType()),
                                 StructField("event_ts", StringType()),
                                 StructField("trade_pr", FloatType())])

        rightSchema = StructType([StructField("symbol", StringType()),
                                  StructField("event_ts", StringType()),
                                  StructField("bid_pr", FloatType()),
                                  StructField("ask_pr", FloatType())])

        expectedSchema = StructType([StructField("symbol", StringType()),
                                     StructField("left_event_ts", StringType()),
                                     StructField("left_trade_pr", FloatType()),
                                     StructField("right_event_ts", StringType()),
                                     StructField("right_bid_pr", FloatType()),
                                     StructField("right_ask_pr", FloatType())])

        left_data = [["S1", "2020-08-01 00:00:02", 349.21],
                     ["S1", "2020-08-01 00:00:08", 351.32],
                     ["S1", "2020-08-01 00:00:11", 361.12],
                     ["S1", "2020-08-01 00:00:18", 364.31],
                     ["S1", "2020-08-01 00:00:19", 362.94],
                     ["S1", "2020-08-01 00:00:21", 364.27],
                     ["S1", "2020-08-01 00:00:23", 367.36]]

        right_data = [["S1", "2020-08-01 00:00:01", 345.11, 351.12],
                      ["S1", "2020-08-01 00:00:09", 348.10, 353.13],
                      ["S1", "2020-08-01 00:00:12", 358.93, 365.12],
                      ["S1", "2020-08-01 00:00:19", 359.21, 365.31]]

        expected_data = [
            ["S1", "2020-08-01 00:00:02", 349.21, "2020-08-01 00:00:01", 345.11, 351.12],
            ["S1", "2020-08-01 00:00:08", 351.32, "2020-08-01 00:00:01", 345.11, 351.12],
            ["S1", "2020-08-01 00:00:11", 361.12, "2020-08-01 00:00:09", 348.10, 353.13],
            ["S1", "2020-08-01 00:00:18", 364.31, "2020-08-01 00:00:12", 358.93, 365.12],
            ["S1", "2020-08-01 00:00:19", 362.94, "2020-08-01 00:00:19", 359.21, 365.31],
            ["S1", "2020-08-01 00:00:21", 364.27, "2020-08-01 00:00:19", 359.21, 365.31],
            ["S1", "2020-08-01 00:00:23", 367.36, "2020-08-01 00:00:19", 359.21, 365.31]]

        # Construct dataframes
        dfLeft = self.buildTestDF(leftSchema, left_data)
        dfRight = self.buildTestDF(rightSchema, right_data)
        dfExpected = self.buildTestDF(expectedSchema, expected_data, ["left_event_ts", "right_event_ts"])

        tsdf_left = TSDF(dfLeft, ts_col="event_ts", partition_cols=["symbol"])
        tsdf_right = TSDF(dfRight, ts_col="event_ts", partition_cols=["symbol"])

        joined_df = tsdf_left.asofJoin(tsdf_right, left_prefix="left", right_prefix="right",
                                       tsPartitionVal=10, fraction=0.1).df

        self.assertDataFramesEqual(joined_df, dfExpected)

    def test_asof_join_nanos(self):
        """As of join with nanosecond timestamps"""
        leftSchema = StructType([StructField("symbol", StringType()),
                                 StructField("event_ts", StringType()),
                                 StructField("trade_pr", FloatType())])

        rightSchema = StructType([StructField("symbol", StringType()),
                                  StructField("event_ts", StringType()),
                                  StructField("bid_pr", FloatType()),
                                  StructField("ask_pr", FloatType())])

        expectedSchema = StructType([StructField("symbol", StringType()),
                                     StructField("left_event_ts", StringType()),
                                     StructField("left_trade_pr", FloatType()),
                                     StructField("right_event_ts", StringType()),
                                     StructField("right_ask_pr", FloatType()),
                                     StructField("right_bid_pr", FloatType())]
                                    )

        left_data = [["S1", "2022-01-01 09:59:59.123456789", 349.21],
                     ["S1", "2022-01-01 10:00:00.123456788", 351.32],
                     ["S1", "2022-01-01 10:00:00.123456789", 361.12],
                     ["S1", "2022-01-01 10:00:01.123456789", 364.31], ]

        right_data = [["S1", "2022-01-01 10:00:00.1234567", 345.11, 351.12],
                      ["S1", "2022-01-01 10:00:00.12345671", 348.10, 353.13],
                      ["S1", "2022-01-01 10:00:00.12345675", 358.93, 365.12],
                      ["S1", "2022-01-01 10:00:00.12345677", 358.91, 365.33],
                      ["S1", "2022-01-01 10:00:01.10000001", 359.21, 365.31]]

        expected_data = [
            ["S1", "2022-01-01 09:59:59.123456789", 349.21, None, None, None],
            ["S1", "2022-01-01 10:00:00.123456788", 351.32, "2022-01-01 10:00:00.12345677", 365.33, 358.91],
            ["S1", "2022-01-01 10:00:00.123456789", 361.12, "2022-01-01 10:00:00.12345677", 365.33, 358.91],
            ["S1", "2022-01-01 10:00:01.123456789", 364.31, "2022-01-01 10:00:01.10000001", 365.31, 359.21]]

        dfLeft = self.buildTestDF(leftSchema, left_data)
        dfRight = self.buildTestDF(rightSchema, right_data)
        dfExpected = self.buildTestDF(expectedSchema, expected_data, ts_cols=["left_event_ts"])

        tsdf_left = TSDF(dfLeft, ts_col="event_ts", partition_cols=["symbol"])
        tsdf_right = TSDF(dfRight, ts_col="event_ts", partition_cols=["symbol"])

        joined_df = tsdf_left.asofJoin(tsdf_right, left_prefix="left", right_prefix="right").df

        self.assertDataFramesEqual(joined_df, dfExpected)



# MAIN
if __name__ == '__main__':
    unittest.main()
