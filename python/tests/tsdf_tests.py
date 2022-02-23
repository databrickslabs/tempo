import logging
import unittest

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import *

from tempo.tsdf import TSDF
from tempo.utils import *

class SparkTest(unittest.TestCase):
    ##
    ## Fixtures
    ##
    def setUp(self):
        self.spark = (SparkSession.builder.appName("myapp") \
                      .config("spark.jars.packages", "io.delta:delta-core_2.12:1.1.0") \
                      .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
                      .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
                      .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true") \
                      .config("spark.executor.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true") \
                      .master("local") \
                      .getOrCreate())
        self.spark.conf.set("spark.sql.shuffle.partitions", 1)

    def tearDown(self) -> None:
        self.spark.stop()

    ##
    ## Utility Functions
    ##

    def buildTestDF(self, schema, data, ts_cols=["event_ts"]):
        """
        Constructs a Spark Dataframe from the given components
        :param schema: the schema to use for the Dataframe
        :param data: values to use for the Dataframe
        :param ts_cols: list of column names to be converted to Timestamp values
        :return: a Spark Dataframe, constructed from the given schema and values
        """
        # build dataframe
        df = self.spark.createDataFrame(data, schema)

        # convert all timestamp columns
        for tsc in ts_cols:
            df = df.withColumn(tsc, F.to_timestamp(F.col(tsc)))

        return df

    ##
    ## DataFrame Assert Functions
    ##

    def assertFieldsEqual(self, fieldA, fieldB):
        """
        Test that two fields are equivalent
        """
        self.assertEqual(fieldA.name.lower(), fieldB.name.lower())
        self.assertEqual(fieldA.dataType, fieldB.dataType)
        # self.assertEqual(fieldA.nullable, fieldB.nullable)

    def assertSchemaContainsField(self, schema, field):
        """
        Test that the given schema contains the given field
        """
        # the schema must contain a field with the right name
        lc_fieldNames = [fc.lower() for fc in schema.fieldNames()]
        self.assertTrue(field.name.lower() in lc_fieldNames)
        # the attributes of the fields must be equal
        self.assertFieldsEqual(field, schema[field.name])

    def assertSchemasEqual(self, schemaA, schemaB):
        """
        Test that the two given schemas are equivalent (column ordering ignored)
        """
        # both schemas must have the same length
        self.assertEqual(len(schemaA.fields), len(schemaB.fields))
        # schemaA must contain every field in schemaB
        for field in schemaB.fields:
            self.assertSchemaContainsField(schemaA, field)

    def assertHasSchema(self, df, expectedSchema):
        """
        Test that the given Dataframe conforms to the expected schema
        """
        self.assertSchemasEqual(df.schema, expectedSchema)

    def assertDataFramesEqual(self, dfA, dfB):
        """
        Test that the two given Dataframes are equivalent.
        That is, they have equivalent schemas, and both contain the same values
        """
        # must have the same schemas
        self.assertSchemasEqual(dfA.schema, dfB.schema)
        # enforce a common column ordering
        colOrder = sorted(dfA.columns)
        sortedA = dfA.select(colOrder)
        sortedB = dfB.select(colOrder)
        # must have identical data
        # that is all rows in A must be in B, and vice-versa

        self.assertEqual(sortedA.subtract(sortedB).count(), 0)
        self.assertEqual(sortedB.subtract(sortedA).count(), 0)


class BasicTests(SparkTest):

    def test_describe(self):
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
        res = tsdf_left.describe()

        # joined dataframe should equal the expected dataframe
        # self.assertDataFramesEqual(res, dfExpected)
        assert res.count() == 7
        assert res.filter(F.col("unique_ts_count") != " ").select(F.max(F.col('unique_ts_count'))).collect()[0][
                   0] == "1"
        assert res.filter(F.col("min_ts") != " ").select(F.col('min_ts').cast("string")).collect()[0][
                   0] == '2020-08-01 00:00:10'
        assert res.filter(F.col("max_ts") != " ").select(F.col('max_ts').cast("string")).collect()[0][
                   0] == '2020-09-01 00:19:12'


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
                     ["S1", "2020-08-01 00:01:12", 351.32, 2],
                     ["S1", "2020-09-01 00:02:10", 361.1, 3],
                     ["S1", "2020-09-01 00:19:12", 362.1, 4]]

        right_data = [["S1", "2020-08-01 00:00:01", 345.11, 351.12, 1],
                      ["S1", "2020-08-01 00:01:05", 348.10, 1000.13, 3],
                      ["S1", "2020-08-01 00:01:05", 348.10, 100.13, 2],
                      ["S1", "2020-09-01 00:02:01", 358.93, 365.12, 4],
                      ["S1", "2020-09-01 00:15:01", 359.21, 365.31, 5]]

        expected_data = [
            ["S1", "2020-08-01 00:00:10", 349.21, 1, "2020-08-01 00:00:01", 345.11, 351.12, 1],
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


class FourierTransformTest(SparkTest):

    def test_fourier_transform(self):
        """Test of fourier transform functionality in TSDF objects"""
        schema = StructType([StructField("group",StringType()),
                             StructField("time",LongType()),
                             StructField("val",DoubleType())])

        expectedSchema = StructType([StructField("group",StringType()),
                                     StructField("time",LongType()),
                                     StructField("val",DoubleType()),
                                     StructField("freq",DoubleType()),
                                     StructField("ft_real",DoubleType()),
                                     StructField("ft_imag",DoubleType())])

        data = [["Emissions", 1949, 2206.690829],
                ["Emissions", 1950, 2382.046176],
                ["Emissions", 1951, 2526.687327],
                ["Emissions", 1952, 2473.373964],
                ["WindGen", 1980, 0.0],
                ["WindGen", 1981, 0.0],
                ["WindGen", 1982, 0.0],
                ["WindGen", 1983, 0.029667962]]

        expected_data = [["Emissions", 1949, 2206.690829, 0.0, 9588.798296, -0.0],
                         ["Emissions", 1950, 2382.046176, 0.25, -319.996498, 91.32778800000006],
                         ["Emissions", 1951, 2526.687327, -0.5, -122.0419839999995, -0.0],
                         ["Emissions", 1952, 2473.373964, -0.25, -319.996498, -91.32778800000006],
                         ["WindGen", 1980, 0.0, 0.0, 0.029667962, -0.0],
                         ["WindGen", 1981, 0.0, 0.25, 0.0, 0.029667962],
                         ["WindGen", 1982, 0.0, -0.5, -0.029667962, -0.0],
                         ["WindGen", 1983, 0.029667962, -0.25, 0.0, -0.029667962]]

        # construct dataframes
        df = self.buildTestDF(schema, data, ts_cols=['time'])
        dfExpected = self.buildTestDF(expectedSchema, expected_data, ts_cols=['time'])

        # convert to TSDF
        tsdf_left = TSDF(df, ts_col="time", partition_cols=["group"])
        result_tsdf = tsdf_left.fourier_transform(1, 'val')

        # should be equal to the expected dataframe
        self.assertDataFramesEqual(result_tsdf.df, dfExpected)


class RangeStatsTest(SparkTest):

    def test_range_stats(self):
        """Test of range stats for 20 minute rolling window"""
        schema = StructType([StructField("symbol", StringType()),
                             StructField("event_ts", StringType()),
                             StructField("trade_pr", FloatType())])

        expectedSchema = StructType([StructField("symbol", StringType()),
                                     StructField("event_ts", StringType()),
                                     StructField("mean_trade_pr", FloatType()),
                                     StructField("count_trade_pr", LongType(), nullable=False),
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

        # construct dataframes
        df = self.buildTestDF(schema, data)
        dfExpected = self.buildTestDF(expectedSchema, expected_data)

        # convert to TSDF
        tsdf_left = TSDF(df, partition_cols=["symbol"])

        # using lookback of 20 minutes
        featured_df = tsdf_left.withRangeStats(rangeBackWindowSecs=1200).df

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

        # should be equal to the expected dataframe
        self.assertDataFramesEqual(featured_df, dfExpected)


class UtilsTest(SparkTest):

    def test_display(self):
        """Test of the display utility"""
        if PLATFORM == 'DATABRICKS':
            self.assertEqual(id(display),id(display_improvised))
        elif ENV_BOOLEAN:
            self.assertEqual(id(display),id(display_html_improvised))
        else:
            self.assertEqual(id(display),id(display_unavailable))

class ResampleTest(SparkTest):

    def test_resample(self):
        """Test of range stats for 20 minute rolling window"""
        schema = StructType([StructField("symbol", StringType()),
                             StructField("date", StringType()),
                             StructField("event_ts", StringType()),
                             StructField("trade_pr", FloatType()),
                             StructField("trade_pr_2", FloatType())])

        expectedSchema = StructType([StructField("symbol", StringType()),
                                     StructField("event_ts", StringType()),
                                     StructField("floor_trade_pr", FloatType()),
                                     StructField("floor_date", StringType()),
                                     StructField("floor_trade_pr_2", FloatType())])

        expected_30m_Schema = StructType([StructField("symbol", StringType()),
                                          StructField("event_ts", StringType(), True),
                                          StructField("date", DoubleType()),
                                          StructField("trade_pr", DoubleType()),
                                          StructField("trade_pr_2", DoubleType())])

        expectedBarsSchema = StructType([StructField("symbol", StringType()),
                                         StructField("event_ts", StringType()),
                                         StructField("close_trade_pr", FloatType()),
                                         StructField("close_trade_pr_2", FloatType()),
                                         StructField("high_trade_pr", FloatType()),
                                         StructField("high_trade_pr_2", FloatType()),
                                         StructField("low_trade_pr", FloatType()),
                                         StructField("low_trade_pr_2", FloatType()),
                                         StructField("open_trade_pr", FloatType()),
                                         StructField("open_trade_pr_2", FloatType())])

        data = [["S1", "SAME_DT", "2020-08-01 00:00:10", 349.21, 10.0],
                ["S1", "SAME_DT", "2020-08-01 00:00:11", 340.21, 9.0],
                ["S1", "SAME_DT", "2020-08-01 00:01:12", 353.32, 8.0],
                ["S1", "SAME_DT", "2020-08-01 00:01:13", 351.32, 7.0],
                ["S1", "SAME_DT", "2020-08-01 00:01:14", 350.32, 6.0],
                ["S1", "SAME_DT", "2020-09-01 00:01:12", 361.1, 5.0],
                ["S1", "SAME_DT", "2020-09-01 00:19:12", 362.1, 4.0]]

        expected_data = [
            ["S1", "2020-08-01 00:00:00", 349.21, "SAME_DT", 10.0],
            ["S1", "2020-08-01 00:01:00", 353.32, "SAME_DT", 8.0],
            ["S1", "2020-09-01 00:01:00", 361.1, "SAME_DT", 5.0],
            ["S1", "2020-09-01 00:19:00", 362.1, "SAME_DT", 4.0]]

        expected_data_30m = [
            ["S1", "2020-08-01 00:00:00", None, 348.88, 8.0],
            ["S1", "2020-09-01 00:00:00", None, 361.1, 5.0],
            ["S1", "2020-09-01 00:15:00", None, 362.1, 4.0]
        ]

        expected_bars = [
            ['S1', '2020-08-01 00:00:00', 340.21, 9.0, 349.21, 10.0, 340.21, 9.0, 349.21, 10.0],
            ['S1', '2020-08-01 00:01:00', 350.32, 6.0, 353.32, 8.0, 350.32, 6.0, 353.32, 8.0],
            ['S1', '2020-09-01 00:01:00', 361.1, 5.0, 361.1, 5.0, 361.1, 5.0, 361.1, 5.0],
            ['S1', '2020-09-01 00:19:00', 362.1, 4.0, 362.1, 4.0, 362.1, 4.0, 362.1, 4.0]
        ]

        # construct dataframes
        df = self.buildTestDF(schema, data)
        dfExpected = self.buildTestDF(expectedSchema, expected_data)
        expected_30s_df = self.buildTestDF(expected_30m_Schema, expected_data_30m)
        barsExpected = self.buildTestDF(expectedBarsSchema, expected_bars)

        # convert to TSDF
        tsdf_left = TSDF(df, partition_cols=["symbol"])

        # 1 minute aggregation
        featured_df = tsdf_left.resample(freq="min", func="floor", prefix='floor').df
        # 30 minute aggregation
        resample_30m = tsdf_left.resample(freq="5 minutes", func="mean").df.withColumn("trade_pr",
                                                                                       F.round(F.col('trade_pr'), 2))

        bars = tsdf_left.calc_bars(freq='min', metricCols=['trade_pr', 'trade_pr_2']).df

        # should be equal to the expected dataframe
        self.assertDataFramesEqual(featured_df, dfExpected)
        self.assertDataFramesEqual(resample_30m, expected_30s_df)

        # test bars summary
        self.assertDataFramesEqual(bars, barsExpected)

    def test_upsample(self):
        """Test of range stats for 20 minute rolling window"""
        schema = StructType([StructField("symbol", StringType()),
                             StructField("date", StringType()),
                             StructField("event_ts", StringType()),
                             StructField("trade_pr", FloatType()),
                             StructField("trade_pr_2", FloatType())])

        expectedSchema = StructType([StructField("symbol", StringType()),
                                     StructField("event_ts", StringType()),
                                     StructField("floor_trade_pr", FloatType()),
                                     StructField("floor_date", StringType()),
                                     StructField("floor_trade_pr_2", FloatType())])

        expected_30m_Schema = StructType([StructField("symbol", StringType()),
                                          StructField("event_ts", StringType()),
                                          StructField("date", DoubleType()),
                                          StructField("trade_pr", DoubleType()),
                                          StructField("trade_pr_2", DoubleType())])

        expectedBarsSchema = StructType([StructField("symbol", StringType()),
                                         StructField("event_ts", StringType()),
                                         StructField("close_trade_pr", FloatType()),
                                         StructField("close_trade_pr_2", FloatType()),
                                         StructField("high_trade_pr", FloatType()),
                                         StructField("high_trade_pr_2", FloatType()),
                                         StructField("low_trade_pr", FloatType()),
                                         StructField("low_trade_pr_2", FloatType()),
                                         StructField("open_trade_pr", FloatType()),
                                         StructField("open_trade_pr_2", FloatType())])

        data = [["S1", "SAME_DT", "2020-08-01 00:00:10", 349.21, 10.0],
                ["S1", "SAME_DT", "2020-08-01 00:00:11", 340.21, 9.0],
                ["S1", "SAME_DT", "2020-08-01 00:01:12", 353.32, 8.0],
                ["S1", "SAME_DT", "2020-08-01 00:01:13", 351.32, 7.0],
                ["S1", "SAME_DT", "2020-08-01 00:01:14", 350.32, 6.0],
                ["S1", "SAME_DT", "2020-09-01 00:01:12", 361.1, 5.0],
                ["S1", "SAME_DT", "2020-09-01 00:19:12", 362.1, 4.0]]

        expected_data = [
            ["S1", "2020-08-01 00:00:00", 349.21, "SAME_DT", 10.0],
            ["S1", "2020-08-01 00:01:00", 353.32, "SAME_DT", 8.0],
            ["S1", "2020-09-01 00:01:00", 361.1, "SAME_DT", 5.0],
            ["S1", "2020-09-01 00:19:00", 362.1, "SAME_DT", 4.0]]

        expected_data_30m = [
            ["S1", "2020-08-01 00:00:00", 0.0, 348.88, 8.0],
            ["S1", "2020-08-01 00:05:00", 0.0, 0.0, 0.0],
            ["S1", "2020-09-01 00:00:00", 0.0, 361.1, 5.0],
            ["S1", "2020-09-01 00:15:00", 0.0, 362.1, 4.0]
        ]

        expected_bars = [
            ['S1', '2020-08-01 00:00:00', 340.21, 9.0, 349.21, 10.0, 340.21, 9.0, 349.21, 10.0],
            ['S1', '2020-08-01 00:01:00', 350.32, 6.0, 353.32, 8.0, 350.32, 6.0, 353.32, 8.0],
            ['S1', '2020-09-01 00:01:00', 361.1, 5.0, 361.1, 5.0, 361.1, 5.0, 361.1, 5.0],
            ['S1', '2020-09-01 00:19:00', 362.1, 4.0, 362.1, 4.0, 362.1, 4.0, 362.1, 4.0]
        ]

        # construct dataframes
        df = self.buildTestDF(schema, data)
        dfExpected = self.buildTestDF(expectedSchema, expected_data)
        expected_30s_df = self.buildTestDF(expected_30m_Schema, expected_data_30m)
        barsExpected = self.buildTestDF(expectedBarsSchema, expected_bars)

        # convert to TSDF
        tsdf_left = TSDF(df, partition_cols=["symbol"])

        resample_30m = tsdf_left.resample(freq="5 minutes", func="mean", fill=True).df.withColumn("trade_pr", F.round(
            F.col('trade_pr'), 2))

        bars = tsdf_left.calc_bars(freq='min', metricCols=['trade_pr', 'trade_pr_2']).df

        upsampled = resample_30m.filter(
            F.col("event_ts").isin('2020-08-01 00:00:00', '2020-08-01 00:05:00', '2020-09-01 00:00:00',
                                   '2020-09-01 00:15:00'))
        self.assertDataFramesEqual(upsampled, expected_30s_df)

        # test bars summary
        self.assertDataFramesEqual(bars, barsExpected)


class DeltaWriteTest(SparkTest):

    def test_write_to_delta(self):
        """Test of range stats for 20 minute rolling window"""
        schema = StructType([StructField("symbol", StringType()),
                             StructField("date", StringType()),
                             StructField("event_ts", StringType()),
                             StructField("trade_pr", FloatType()),
                             StructField("trade_pr_2", FloatType())])

        expectedSchema = StructType([StructField("symbol", StringType()),
                                     StructField("event_ts", StringType()),
                                     StructField("date", StringType()),
                                     StructField("trade_pr_2", FloatType()),
                                     StructField("trade_pr", FloatType())])

        data = [["S1", "SAME_DT", "2020-08-01 00:00:10", 349.21, 10.0],
                ["S1", "SAME_DT", "2020-08-01 00:00:11", 340.21, 9.0],
                ["S1", "SAME_DT", "2020-08-01 00:01:12", 353.32, 8.0],
                ["S1", "SAME_DT", "2020-08-01 00:01:13", 351.32, 7.0],
                ["S1", "SAME_DT", "2020-08-01 00:01:14", 350.32, 6.0],
                ["S1", "SAME_DT", "2020-09-01 00:01:12", 361.1, 5.0],
                ["S1", "SAME_DT", "2020-09-01 00:19:12", 362.1, 4.0]]

        expected_data = [
            ["S1", "2020-08-01 00:00:00", "SAME_DT", 10.0, 349.21],
            ["S1", "2020-08-01 00:01:00", "SAME_DT", 8.0, 353.32],
            ["S1", "2020-09-01 00:01:00", "SAME_DT", 5.0, 361.1],
            ["S1", "2020-09-01 00:19:00", "SAME_DT", 4.0, 362.1]]

        # construct dataframes
        df = self.buildTestDF(schema, data)
        dfExpected = self.buildTestDF(expectedSchema, expected_data)

        # convert to TSDF
        tsdf_left = TSDF(df, partition_cols=["symbol"])

        # using lookback of 20 minutes
        # featured_df = tsdf_left.resample(freq = "min", func = "closest_lead").df
        tsdf_left.write(self.spark, "my_table")
        logging.info('delta table count ' + str(self.spark.table("my_table").count()))

        # should be equal to the expected dataframe
        assert self.spark.table("my_table").count() == 7
        # self.assertDataFramesEqual(tsdf_left.df, tsdf_left.df)


## MAIN
if __name__ == '__main__':
    unittest.main()
