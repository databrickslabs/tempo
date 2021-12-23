import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from python.tempo.interpol import Interpolation
from python.tests.tests import SparkTest
from chispa.dataframe_comparer import *
from tempo.tsdf import TSDF
from tempo.utils import *


class InterpolationTest(SparkTest):
    def buildTestDataFrame(self):
        """Test of range stats for 20 minute rolling window"""
        schema = StructType(
            [
                StructField("partition_a", StringType()),
                StructField("partition_b", StringType()),
                StructField("event_ts", StringType()),
                StructField("value_a", FloatType()),
                StructField("value_b", FloatType()),
            ]
        )

        data = [
            ["A", "A-1", "2020-01-01 00:01:10", 349.21, 10.0],
            ["A", "A-2", "2020-01-01 00:01:15", 340.21, 9.0],
            ["B", "B-1", "2020-01-01 00:01:15", 362.1, 4.0],
            ["A", "A-2", "2020-01-01 00:01:17", 353.32, 8.0],
            ["B", "B-2", "2020-01-01 00:02:14", 350.32, 6.0],
            ["A", "A-1", "2020-01-01 00:02:13", 351.32, 7.0],
            ["B", "B-2", "2020-01-01 00:01:12", 361.1, 5.0],
        ]

        # construct dataframes
        self.input_df = self.buildTestDF(schema, data)

        # generate TSDF
        self.input_tsdf = TSDF(
            self.input_df,
            partition_cols=["partition_a", "partition_b"],
            ts_col="event_ts",
        )

        # register interpolation helper
        self.interpolate_helper = Interpolation()

    def test_zero_fill_interpolation(self):
        self.buildTestDataFrame()

        expected_schema = StructType(
            [
                StructField("partition_a", StringType()),
                StructField("partition_b", StringType()),
                StructField("event_ts", StringType()),
                StructField("value_a", DoubleType()),
                StructField("is_value_a_interpolated", BooleanType(), False),
                StructField("value_b", DoubleType()),
                StructField("is_value_b_interpolated", BooleanType()),
            ]
        )

        expected_data = [
            ["A", "A-1", "2020-01-01 00:01:00", 349.2099914550781, False, 10.0, False],
            ["A", "A-1", "2020-01-01 00:01:30", 0.0, True, 0.0, True],
            ["A", "A-1", "2020-01-01 00:02:00", 351.32000732421875, False, 7.0, False],
            ["A", "A-2", "2020-01-01 00:01:00", 346.76499938964844, False, 8.5, False],
            ["B", "B-1", "2020-01-01 00:01:00", 362.1000061035156, False, 4.0, False],
            ["B", "B-2", "2020-01-01 00:01:00", 361.1000061035156, False, 5.0, False],
            ["B", "B-2", "2020-01-01 00:01:30", 0.0, True, 0.0, True],
            ["B", "B-2", "2020-01-01 00:02:00", 350.32000732421875, False, 6.0, False],
        ]

        expected_df: DataFrame = self.buildTestDF(expected_schema, expected_data)

        actual_df: DataFrame = self.interpolate_helper.interpolate(
            tsdf=self.input_tsdf,
            partition_cols=["partition_a", "partition_b"],
            target_cols=["value_a", "value_b"],
            sample_freq="30 seconds",
            ts_col="event_ts",
            sample_func="mean",
            fill="zero",
        )

        # actual_df.show()
        # expected_df.show()
        # actual_df.printSchema()
        # expected_df.printSchema()

        assert_df_equality(expected_df, actual_df)

    def test_zero_fill_interpolation(self):
        self.buildTestDataFrame()

        expected_schema = StructType(
            [
                StructField("partition_a", StringType()),
                StructField("partition_b", StringType()),
                StructField("event_ts", StringType()),
                StructField("value_a", DoubleType()),
                StructField("is_value_a_interpolated", BooleanType(), False),
                StructField("value_b", DoubleType()),
                StructField("is_value_b_interpolated", BooleanType()),
            ]
        )

        expected_data = [
            ["A", "A-1", "2020-01-01 00:01:00", 349.2099914550781, False, 10.0, False],
            ["A", "A-1", "2020-01-01 00:01:30", 0.0, True, 0.0, True],
            ["A", "A-1", "2020-01-01 00:02:00", 351.32000732421875, False, 7.0, False],
            ["A", "A-2", "2020-01-01 00:01:00", 346.76499938964844, False, 8.5, False],
            ["B", "B-1", "2020-01-01 00:01:00", 362.1000061035156, False, 4.0, False],
            ["B", "B-2", "2020-01-01 00:01:00", 361.1000061035156, False, 5.0, False],
            ["B", "B-2", "2020-01-01 00:01:30", 0.0, True, 0.0, True],
            ["B", "B-2", "2020-01-01 00:02:00", 350.32000732421875, False, 6.0, False],
        ]

        expected_df: DataFrame = self.buildTestDF(expected_schema, expected_data)

        actual_df: DataFrame = self.interpolate_helper.interpolate(
            tsdf=self.input_tsdf,
            partition_cols=["partition_a", "partition_b"],
            target_cols=["value_a", "value_b"],
            sample_freq="30 seconds",
            ts_col="event_ts",
            sample_func="mean",
            fill="zero",
        )

        # actual_df.show()
        # expected_df.show()
        # actual_df.printSchema()
        # expected_df.printSchema()

        assert_df_equality(expected_df, actual_df)

    def test_zero_fill_interpolation(self):
        self.buildTestDataFrame()

        expected_schema = StructType(
            [
                StructField("partition_a", StringType()),
                StructField("partition_b", StringType()),
                StructField("event_ts", StringType()),
                StructField("value_a", DoubleType()),
                StructField("is_value_a_interpolated", BooleanType(), False),
                StructField("value_b", DoubleType()),
                StructField("is_value_b_interpolated", BooleanType()),
            ]
        )

        expected_data = [
            ["A", "A-1", "2020-01-01 00:01:00", 349.2099914550781, False, 10.0, False],
            ["A", "A-1", "2020-01-01 00:01:30", 0.0, True, 0.0, True],
            ["A", "A-1", "2020-01-01 00:02:00", 351.32000732421875, False, 7.0, False],
            ["A", "A-2", "2020-01-01 00:01:00", 346.76499938964844, False, 8.5, False],
            ["B", "B-1", "2020-01-01 00:01:00", 362.1000061035156, False, 4.0, False],
            ["B", "B-2", "2020-01-01 00:01:00", 361.1000061035156, False, 5.0, False],
            ["B", "B-2", "2020-01-01 00:01:30", 0.0, True, 0.0, True],
            ["B", "B-2", "2020-01-01 00:02:00", 350.32000732421875, False, 6.0, False],
        ]

        expected_df: DataFrame = self.buildTestDF(expected_schema, expected_data)

        actual_df: DataFrame = self.interpolate_helper.interpolate(
            tsdf=self.input_tsdf,
            partition_cols=["partition_a", "partition_b"],
            target_cols=["value_a", "value_b"],
            sample_freq="30 seconds",
            ts_col="event_ts",
            sample_func="mean",
            fill="zero",
        )

        assert_df_equality(expected_df, actual_df)
