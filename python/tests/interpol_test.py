import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from python.tempo.interpol import Interpolation
from python.tests.tests import SparkTest
from chispa.dataframe_comparer import *
from tempo.tsdf import TSDF
from tempo.utils import *


class InterpolationTest(SparkTest):
    def buildTestingDataFrame(self):
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

    def test_fill_validation(self):
        """Test fill parameter is valid."""
        self.buildTestingDataFrame()
        try:
            self.interpolate_helper.interpolate(
                tsdf=self.input_tsdf,
                partition_cols=["partition_a", "partition_b"],
                target_cols=["value_a", "value_b"],
                sample_freq="30 seconds",
                ts_col="event_ts",
                sample_func="mean",
                fill="abcd",
            )
        except ValueError as e:
            self.assertEqual(type(e), ValueError)
        else:
            self.fail("ValueError not raised")

    def test_target_column_validation(self):
        """Test target columns exist in schema, and are of the right type (numeric)."""
        self.buildTestingDataFrame()
        try:
            self.interpolate_helper.interpolate(
                tsdf=self.input_tsdf,
                partition_cols=["partition_a", "partition_b"],
                target_cols=["partition_a", "value_b"],
                sample_freq="30 seconds",
                ts_col="event_ts",
                sample_func="mean",
                fill="zero",
            )
        except ValueError as e:
            self.assertEqual(type(e), ValueError)
        else:
            self.fail("ValueError not raised")

    def test_partition_column_validation(self):
        """Test partition columns exist in schema."""
        self.buildTestingDataFrame()
        try:
            self.interpolate_helper.interpolate(
                tsdf=self.input_tsdf,
                partition_cols=["partition_c", "partition_b"],
                target_cols=["value_a", "value_b"],
                sample_freq="30 seconds",
                ts_col="event_ts",
                sample_func="mean",
                fill="zero",
            )
        except ValueError as e:
            self.assertEqual(type(e), ValueError)
        else:
            self.fail("ValueError not raised")

    def test_ts_column_validation(self):
        """Test time series column exist in schema."""
        self.buildTestingDataFrame()
        try:
            self.interpolate_helper.interpolate(
                tsdf=self.input_tsdf,
                partition_cols=["partition_a", "partition_b"],
                target_cols=["value_a", "value_b"],
                sample_freq="30 seconds",
                ts_col="value_a",
                sample_func="mean",
                fill="zero",
            )
        except ValueError as e:
            self.assertEqual(type(e), ValueError)
        else:
            self.fail("ValueError not raised")

    def test_zero_fill_interpolation(self):
        """Test zero fill interpolation."""
        self.buildTestingDataFrame()

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

    def test_null_fill_interpolation(self):
        """Test null fill interpolation."""
        self.buildTestingDataFrame()

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
            ["A", "A-1", "2020-01-01 00:01:30", None, True, None, True],
            ["A", "A-1", "2020-01-01 00:02:00", 351.32000732421875, False, 7.0, False],
            ["A", "A-2", "2020-01-01 00:01:00", 346.76499938964844, False, 8.5, False],
            ["B", "B-1", "2020-01-01 00:01:00", 362.1000061035156, False, 4.0, False],
            ["B", "B-2", "2020-01-01 00:01:00", 361.1000061035156, False, 5.0, False],
            ["B", "B-2", "2020-01-01 00:01:30", None, True, None, True],
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
            fill="null",
        )

        assert_df_equality(expected_df, actual_df)

    def test_back_fill_interpolation(self):
        """Test backwards fill interpolation."""
        self.buildTestingDataFrame()

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
            ["A", "A-1", "2020-01-01 00:01:30", 351.32000732421875, True, 7.0, True],
            ["A", "A-1", "2020-01-01 00:02:00", 351.32000732421875, False, 7.0, False],
            ["A", "A-2", "2020-01-01 00:01:00", 346.76499938964844, False, 8.5, False],
            ["B", "B-1", "2020-01-01 00:01:00", 362.1000061035156, False, 4.0, False],
            ["B", "B-2", "2020-01-01 00:01:00", 361.1000061035156, False, 5.0, False],
            ["B", "B-2", "2020-01-01 00:01:30", 350.32000732421875, True, 6.0, True],
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
            fill="back",
        )

        assert_df_equality(expected_df, actual_df)

    def test_forward_fill_interpolation(self):
        """Test forward fill interpolation."""
        self.buildTestingDataFrame()

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
            ["A", "A-1", "2020-01-01 00:01:30", 349.2099914550781, True, 10.0, True],
            ["A", "A-1", "2020-01-01 00:02:00", 351.32000732421875, False, 7.0, False],
            ["A", "A-2", "2020-01-01 00:01:00", 346.76499938964844, False, 8.5, False],
            ["B", "B-1", "2020-01-01 00:01:00", 362.1000061035156, False, 4.0, False],
            ["B", "B-2", "2020-01-01 00:01:00", 361.1000061035156, False, 5.0, False],
            ["B", "B-2", "2020-01-01 00:01:30", 361.1000061035156, True, 5.0, True],
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
            fill="forward",
        )

        assert_df_equality(expected_df, actual_df)

    def test_linear_fill_interpolation(self):
        """Test forward fill interpolation."""
        self.buildTestingDataFrame()

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
            ["A", "A-1", "2020-01-01 00:01:30", 350.26499938964844, True, 8.5, True],
            ["A", "A-1", "2020-01-01 00:02:00", 351.32000732421875, False, 7.0, False],
            ["A", "A-2", "2020-01-01 00:01:00", 346.76499938964844, False, 8.5, False],
            ["B", "B-1", "2020-01-01 00:01:00", 362.1000061035156, False, 4.0, False],
            ["B", "B-2", "2020-01-01 00:01:00", 361.1000061035156, False, 5.0, False],
            ["B", "B-2", "2020-01-01 00:01:30", 355.7100067138672, True, 5.5, True],
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
            fill="linear",
        )

        assert_df_equality(expected_df, actual_df)
