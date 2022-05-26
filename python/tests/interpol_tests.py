import unittest

from chispa.dataframe_comparer import *
from pyspark.sql.types import *

from tempo.interpol import Interpolation
from tempo.tsdf import TSDF
from tempo.utils import *

from tests.tsdf_tests import SparkTest

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

        self.expected_schema = StructType(
            [
                StructField("partition_a", StringType()),
                StructField("partition_b", StringType()),
                StructField("event_ts", StringType(), False),
                StructField("value_a", DoubleType()),
                StructField("value_b", DoubleType()),
                StructField("is_ts_interpolated", BooleanType(), False),
                StructField("is_interpolated_value_a", BooleanType(), False),
                StructField("is_interpolated_value_b", BooleanType(), False),
            ]
        )

        # TODO: This data set tests with multiple partitions, should still be implemented at some time.
        data = [
            ["A", "A-1", "2020-01-01 00:01:10", 349.21, None],
            ["A", "A-1", "2020-01-01 00:02:03", None, 4.0],
            ["A", "A-2", "2020-01-01 00:01:15", 340.21, 9.0],
            ["B", "B-1", "2020-01-01 00:01:15", 362.1, 4.0],
            ["A", "A-2", "2020-01-01 00:01:17", 353.32, 8.0],
            ["B", "B-2", "2020-01-01 00:02:14", None, 6.0],
            ["A", "A-1", "2020-01-01 00:03:02", 351.32, 7.0],
            ["B", "B-2", "2020-01-01 00:01:12", 361.1, 5.0],
        ]

        simple_data = [
            ["A", "A-1", "2020-01-01 00:00:10", 0.0, None],
            ["A", "A-1", "2020-01-01 00:01:10", 2.0, 2.0],
            ["A", "A-1", "2020-01-01 00:01:32", None, None],
            ["A", "A-1", "2020-01-01 00:02:03", None, None],
            ["A", "A-1", "2020-01-01 00:03:32", None, 7.0],
            ["A", "A-1", "2020-01-01 00:04:12", 8.0, 8.0],
            ["A", "A-1", "2020-01-01 00:05:31", 11.0, None],
        ]

        # construct dataframes
        self.input_df = self.buildTestDF(schema, data)
        self.simple_input_df = self.buildTestDF(schema, simple_data)

        # generate TSDF
        self.input_tsdf = TSDF(
            self.input_df,
            partition_cols=["partition_a", "partition_b"],
            ts_col="event_ts",
        )
        self.simple_input_tsdf = TSDF(
            self.simple_input_df,
            partition_cols=["partition_a", "partition_b"],
            ts_col="event_ts",
        )

        # register interpolation helper
        self.interpolate_helper = Interpolation(is_resampled=False)


class InterpolationUnitTest(InterpolationTest):
    def test_fill_validation(self):
        """Test fill parameter is valid."""
        self.buildTestingDataFrame()
        try:
            self.interpolate_helper.interpolate(
                tsdf=self.input_tsdf,
                partition_cols=["partition_a", "partition_b"],
                target_cols=["value_a", "value_b"],
                freq="30 seconds",
                ts_col="event_ts",
                func="mean",
                method="abcd",
                show_interpolated=True,
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
                freq="30 seconds",
                ts_col="event_ts",
                func="mean",
                method="zero",
                show_interpolated=True,
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
                freq="30 seconds",
                ts_col="event_ts",
                func="mean",
                method="zero",
                show_interpolated=True,
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
                freq="30 seconds",
                ts_col="value_a",
                func="mean",
                method="zero",
                show_interpolated=True,
            )
        except ValueError as e:
            self.assertEqual(type(e), ValueError)
        else:
            self.fail("ValueError not raised")

    def test_zero_fill_interpolation(self):
        """Test zero fill interpolation.

        For zero fill interpolation we expect any missing timeseries values to be generated and filled in with zeroes.
        If after sampling there are null values in the target column these will also be filled with zeroes.

        """
        self.buildTestingDataFrame()

        expected_data = [
            ["A", "A-1", "2020-01-01 00:00:00", 0.0, 0.0, False, False, True],
            ["A", "A-1", "2020-01-01 00:00:30", 0.0, 0.0, True, True, True],
            ["A", "A-1", "2020-01-01 00:01:00", 2.0, 2.0, False, False, False],
            ["A", "A-1", "2020-01-01 00:01:30", 0.0, 0.0, False, True, True],
            ["A", "A-1", "2020-01-01 00:02:00", 0.0, 0.0, False, True, True],
            ["A", "A-1", "2020-01-01 00:02:30", 0.0, 0.0, True, True, True],
            ["A", "A-1", "2020-01-01 00:03:00", 0.0, 0.0, True, True, True],
            ["A", "A-1", "2020-01-01 00:03:30", 0.0, 7.0, False, True, False],
            ["A", "A-1", "2020-01-01 00:04:00", 8.0, 8.0, False, False, False],
            ["A", "A-1", "2020-01-01 00:04:30", 0.0, 0.0, True, True, True],
            ["A", "A-1", "2020-01-01 00:05:00", 0.0, 0.0, True, True, True],
            ["A", "A-1", "2020-01-01 00:05:30", 11.0, 0.0, False, False, True],
        ]

        expected_df: DataFrame = self.buildTestDF(self.expected_schema, expected_data)

        actual_df: DataFrame = self.interpolate_helper.interpolate(
            tsdf=self.simple_input_tsdf,
            partition_cols=["partition_a", "partition_b"],
            target_cols=["value_a", "value_b"],
            freq="30 seconds",
            ts_col="event_ts",
            func="mean",
            method="zero",
            show_interpolated=True,
        )

        assert_df_equality(expected_df, actual_df, ignore_nullable=True)

    def test_null_fill_interpolation(self):
        """Test null fill interpolation.

        For null fill interpolation we expect any missing timeseries values to be generated and filled in with nulls.
        If after sampling there are null values in the target column these will also be kept as nulls.


        """
        self.buildTestingDataFrame()

        expected_data = [
            ["A", "A-1", "2020-01-01 00:00:00", 0.0, None, False, False, True],
            ["A", "A-1", "2020-01-01 00:00:30", None, None, True, True, True],
            ["A", "A-1", "2020-01-01 00:01:00", 2.0, 2.0, False, False, False],
            ["A", "A-1", "2020-01-01 00:01:30", None, None, False, True, True],
            ["A", "A-1", "2020-01-01 00:02:00", None, None, False, True, True],
            ["A", "A-1", "2020-01-01 00:02:30", None, None, True, True, True],
            ["A", "A-1", "2020-01-01 00:03:00", None, None, True, True, True],
            ["A", "A-1", "2020-01-01 00:03:30", None, 7.0, False, True, False],
            ["A", "A-1", "2020-01-01 00:04:00", 8.0, 8.0, False, False, False],
            ["A", "A-1", "2020-01-01 00:04:30", None, None, True, True, True],
            ["A", "A-1", "2020-01-01 00:05:00", None, None, True, True, True],
            ["A", "A-1", "2020-01-01 00:05:30", 11.0, None, False, False, True],
        ]

        expected_df: DataFrame = self.buildTestDF(self.expected_schema, expected_data)

        actual_df: DataFrame = self.interpolate_helper.interpolate(
            tsdf=self.simple_input_tsdf,
            partition_cols=["partition_a", "partition_b"],
            target_cols=["value_a", "value_b"],
            freq="30 seconds",
            ts_col="event_ts",
            func="mean",
            method="null",
            show_interpolated=True,
        )

        assert_df_equality(expected_df, actual_df, ignore_nullable=True)

    def test_back_fill_interpolation(self):
        """Test back fill interpolation.

        For back fill interpolation we expect any missing timeseries values to be generated and filled with the nearest subsequent non-null value.
        If the right (latest) edge contains is null then preceding interpolated values will be null until the next non-null value.
        Pre-existing nulls are treated as the same as missing values, and will be replaced with an interpolated value.

        """
        self.buildTestingDataFrame()

        expected_data = [
            ["A", "A-1", "2020-01-01 00:00:00", 0.0, 2.0, False, False, True],
            ["A", "A-1", "2020-01-01 00:00:30", 2.0, 2.0, True, True, True],
            ["A", "A-1", "2020-01-01 00:01:00", 2.0, 2.0, False, False, False],
            ["A", "A-1", "2020-01-01 00:01:30", 8.0, 7.0, False, True, True],
            ["A", "A-1", "2020-01-01 00:02:00", 8.0, 7.0, False, True, True],
            ["A", "A-1", "2020-01-01 00:02:30", 8.0, 7.0, True, True, True],
            ["A", "A-1", "2020-01-01 00:03:00", 8.0, 7.0, True, True, True],
            ["A", "A-1", "2020-01-01 00:03:30", 8.0, 7.0, False, True, False],
            ["A", "A-1", "2020-01-01 00:04:00", 8.0, 8.0, False, False, False],
            ["A", "A-1", "2020-01-01 00:04:30", 11.0, None, True, True, True],
            ["A", "A-1", "2020-01-01 00:05:00", 11.0, None, True, True, True],
            ["A", "A-1", "2020-01-01 00:05:30", 11.0, None, False, False, True],
        ]

        expected_df: DataFrame = self.buildTestDF(self.expected_schema, expected_data)

  
        actual_df: DataFrame = self.interpolate_helper.interpolate(
            tsdf=self.simple_input_tsdf,
            partition_cols=["partition_a", "partition_b"],
            target_cols=["value_a", "value_b"],
            freq="30 seconds",
            ts_col="event_ts",
            func="mean",
            method="bfill",
            show_interpolated=True,
        )

        assert_df_equality(expected_df, actual_df, ignore_nullable=True)

    def test_forward_fill_interpolation(self):
        """Test forward fill interpolation.

        For forward fill interpolation we expect any missing timeseries values to be generated and filled with the nearest preceding non-null value.
        If the left (earliest) edge  is null then subsequent interpolated values will be null until the next non-null value.
        Pre-existing nulls are treated as the same as missing values, and will be replaced with an interpolated value.

        """
        self.buildTestingDataFrame()

        expected_data = [
            ["A", "A-1", "2020-01-01 00:00:00", 0.0, None, False, False, True],
            ["A", "A-1", "2020-01-01 00:00:30", 0.0, None, True, True, True],
            ["A", "A-1", "2020-01-01 00:01:00", 2.0, 2.0, False, False, False],
            ["A", "A-1", "2020-01-01 00:01:30", 2.0, 2.0, False, True, True],
            ["A", "A-1", "2020-01-01 00:02:00", 2.0, 2.0, False, True, True],
            ["A", "A-1", "2020-01-01 00:02:30", 2.0, 2.0, True, True, True],
            ["A", "A-1", "2020-01-01 00:03:00", 2.0, 2.0, True, True, True],
            ["A", "A-1", "2020-01-01 00:03:30", 2.0, 7.0, False, True, False],
            ["A", "A-1", "2020-01-01 00:04:00", 8.0, 8.0, False, False, False],
            ["A", "A-1", "2020-01-01 00:04:30", 8.0, 8.0, True, True, True],
            ["A", "A-1", "2020-01-01 00:05:00", 8.0, 8.0, True, True, True],
            ["A", "A-1", "2020-01-01 00:05:30", 11.0, 8.0, False, False, True],
        ]

        expected_df: DataFrame = self.buildTestDF(self.expected_schema, expected_data)

        actual_df: DataFrame = self.interpolate_helper.interpolate(
            tsdf=self.simple_input_tsdf,
            partition_cols=["partition_a", "partition_b"],
            target_cols=["value_a", "value_b"],
            freq="30 seconds",
            ts_col="event_ts",
            func="mean",
            method="ffill",
            show_interpolated=True,
        )

        assert_df_equality(expected_df, actual_df, ignore_nullable=True)

    def test_linear_fill_interpolation(self):
        """Test linear fill interpolation.

        For linear fill interpolation we expect any missing timeseries values to be generated and filled using linear interpolation.
        If the right (latest) or left (earliest) edges  is null then subsequent interpolated values will be null until the next non-null value.
        Pre-existing nulls are treated as the same as missing values, and will be replaced with an interpolated value.

        """
        self.buildTestingDataFrame()

        expected_data = [
            ["A", "A-1", "2020-01-01 00:00:00", 0.0, None, False, False, True],
            ["A", "A-1", "2020-01-01 00:00:30", 1.0, None, True, True, True],
            ["A", "A-1", "2020-01-01 00:01:00", 2.0, 2.0, False, False, False],
            ["A", "A-1", "2020-01-01 00:01:30", 3.0, 3.0, False, True, True],
            ["A", "A-1", "2020-01-01 00:02:00", 4.0, 4.0, False, True, True],
            ["A", "A-1", "2020-01-01 00:02:30", 5.0, 5.0, True, True, True],
            ["A", "A-1", "2020-01-01 00:03:00", 6.0, 6.0, True, True, True],
            ["A", "A-1", "2020-01-01 00:03:30", 7.0, 7.0, False, True, False],
            ["A", "A-1", "2020-01-01 00:04:00", 8.0, 8.0, False, False, False],
            ["A", "A-1", "2020-01-01 00:04:30", 9.0, None, True, True, True],
            ["A", "A-1", "2020-01-01 00:05:00", 10.0, None, True, True, True],
            ["A", "A-1", "2020-01-01 00:05:30", 11.0, None, False, False, True],
        ]

        expected_df: DataFrame = self.buildTestDF(self.expected_schema, expected_data)

        actual_df: DataFrame = self.interpolate_helper.interpolate(
            tsdf=self.simple_input_tsdf,
            partition_cols=["partition_a", "partition_b"],
            target_cols=["value_a", "value_b"],
            freq="30 seconds",
            ts_col="event_ts",
            func="mean",
            method="linear",
            show_interpolated=True,
        )

        assert_df_equality(expected_df, actual_df, ignore_nullable=True)

    def test_different_freq_abbreviations(self):
        """Test abbreviated frequency values

            e.g. sec and seconds will both work.

        """
        self.buildTestingDataFrame()

        expected_data = [
            ["A", "A-1", "2020-01-01 00:00:00", 0.0, None, False, False, True],
            ["A", "A-1", "2020-01-01 00:00:30", 1.0, None, True, True, True],
            ["A", "A-1", "2020-01-01 00:01:00", 2.0, 2.0, False, False, False],
            ["A", "A-1", "2020-01-01 00:01:30", 3.0, 3.0, False, True, True],
            ["A", "A-1", "2020-01-01 00:02:00", 4.0, 4.0, False, True, True],
            ["A", "A-1", "2020-01-01 00:02:30", 5.0, 5.0, True, True, True],
            ["A", "A-1", "2020-01-01 00:03:00", 6.0, 6.0, True, True, True],
            ["A", "A-1", "2020-01-01 00:03:30", 7.0, 7.0, False, True, False],
            ["A", "A-1", "2020-01-01 00:04:00", 8.0, 8.0, False, False, False],
            ["A", "A-1", "2020-01-01 00:04:30", 9.0, None, True, True, True],
            ["A", "A-1", "2020-01-01 00:05:00", 10.0, None, True, True, True],
            ["A", "A-1", "2020-01-01 00:05:30", 11.0, None, False, False, True],
        ]

        expected_df: DataFrame = self.buildTestDF(self.expected_schema, expected_data)

        actual_df: DataFrame = self.interpolate_helper.interpolate(
            tsdf=self.simple_input_tsdf,
            partition_cols=["partition_a", "partition_b"],
            target_cols=["value_a", "value_b"],
            freq="30 sec",
            ts_col="event_ts",
            func="mean",
            method="linear",
            show_interpolated=True,
        )

        assert_df_equality(expected_df, actual_df, ignore_nullable=True)

    def test_show_interpolated(self):
        """Test linear `show_interpolated` flag

        For linear fill interpolation we expect any missing timeseries values to be generated and filled using linear interpolation.
        If the right (latest) or left (earliest) edges  is null then subsequent interpolated values will be null until the next non-null value.
        Pre-existing nulls are treated as the same as missing values, and will be replaced with an interpolated value.

        """
        self.buildTestingDataFrame()

        expected_data = [
            ["A", "A-1", "2020-01-01 00:00:00", 0.0, None],
            ["A", "A-1", "2020-01-01 00:00:30", 1.0, None],
            ["A", "A-1", "2020-01-01 00:01:00", 2.0, 2.0],
            ["A", "A-1", "2020-01-01 00:01:30", 3.0, 3.0],
            ["A", "A-1", "2020-01-01 00:02:00", 4.0, 4.0],
            ["A", "A-1", "2020-01-01 00:02:30", 5.0, 5.0],
            ["A", "A-1", "2020-01-01 00:03:00", 6.0, 6.0],
            ["A", "A-1", "2020-01-01 00:03:30", 7.0, 7.0],
            ["A", "A-1", "2020-01-01 00:04:00", 8.0, 8.0],
            ["A", "A-1", "2020-01-01 00:04:30", 9.0, None],
            ["A", "A-1", "2020-01-01 00:05:00", 10.0, None],
            ["A", "A-1", "2020-01-01 00:05:30", 11.0, None],
        ]

        expected_schema = StructType(
            [
                StructField("partition_a", StringType()),
                StructField("partition_b", StringType()),
                StructField("event_ts", StringType()),
                StructField("value_a", DoubleType()),
                StructField("value_b", DoubleType()),
            ]
        )

        expected_df: DataFrame = self.buildTestDF(expected_schema, expected_data)

        actual_df: DataFrame = self.interpolate_helper.interpolate(
            tsdf=self.simple_input_tsdf,
            partition_cols=["partition_a", "partition_b"],
            target_cols=["value_a", "value_b"],
            freq="30 seconds",
            ts_col="event_ts",
            func="mean",
            method="linear",
            show_interpolated=False,
        )

        assert_df_equality(expected_df, actual_df, ignore_nullable=True)


class InterpolationIntegrationTest(InterpolationTest):
    def test_interpolation_using_default_tsdf_params(self):
        """
        Verify that interpolate uses the ts_col and partition_col from TSDF if not explicitly specified,
        and all columns numeric are automatically interpolated if target_col is not specified.
        """
        self.buildTestingDataFrame()

        expected_data = [
            ["A", "A-1", "2020-01-01 00:00:00", 0.0, None],
            ["A", "A-1", "2020-01-01 00:00:30", 1.0, None],
            ["A", "A-1", "2020-01-01 00:01:00", 2.0, 2.0],
            ["A", "A-1", "2020-01-01 00:01:30", 3.0, 3.0],
            ["A", "A-1", "2020-01-01 00:02:00", 4.0, 4.0],
            ["A", "A-1", "2020-01-01 00:02:30", 5.0, 5.0],
            ["A", "A-1", "2020-01-01 00:03:00", 6.0, 6.0],
            ["A", "A-1", "2020-01-01 00:03:30", 7.0, 7.0],
            ["A", "A-1", "2020-01-01 00:04:00", 8.0, 8.0],
            ["A", "A-1", "2020-01-01 00:04:30", 9.0, None],
            ["A", "A-1", "2020-01-01 00:05:00", 10.0, None],
            ["A", "A-1", "2020-01-01 00:05:30", 11.0, None],
        ]

        expected_schema = StructType(
            [
                StructField("partition_a", StringType()),
                StructField("partition_b", StringType()),
                StructField("event_ts", StringType()),
                StructField("value_a", DoubleType()),
                StructField("value_b", DoubleType()),
            ]
        )

        expected_df: DataFrame = self.buildTestDF(expected_schema, expected_data)

        actual_df: DataFrame = self.simple_input_tsdf.interpolate(
            freq="30 seconds", func="mean", method="linear"
        ).df

        assert_df_equality(expected_df, actual_df, ignore_nullable=True)

    def test_interpolation_using_custom_params(self):
        """Verify that by specifying optional paramters it will change the result of the interpolation based on those modified params."""
        self.buildTestingDataFrame()

        expected_data = [
            ["A", "A-1", "2020-01-01 00:00:00", 0.0, False, False],
            ["A", "A-1", "2020-01-01 00:00:30", 1.0, True, True],
            ["A", "A-1", "2020-01-01 00:01:00", 2.0, False, False],
            ["A", "A-1", "2020-01-01 00:01:30", 3.0, False, True],
            ["A", "A-1", "2020-01-01 00:02:00", 4.0, False, True],
            ["A", "A-1", "2020-01-01 00:02:30", 5.0, True, True],
            ["A", "A-1", "2020-01-01 00:03:00", 6.0, True, True],
            ["A", "A-1", "2020-01-01 00:03:30", 7.0, False, True],
            ["A", "A-1", "2020-01-01 00:04:00", 8.0, False, False],
            ["A", "A-1", "2020-01-01 00:04:30", 9.0, True, True],
            ["A", "A-1", "2020-01-01 00:05:00", 10.0, True, True],
            ["A", "A-1", "2020-01-01 00:05:30", 11.0, False, False],
        ]

        expected_schema = StructType(
            [
                StructField("partition_a", StringType()),
                StructField("partition_b", StringType()),
                StructField("other_ts_col", StringType(), False),
                StructField("value_a", DoubleType()),
                StructField("is_ts_interpolated", BooleanType(), False),
                StructField("is_interpolated_value_a", BooleanType(), False),
            ]
        )

        # Modify input DataFrame using different ts_col
        expected_df: DataFrame = self.buildTestDF(
            expected_schema, expected_data, ts_cols=["other_ts_col"]
        )

        input_tsdf = TSDF(
            self.simple_input_tsdf.df.withColumnRenamed("event_ts", "other_ts_col"),
            partition_cols=["partition_a", "partition_b"],
            ts_col="other_ts_col",
        )

        actual_df: DataFrame = input_tsdf.interpolate(
            ts_col="other_ts_col",
            show_interpolated=True,
            partition_cols=["partition_a", "partition_b"],
            target_cols=["value_a"],
            freq="30 seconds",
            func="mean",
            method="linear",
        ).df

        assert_df_equality(expected_df, actual_df, ignore_nullable=True)

    def test_tsdf_constructor_params_are_updated(self):
        """Verify that resulting TSDF class has the correct values for ts_col and partition_col based on the interpolation."""
        self.buildTestingDataFrame()

        actual_tsdf: TSDF = self.simple_input_tsdf.interpolate(
            ts_col="event_ts",
            show_interpolated=True,
            partition_cols=["partition_b"],
            target_cols=["value_a"],
            freq="30 seconds",
            func="mean",
            method="linear",
        )

        self.assertEqual(actual_tsdf.ts_col, "event_ts")
        self.assertEqual(actual_tsdf.partitionCols, ["partition_b"])

    def test_interpolation_on_sampled_data(self):
        """Verify interpolation can be chained with resample within the TSDF class"""
        self.buildTestingDataFrame()

        expected_data = [
            ["A", "A-1", "2020-01-01 00:00:00", 0.0, False, False],
            ["A", "A-1", "2020-01-01 00:00:30", 1.0, True, True],
            ["A", "A-1", "2020-01-01 00:01:00", 2.0, False, False],
            ["A", "A-1", "2020-01-01 00:01:30", 3.0, False, True],
            ["A", "A-1", "2020-01-01 00:02:00", 4.0, False, True],
            ["A", "A-1", "2020-01-01 00:02:30", 5.0, True, True],
            ["A", "A-1", "2020-01-01 00:03:00", 6.0, True, True],
            ["A", "A-1", "2020-01-01 00:03:30", 7.0, False, True],
            ["A", "A-1", "2020-01-01 00:04:00", 8.0, False, False],
            ["A", "A-1", "2020-01-01 00:04:30", 9.0, True, True],
            ["A", "A-1", "2020-01-01 00:05:00", 10.0, True, True],
            ["A", "A-1", "2020-01-01 00:05:30", 11.0, False, False],
        ]

        expected_schema = StructType(
            [
                StructField("partition_a", StringType()),
                StructField("partition_b", StringType()),
                StructField("event_ts", StringType(), False),
                StructField("value_a", DoubleType()),
                StructField("is_ts_interpolated", BooleanType(), False),
                StructField("is_interpolated_value_a", BooleanType(), False),
            ]
        )

        expected_df: DataFrame = self.buildTestDF(expected_schema, expected_data)

        actual_df: DataFrame = (
            self.simple_input_tsdf.resample(freq="30 seconds", func="mean", fill=None)
            .interpolate(
                method="linear", target_cols=["value_a"], show_interpolated=True
            )
            .df
        )

        assert_df_equality(expected_df, actual_df, ignore_nullable=True)

    def test_defaults_with_resampled_df(self):
        """Verify interpolation can be chained with resample within the TSDF class"""
        self.buildTestingDataFrame()

        expected_data = [
            ["A", "A-1", "2020-01-01 00:00:00", 0.0, None],
            ["A", "A-1", "2020-01-01 00:00:30", 0.0, None],
            ["A", "A-1", "2020-01-01 00:01:00", 2.0, 2.0],
            ["A", "A-1", "2020-01-01 00:01:30", 2.0, 2.0],
            ["A", "A-1", "2020-01-01 00:02:00", 2.0, 2.0],
            ["A", "A-1", "2020-01-01 00:02:30", 2.0, 2.0],
            ["A", "A-1", "2020-01-01 00:03:00", 2.0, 2.0],
            ["A", "A-1", "2020-01-01 00:03:30", 2.0, 7.0],
            ["A", "A-1", "2020-01-01 00:04:00", 8.0, 8.0],
            ["A", "A-1", "2020-01-01 00:04:30", 8.0, 8.0],
            ["A", "A-1", "2020-01-01 00:05:00", 8.0, 8.0],
            ["A", "A-1", "2020-01-01 00:05:30", 11.0, 8.0],
        ]

        expected_schema = StructType(
            [
                StructField("partition_a", StringType()),
                StructField("partition_b", StringType()),
                StructField("event_ts", StringType(), False),
                StructField("value_a", DoubleType()),
                StructField("value_b", DoubleType()),
            ]
        )

        expected_df: DataFrame = self.buildTestDF(expected_schema, expected_data)

        actual_df: DataFrame = (
            self.simple_input_tsdf.resample(freq="30 seconds", func="mean", fill=None)
            .interpolate(method="ffill")
            .df
        )

        assert_df_equality(expected_df, actual_df, ignore_nullable=True)


## MAIN
if __name__ == "__main__":
    unittest.main()
