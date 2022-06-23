import unittest

from chispa.dataframe_comparer import *
from pyspark.sql.types import *

from tempo.interpol import Interpolation
from tempo.tsdf import TSDF
from tempo.utils import *

from tests.tsdf_tests import SparkTest


class InterpolationUnitTest(SparkTest):
    def setUp(self) -> None:
        super().setUp()
        # register interpolation helper
        self.interpolate_helper = Interpolation(is_resampled=False)

    def test_fill_validation(self):
        """Test fill parameter is valid."""

        # load test data
        input_tsdf: TSDF = self.get_data_as_tsdf("input_data")

        # interpolate
        try:
            self.interpolate_helper.interpolate(
                tsdf=input_tsdf,
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

        # load test data
        input_tsdf: TSDF = self.get_data_as_tsdf("input_data")

        # interpolate
        try:
            self.interpolate_helper.interpolate(
                tsdf=input_tsdf,
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

        # load test data
        input_tsdf: TSDF = self.get_data_as_tsdf("input_data")

        # interpolate
        try:
            self.interpolate_helper.interpolate(
                tsdf=input_tsdf,
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

        # load test data
        input_tsdf: TSDF = self.get_data_as_tsdf("input_data")

        # interpolate
        try:
            self.interpolate_helper.interpolate(
                tsdf=input_tsdf,
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

        # load test data
        simple_input_tsdf: TSDF = self.get_data_as_tsdf("simple_input_data")
        expected_df: DataFrame = self.get_data_as_sdf("expected_data")

        # interpolate
        actual_df: DataFrame = self.interpolate_helper.interpolate(
            tsdf=simple_input_tsdf,
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

        # load test data
        simple_input_tsdf: TSDF = self.get_data_as_tsdf("simple_input_data")
        expected_df: DataFrame = self.get_data_as_sdf("expected_data")

        # interpolate
        actual_df: DataFrame = self.interpolate_helper.interpolate(
            tsdf=simple_input_tsdf,
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

        # load test data
        simple_input_tsdf: TSDF = self.get_data_as_tsdf("simple_input_data")
        expected_df: DataFrame = self.get_data_as_sdf("expected_data")

        # interpolate
        actual_df: DataFrame = self.interpolate_helper.interpolate(
            tsdf=simple_input_tsdf,
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

        # load test data
        simple_input_tsdf: TSDF = self.get_data_as_tsdf("simple_input_data")
        expected_df: DataFrame = self.get_data_as_sdf("expected_data")

        # interpolate
        actual_df: DataFrame = self.interpolate_helper.interpolate(
            tsdf=simple_input_tsdf,
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

        # load test data
        simple_input_tsdf: TSDF = self.get_data_as_tsdf("simple_input_data")
        expected_df: DataFrame = self.get_data_as_sdf("expected_data")

        # interpolate
        actual_df: DataFrame = self.interpolate_helper.interpolate(
            tsdf=simple_input_tsdf,
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

        # load test data
        simple_input_tsdf: TSDF = self.get_data_as_tsdf("simple_input_data")
        expected_df: DataFrame = self.get_data_as_sdf("expected_data")

        # interpolate
        actual_df: DataFrame = self.interpolate_helper.interpolate(
            tsdf=simple_input_tsdf,
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

        # load test data
        simple_input_tsdf: TSDF = self.get_data_as_tsdf("simple_input_data")
        expected_df: DataFrame = self.get_data_as_sdf("expected_data")

        # interpolate
        actual_df: DataFrame = self.interpolate_helper.interpolate(
            tsdf=simple_input_tsdf,
            partition_cols=["partition_a", "partition_b"],
            target_cols=["value_a", "value_b"],
            freq="30 seconds",
            ts_col="event_ts",
            func="mean",
            method="linear",
            show_interpolated=False,
        )

        assert_df_equality(expected_df, actual_df, ignore_nullable=True)


class InterpolationIntegrationTest(SparkTest):
    def test_interpolation_using_default_tsdf_params(self):
        """
        Verify that interpolate uses the ts_col and partition_col from TSDF if not explicitly specified,
        and all columns numeric are automatically interpolated if target_col is not specified.
        """

        # load test data
        simple_input_tsdf: TSDF = self.get_data_as_tsdf("simple_input_data")
        expected_df: DataFrame = self.get_data_as_sdf("expected")

        # interpolate
        actual_df: DataFrame = simple_input_tsdf.interpolate(
            freq="30 seconds", func="mean", method="linear"
        ).df

        # compare with expected
        assert_df_equality(expected_df, actual_df, ignore_nullable=True)

    def test_interpolation_using_custom_params(self):
        """Verify that by specifying optional paramters it will change the result of the interpolation based on those modified params."""

        # Modify input DataFrame using different ts_col
        simple_input_tsdf: TSDF = self.get_data_as_tsdf("simple_input_data")
        expected_df: DataFrame = self.get_data_as_sdf("expected")

        input_tsdf = TSDF(
            simple_input_tsdf.df.withColumnRenamed("event_ts", "other_ts_col"),
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

        # load test data
        simple_input_tsdf: TSDF = self.get_data_as_tsdf("simple_input_data")

        actual_tsdf: TSDF = simple_input_tsdf.interpolate(
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

        # load test data
        simple_input_tsdf: TSDF = self.get_data_as_tsdf("simple_input_data")
        expected_df: DataFrame = self.get_data_as_sdf("expected")

        actual_df: DataFrame = (
            simple_input_tsdf.resample(freq="30 seconds", func="mean", fill=None)
            .interpolate(
                method="linear", target_cols=["value_a"], show_interpolated=True
            )
            .df
        )

        assert_df_equality(expected_df, actual_df, ignore_nullable=True)

    def test_defaults_with_resampled_df(self):
        """Verify interpolation can be chained with resample within the TSDF class"""
        # self.buildTestingDataFrame()

        # load test data
        simple_input_tsdf = self.get_data_as_tsdf("simple_input_data")
        expected_df: DataFrame = self.get_data_as_sdf("expected", convert_ts_col=True)

        actual_df: DataFrame = (
            simple_input_tsdf.resample(freq="30 seconds", func="mean", fill=None)
            .interpolate(method="ffill")
            .df
        )

        assert_df_equality(expected_df, actual_df, ignore_nullable=True)


# MAIN
if __name__ == "__main__":
    unittest.main()
