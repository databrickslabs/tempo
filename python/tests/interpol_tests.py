import unittest
from typing import List

from parameterized import parameterized, parameterized_class

from tempo.interpol import interpolate, zero_fill
from tempo.tsdf import TSDF
from tests.base import SparkTest

@parameterized_class(("data_type", "interpol_cols"),[
    ('simple_ts_idx', ["open", "close"] ),
    ('simple_ts_no_series', ["trade_pr"])
])
class InterpolationTests(SparkTest):

    def test_zero_fill(self):
        # load the initial & expected dataframes
        init_tsdf: TSDF = self.get_test_function_df_builder(self.data_type,
                                                            "init").as_tsdf()
        expected_tsdf: TSDF = self.get_test_function_df_builder(self.data_type,
                                                                "expected").as_tsdf()

        # interpolate
        actual_tsdf: TSDF = interpolate(init_tsdf,
                                        self.interpol_cols,
                                        zero_fill,
                                        0,
                                        0)
        actual_tsdf.withNaturalOrdering().show()

        # compare
        self.assertDataFrameEquality(expected_tsdf.withNaturalOrdering(),
                                     actual_tsdf.withNaturalOrdering())

    def test_linear(self):
        # load the initial & expected dataframes
        init_tsdf: TSDF = self.get_test_function_df_builder(self.data_type,
                                                            "init").as_tsdf()
        expected_tsdf: TSDF = self.get_test_function_df_builder(self.data_type,
                                                                "expected").as_tsdf()

        # interpolate
        actual_tsdf: TSDF = interpolate(init_tsdf,
                                        self.interpol_cols,
                                        "linear",
                                        1,
                                        1)
        actual_tsdf.withNaturalOrdering().show()

        # compare
        self.assertDataFrameEquality(expected_tsdf.withNaturalOrdering(),
                                     actual_tsdf.withNaturalOrdering())



# class InterpolationUnitTest(SparkTest):
#
#     def test_validate_col_exist_in_df(self):
#         input_df: DataFrame = self.get_test_df_builder("init").as_sdf()
#
#         self.assertRaises(
#             ValueError,
#             self.interpolate_helper._Interpolation__validate_col,
#             input_df,
#             ["partition_a", "does_not_exist"],
#             ["value_a", "value_b"],
#             "event_ts",
#         )
#
#         self.assertRaises(
#             ValueError,
#             self.interpolate_helper._Interpolation__validate_col,
#             input_df,
#             ["partition_a", "partition_b"],
#             ["does_not_exist", "value_b"],
#             "event_ts",
#         )
#
#         self.assertRaises(
#             ValueError,
#             self.interpolate_helper._Interpolation__validate_col,
#             input_df,
#             ["partition_a", "partition_b"],
#             ["value_a", "value_b"],
#             "wrongly_named",
#         )
#
#     def test_validate_col_target_cols_data_type(self):
#         input_df: DataFrame = self.get_test_df_builder("init").as_sdf()
#
#         self.assertRaises(
#             TypeError,
#             self.interpolate_helper._Interpolation__validate_col,
#             input_df,
#             ["partition_a", "partition_b"],
#             ["string_target", "float_target"],
#             "event_ts",
#         )
#
#     def test_fill_validation(self):
#         """Test fill parameter is valid."""
#
#         # load test data
#         input_tsdf: TSDF = self.get_test_df_builder("init").as_tsdf()
#
#         # interpolate
#         self.assertRaises(
#             ValueError,
#             self.interpolate_helper.interpolate,
#             input_tsdf,
#             ["partition_a", "partition_b"],
#             ["value_a", "value_b"],
#             "30 seconds",
#             "event_ts",
#             "mean",
#             "fill_wrong",
#             True,
#         )
#
#     def test_target_column_validation(self):
#         """Test target columns exist in schema, and are of the right type (numeric)."""
#
#         # load test data
#         input_tsdf: TSDF = self.get_test_df_builder("init").as_tsdf()
#
#         # interpolate
#         self.assertRaises(
#             ValueError,
#             self.interpolate_helper.interpolate,
#             input_tsdf,
#             ["partition_a", "partition_b"],
#             ["target_column_wrong", "value_b"],
#             "30 seconds",
#             "event_ts",
#             "mean",
#             "zero",
#             True,
#         )
#
#     def test_partition_column_validation(self):
#         """Test partition columns exist in schema."""
#
#         # load test data
#         input_tsdf: TSDF = self.get_test_df_builder("init").as_tsdf()
#
#         # interpolate
#         self.assertRaises(
#             ValueError,
#             self.interpolate_helper.interpolate,
#             input_tsdf,
#             ["partition_c", "partition_column_wrong"],
#             ["value_a", "value_b"],
#             "30 seconds",
#             "event_ts",
#             "mean",
#             "zero",
#             True,
#         )
#
#     def test_ts_column_validation(self):
#         """Test time series column exist in schema."""
#
#         # load test data
#         input_tsdf: TSDF = self.get_test_df_builder("init").as_tsdf()
#
#         # interpolate
#         self.assertRaises(
#             ValueError,
#             self.interpolate_helper.interpolate,
#             input_tsdf,
#             ["partition_a", "partition_b"],
#             ["value_a", "value_b"],
#             "30 seconds",
#             "event_ts_wrong",
#             "mean",
#             "zero",
#             True,
#         )
#
#     def test_zero_fill_interpolation(self):
#         """Test zero fill interpolation.
#
#         For zero fill interpolation we expect any missing timeseries values to be generated and filled in with zeroes.
#         If after sampling there are null values in the target column these will also be filled with zeroes.
#
#         """
#
#         # load test data
#         simple_input_tsdf: TSDF = self.get_test_df_builder("simple_init").as_tsdf()
#         expected_df: DataFrame = self.get_test_df_builder("expected").as_sdf()
#
#         # interpolate
#         actual_df: DataFrame = self.interpolate_helper.interpolate(
#             tsdf=simple_input_tsdf,
#             target_cols=["value_a", "value_b"],
#             freq="30 seconds",
#             func="mean",
#             method="zero",
#             show_interpolated=True,
#         )
#
#         self.assertDataFrameEquality(expected_df, actual_df, ignore_nullable=True)
#
#     def test_zero_fill_interpolation_no_perform_checks(self):
#         """Test zero fill interpolation.
#
#         For zero fill interpolation we expect any missing timeseries values to be generated and filled in with zeroes.
#         If after sampling there are null values in the target column these will also be filled with zeroes.
#
#         """
#
#         # load test data
#         simple_input_tsdf: TSDF = self.get_test_df_builder("simple_init").as_tsdf()
#         expected_df: DataFrame = self.get_test_df_builder("expected").as_sdf()
#
#         # interpolate
#         actual_df: DataFrame = self.interpolate_helper.interpolate(
#             tsdf=simple_input_tsdf,
#             target_cols=["value_a", "value_b"],
#             freq="30 seconds",
#             func="mean",
#             method="zero",
#             show_interpolated=True,
#             perform_checks=False,
#         )
#
#         self.assertDataFrameEquality(expected_df, actual_df, ignore_nullable=True)
#
#     def test_null_fill_interpolation(self):
#         """Test null fill interpolation.
#
#         For null fill interpolation we expect any missing timeseries values to be generated and filled in with nulls.
#         If after sampling there are null values in the target column these will also be kept as nulls.
#
#         """
#
#         # load test data
#         simple_input_tsdf: TSDF = self.get_test_df_builder("simple_init").as_tsdf()
#         expected_df: DataFrame = self.get_test_df_builder("expected").as_sdf()
#
#         # interpolate
#         actual_df: DataFrame = self.interpolate_helper.interpolate(
#             tsdf=simple_input_tsdf,
#             target_cols=["value_a", "value_b"],
#             freq="30 seconds",
#             func="mean",
#             method="null",
#             show_interpolated=True,
#         )
#
#         self.assertDataFrameEquality(expected_df, actual_df, ignore_nullable=True)
#
#     def test_back_fill_interpolation(self):
#         """Test back fill interpolation.
#
#         For back fill interpolation we expect any missing timeseries values to be generated and filled with the nearest subsequent non-null value.
#         If the right (latest) edge contains is null then preceding interpolated values will be null until the next non-null value.
#         Pre-existing nulls are treated as the same as missing values, and will be replaced with an interpolated value.
#
#         """
#
#         # load test data
#         simple_input_tsdf: TSDF = self.get_test_df_builder("simple_init").as_tsdf()
#         expected_df: DataFrame = self.get_test_df_builder("expected").as_sdf()
#
#         # interpolate
#         actual_df: DataFrame = self.interpolate_helper.interpolate(
#             tsdf=simple_input_tsdf,
#             target_cols=["value_a", "value_b"],
#             freq="30 seconds",
#             func="mean",
#             method="bfill",
#             show_interpolated=True,
#         )
#
#         self.assertDataFrameEquality(expected_df, actual_df, ignore_nullable=True)
#
#     def test_forward_fill_interpolation(self):
#         """Test forward fill interpolation.
#
#         For forward fill interpolation we expect any missing timeseries values to be generated and filled with the nearest preceding non-null value.
#         If the left (earliest) edge  is null then subsequent interpolated values will be null until the next non-null value.
#         Pre-existing nulls are treated as the same as missing values, and will be replaced with an interpolated value.
#
#         """
#
#         # load test data
#         simple_input_tsdf: TSDF = self.get_test_df_builder("simple_init").as_tsdf()
#         expected_df: DataFrame = self.get_test_df_builder("expected").as_sdf()
#
#         # interpolate
#         actual_df: DataFrame = self.interpolate_helper.interpolate(
#             tsdf=simple_input_tsdf,
#             target_cols=["value_a", "value_b"],
#             freq="30 seconds",
#             func="mean",
#             method="ffill",
#             show_interpolated=True,
#         )
#
#         self.assertDataFrameEquality(expected_df, actual_df, ignore_nullable=True)
#
#     def test_linear_fill_interpolation(self):
#         """Test linear fill interpolation.
#
#         For linear fill interpolation we expect any missing timeseries values to be generated and filled using linear interpolation.
#         If the right (latest) or left (earliest) edges  is null then subsequent interpolated values will be null until the next non-null value.
#         Pre-existing nulls are treated as the same as missing values, and will be replaced with an interpolated value.
#
#         """
#
#         # load test data
#         simple_input_tsdf: TSDF = self.get_test_df_builder("simple_init").as_tsdf()
#         expected_df: DataFrame = self.get_test_df_builder("expected").as_sdf()
#
#         # interpolate
#         actual_df: DataFrame = self.interpolate_helper.interpolate(
#             tsdf=simple_input_tsdf,
#             target_cols=["value_a", "value_b"],
#             freq="30 seconds",
#             func="mean",
#             method="linear",
#             show_interpolated=True,
#         )
#
#         self.assertDataFrameEquality(expected_df, actual_df, ignore_nullable=True)
#
#     def test_different_freq_abbreviations(self):
#         """Test abbreviated frequency values
#
#         e.g. sec and seconds will both work.
#
#         """
#
#         # load test data
#         simple_input_tsdf: TSDF = self.get_test_df_builder("simple_init").as_tsdf()
#         expected_df: DataFrame = self.get_test_df_builder("expected").as_sdf()
#
#         # interpolate
#         actual_df: DataFrame = self.interpolate_helper.interpolate(
#             tsdf=simple_input_tsdf,
#             target_cols=["value_a", "value_b"],
#             freq="30 sec",
#             func="mean",
#             method="linear",
#             show_interpolated=True,
#         )
#
#         self.assertDataFrameEquality(expected_df, actual_df, ignore_nullable=True)
#
#     def test_show_interpolated(self):
#         """Test linear `show_interpolated` flag
#
#         For linear fill interpolation we expect any missing timeseries values to be generated and filled using linear interpolation.
#         If the right (latest) or left (earliest) edges  is null then subsequent interpolated values will be null until the next non-null value.
#         Pre-existing nulls are treated as the same as missing values, and will be replaced with an interpolated value.
#
#         """
#
#         # load test data
#         simple_input_tsdf: TSDF = self.get_test_df_builder("simple_init").as_tsdf()
#         expected_df: DataFrame = self.get_test_df_builder("expected").as_sdf()
#
#         # interpolate
#         actual_df: DataFrame = self.interpolate_helper.interpolate(
#             tsdf=simple_input_tsdf,
#             target_cols=["value_a", "value_b"],
#             freq="30 seconds",
#             func="mean",
#             method="linear",
#             show_interpolated=False,
#         )
#
#         self.assertDataFrameEquality(expected_df, actual_df, ignore_nullable=True)
#
#     def test_validate_ts_col_data_type_is_not_timestamp(self):
#         input_df: DataFrame = self.get_test_df_builder("init").as_sdf()
#
#         self.assertRaises(
#             ValueError,
#             self.interpolate_helper._Interpolation__validate_col,
#             input_df,
#             ["partition_a", "partition_b"],
#             ["value_a", "value_b"],
#             "event_ts",
#             "not_timestamp",
#         )
#
#     def test_interpolation_freq_is_none(self):
#         """Test a ValueError is raised when freq is None."""
#
#         # load test data
#         simple_input_tsdf: TSDF = self.get_test_df_builder("init").as_tsdf()
#
#         # interpolate
#         self.assertRaises(
#             ValueError,
#             self.interpolate_helper.interpolate,
#             simple_input_tsdf,
#             "event_ts",
#             ["partition_a", "partition_b"],
#             ["value_a", "value_b"],
#             None,
#             "mean",
#             "zero",
#             True,
#         )
#
#     def test_interpolation_func_is_none(self):
#         """Test a ValueError is raised when func is None."""
#
#         # load test data
#         simple_input_tsdf: TSDF = self.get_test_df_builder("init").as_tsdf()
#
#         # interpolate
#         self.assertRaises(
#             ValueError,
#             self.interpolate_helper.interpolate,
#             simple_input_tsdf,
#             "event_ts",
#             ["partition_a", "partition_b"],
#             ["value_a", "value_b"],
#             "30 seconds",
#             None,
#             "zero",
#             True,
#         )
#
#     def test_interpolation_func_is_callable(self):
#         """Test ValueError is raised when func is callable."""
#
#         # load test data
#         simple_input_tsdf: TSDF = self.get_test_df_builder("init").as_tsdf()
#
#         # interpolate
#         self.assertRaises(
#             ValueError,
#             self.interpolate_helper.interpolate,
#             simple_input_tsdf,
#             "event_ts",
#             ["partition_a", "partition_b"],
#             ["value_a", "value_b"],
#             "30 seconds",
#             sum,
#             "zero",
#             True,
#         )
#
#     def test_interpolation_freq_is_not_supported_type(self):
#         """Test ValueError is raised when func is callable."""
#
#         # load test data
#         simple_input_tsdf: TSDF = self.get_test_df_builder("init").as_tsdf()
#
#         # interpolate
#         self.assertRaises(
#             ValueError,
#             self.interpolate_helper.interpolate,
#             simple_input_tsdf,
#             "event_ts",
#             ["partition_a", "partition_b"],
#             ["value_a", "value_b"],
#             "30 not_supported_type",
#             "mean",
#             "zero",
#             True,
#         )


# MAIN
if __name__ == "__main__":
    unittest.main()
