import unittest

from parameterized import parameterized_class

from tempo.interpol import backward_fill, forward_fill, interpolate, zero_fill
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
        # Note: Linear interpolation behavior:
        # - Values between known points are linearly interpolated
        # - Values at the end with no following point are forward-filled with the last known value
        # - This is the default pandas behavior (limit_direction='forward')
        actual_tsdf: TSDF = interpolate(init_tsdf,
                                        self.interpol_cols,
                                        "linear",
                                        1,
                                        1)
        actual_tsdf.withNaturalOrdering().show()

        # compare
        self.assertDataFrameEquality(expected_tsdf.withNaturalOrdering(),
                                     actual_tsdf.withNaturalOrdering())

    def test_forward_fill(self):
        # load the initial & expected dataframes
        init_tsdf: TSDF = self.get_test_function_df_builder(self.data_type,
                                                            "init").as_tsdf()
        expected_tsdf: TSDF = self.get_test_function_df_builder(self.data_type,
                                                                "expected").as_tsdf()

        # interpolate
        actual_tsdf: TSDF = interpolate(init_tsdf,
                                        self.interpol_cols,
                                        forward_fill,
                                        1,
                                        0)
        actual_tsdf.withNaturalOrdering().show()

        # compare
        self.assertDataFrameEquality(expected_tsdf.withNaturalOrdering(),
                                     actual_tsdf.withNaturalOrdering())

    def test_backward_fill(self):
        # load the initial & expected dataframes
        init_tsdf: TSDF = self.get_test_function_df_builder(self.data_type,
                                                            "init").as_tsdf()
        expected_tsdf: TSDF = self.get_test_function_df_builder(self.data_type,
                                                                "expected").as_tsdf()

        # interpolate
        actual_tsdf: TSDF = interpolate(init_tsdf,
                                        self.interpol_cols,
                                        backward_fill,
                                        0,
                                        1)
        actual_tsdf.withNaturalOrdering().show()

        # compare
        self.assertDataFrameEquality(expected_tsdf.withNaturalOrdering(),
                                     actual_tsdf.withNaturalOrdering())


# Add tests for non-numeric columns from PR-421
class NonNumericInterpolationTests(SparkTest):
    """Tests for non-numeric column interpolation support"""

    def test_non_numeric_forward_fill(self):
        """Verify that forward fill interpolation works on non-numeric columns."""

        # Load test data from JSON
        tsdf = self.get_test_function_df_builder("test_data").as_tsdf()

        # Apply forward fill to all columns
        # Use leading_margin=1 to include previous values for forward fill
        result_tsdf = interpolate(
            tsdf,
            ["string_col", "bool_col", "int_col"],
            "ffill",
            1,
            0
        )

        result_df = result_tsdf.df.orderBy("event_ts").collect()

        # Verify string column forward fill
        self.assertEqual(result_df[1]["string_col"], "alpha")  # filled from previous
        self.assertEqual(result_df[3]["string_col"], "beta")  # filled from previous

        # Verify boolean column forward fill
        self.assertEqual(result_df[1]["bool_col"], True)  # filled from previous
        self.assertEqual(result_df[3]["bool_col"], False)  # filled from previous

        # Verify int column forward fill
        self.assertEqual(result_df[1]["int_col"], 1)  # filled from previous
        self.assertEqual(result_df[3]["int_col"], 2)  # filled from previous

    def test_non_numeric_backward_fill(self):
        """Verify that backward fill interpolation works on non-numeric columns."""

        # Load test data from JSON
        tsdf = self.get_test_function_df_builder("test_data").as_tsdf()

        # Apply backward fill to all columns
        # Use lagging_margin=1 to include next values for backward fill
        result_tsdf = interpolate(
            tsdf,
            ["string_col", "bool_col", "int_col"],
            "bfill",
            0,
            1
        )

        result_df = result_tsdf.df.orderBy("event_ts").collect()

        # Verify string column backward fill
        self.assertEqual(result_df[1]["string_col"], "beta")  # filled from next
        self.assertEqual(result_df[3]["string_col"], "gamma")  # filled from next

        # Verify boolean column backward fill
        self.assertEqual(result_df[1]["bool_col"], False)  # filled from next
        self.assertEqual(result_df[3]["bool_col"], True)  # filled from next

        # Verify int column backward fill
        self.assertEqual(result_df[1]["int_col"], 2)  # filled from next
        self.assertEqual(result_df[3]["int_col"], 3)  # filled from next

    def test_non_numeric_null_fill(self):
        """Verify that null fill works on non-numeric columns (keeps nulls)."""

        # Load test data from JSON
        tsdf = self.get_test_function_df_builder("test_data").as_tsdf()

        # Apply null fill
        result_tsdf = interpolate(
            tsdf,
            ["string_col", "bool_col", "int_col"],
            "null",
            0,
            0
        )

        result_df = result_tsdf.df.orderBy("event_ts").collect()

        # Verify nulls remain
        self.assertIsNone(result_df[1]["string_col"])
        self.assertIsNone(result_df[1]["bool_col"])
        self.assertIsNone(result_df[1]["int_col"])

    def test_zero_fill_numeric_only(self):
        """Verify that zero fill only works on numeric columns."""

        # Load test data from JSON
        tsdf = self.get_test_function_df_builder("test_data").as_tsdf()

        # Zero fill should raise an error for string column
        with self.assertRaises(ValueError) as context:
            interpolate(tsdf, ["string_col"], "zero", 0, 0)

        self.assertIn("not supported for column 'string_col'", str(context.exception))

        # Zero fill should work for numeric column
        result_tsdf = interpolate(tsdf, ["numeric_col"], "zero", 0, 0)
        result_df = result_tsdf.df.orderBy("event_ts").collect()
        self.assertEqual(result_df[1]["numeric_col"], 0.0)  # filled with zero

    def test_linear_numeric_only(self):
        """Verify that linear interpolation only works on numeric columns."""

        # Load test data from JSON - reuse same data as zero_fill test
        tsdf = self.get_test_function_df_builder("test_data").as_tsdf()

        # Linear interpolation should raise an error for string column
        with self.assertRaises(ValueError) as context:
            interpolate(tsdf, ["string_col"], "linear", 1, 1)

        self.assertIn("not supported for column 'string_col'", str(context.exception))

        # Linear interpolation should work for numeric column
        result_tsdf = interpolate(tsdf, ["numeric_col"], "linear", 1, 1)
        result_df = result_tsdf.df.orderBy("event_ts").collect()
        self.assertAlmostEqual(result_df[1]["numeric_col"], 1.5)  # linear interpolation


class TSDBInterpolationTests(SparkTest):
    """Tests for TSDF.interpolate method"""

    def test_tsdf_interpolate_method(self):
        """Test interpolation through TSDF method"""

        # Load test data from JSON
        tsdf = self.get_test_function_df_builder("test_data").as_tsdf()

        # Test linear interpolation through TSDF method
        result_tsdf = tsdf.interpolate(
            method="linear",
            freq="30 min",
            func="mean"
        )

        result_df = result_tsdf.df.orderBy("event_ts").collect()

        # Verify interpolated values
        self.assertAlmostEqual(result_df[1]["value_a"], 1.5)
        self.assertAlmostEqual(result_df[1]["value_b"], 15.0)
        self.assertAlmostEqual(result_df[3]["value_a"], 2.5)
        self.assertAlmostEqual(result_df[3]["value_b"], 25.0)


# MAIN
if __name__ == "__main__":
    unittest.main()
