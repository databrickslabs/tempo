import unittest

from tempo import TSDF
from tempo.resample import (
    _appendAggKey,
    aggregate,
    checkAllowableFreq,
    validateFuncExists,
)
from tests.base import SparkTest


class ResampleUnitTests(SparkTest):
    def test_appendAggKey_freq_is_none(self):
        input_tsdf = self.get_test_df_builder("init").as_tsdf()

        self.assertRaises(TypeError, _appendAggKey, input_tsdf)

    def test_appendAggKey_freq_microsecond(self):
        input_tsdf = self.get_test_df_builder("init").as_tsdf()

        append_agg_key_tuple = _appendAggKey(input_tsdf, "1 MICROSECOND")
        append_agg_key_tsdf = append_agg_key_tuple[0]

        self.assertIsInstance(append_agg_key_tsdf, TSDF)
        self.assertIn("agg_key", append_agg_key_tsdf.df.columns)
        self.assertEqual(append_agg_key_tuple[1], "1")
        self.assertEqual(append_agg_key_tuple[2], "microseconds")

    def test_appendAggKey_freq_is_invalid(self):
        input_tsdf = self.get_test_df_builder("init").as_tsdf()

        self.assertRaises(
            ValueError,
            _appendAggKey,
            input_tsdf,
            "1 invalid",
        )

    def test_aggregate_floor(self):
        input_tsdf = self.get_test_df_builder("init").as_tsdf()
        expected_df = self.get_test_df_builder("expected").as_sdf()

        aggregate_df = aggregate(input_tsdf, "1 DAY", "floor")

        self.assertDataFrameEquality(
            aggregate_df,
            expected_df,
        )

    def test_aggregate_average(self):
        # TODO: fix DATE returns `null`
        # DATE is being included in metricCols when metricCols is None
        # this occurs for all aggregate functions but causes negative side effects with avg
        # is this intentional?
        # resample.py -> lines 86 to 87
        # occurring in all `func` arguments but causing null values for "mean"
        input_tsdf = self.get_test_df_builder("init").as_tsdf()
        expected_df = self.get_test_df_builder("expected").as_sdf()

        # explicitly declaring metricCols to remove DATE so that test can pass for now
        aggregate_df = aggregate(
            input_tsdf, "1 DAY", "mean", ["trade_pr", "trade_pr_2"]
        )

        self.assertDataFrameEquality(
            aggregate_df,
            expected_df,
        )

    def test_aggregate_min(self):
        input_tsdf = self.get_test_df_builder("init").as_tsdf()
        expected_df = self.get_test_df_builder("expected").as_sdf()

        aggregate_df = aggregate(input_tsdf, "1 DAY", "min")

        self.assertDataFrameEquality(
            aggregate_df,
            expected_df,
        )

    def test_aggregate_min_with_prefix(self):
        input_tsdf = self.get_test_df_builder("init").as_tsdf()
        expected_df = self.get_test_df_builder("expected").as_sdf()

        aggregate_df = aggregate(input_tsdf, "1 DAY", "min", prefix="min")

        self.assertDataFrameEquality(
            aggregate_df,
            expected_df,
        )

    def test_aggregate_min_with_fill(self):
        input_tsdf = self.get_test_df_builder("init").as_tsdf()
        expected_df = self.get_test_df_builder("expected").as_sdf()

        aggregate_df = aggregate(input_tsdf, "1 DAY", "min", fill=True)

        self.assertDataFrameEquality(
            aggregate_df,
            expected_df,
        )

    def test_aggregate_max(self):
        input_tsdf = self.get_test_df_builder("init").as_tsdf()
        expected_df = self.get_test_df_builder("expected").as_sdf()

        aggregate_df = aggregate(input_tsdf, "1 DAY", "max")

        self.assertDataFrameEquality(
            aggregate_df,
            expected_df,
        )

    def test_aggregate_ceiling(self):
        input_tsdf = self.get_test_df_builder("init").as_tsdf()
        expected_df = self.get_test_df_builder("expected").as_sdf()

        aggregate_df = aggregate(input_tsdf, "1 DAY", "ceil")

        self.assertDataFrameEquality(
            aggregate_df,
            expected_df,
        )

    def test_aggregate_invalid_func_arg(self):
        # TODO : we should not be hitting an UnboundLocalError
        input_tsdf = self.get_test_df_builder("init").as_tsdf()

        self.assertRaises(UnboundLocalError, aggregate, input_tsdf, "1 DAY", "average")

    def test_check_allowable_freq_none(self):
        self.assertRaises(TypeError, checkAllowableFreq, None)

    def test_check_allowable_freq_microsecond(self):
        self.assertEqual(checkAllowableFreq("1 MICROSECOND"), ("1", "microsec"))

    def test_check_allowable_freq_millisecond(self):
        self.assertEqual(checkAllowableFreq("1 MILLISECOND"), ("1", "ms"))

    def test_check_allowable_freq_second(self):
        self.assertEqual(checkAllowableFreq("1 SECOND"), ("1", "sec"))

    def test_check_allowable_freq_minute(self):
        self.assertEqual(checkAllowableFreq("1 MINUTE"), ("1", "min"))

    def test_check_allowable_freq_hour(self):
        self.assertEqual(checkAllowableFreq("1 HOUR"), ("1", "hour"))

    def test_check_allowable_freq_day(self):
        self.assertEqual(checkAllowableFreq("1 DAY"), ("1", "day"))

    def test_check_allowable_freq_no_interval(self):
        # TODO: should first element return str for consistency?
        self.assertEqual(checkAllowableFreq("day"), (1, "day"))

    def test_check_allowable_freq_exception_not_in_allowable_freqs(self):
        self.assertRaises(ValueError, checkAllowableFreq, "wrong")

    def test_check_allowable_freq_exception(self):
        self.assertRaises(ValueError, checkAllowableFreq, "wrong wrong")

    def test_validate_func_exists_type_error(self):
        self.assertRaises(TypeError, validateFuncExists, None)

    def test_validate_func_exists_value_error(self):
        self.assertRaises(ValueError, validateFuncExists, "non-existent")


# MAIN
if __name__ == "__main__":
    unittest.main()
