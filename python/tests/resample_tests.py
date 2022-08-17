import unittest

from tempo import TSDF
from tempo.resample import (
    checkAllowableFreq,
    _appendAggKey,
    aggregate,
    validateFuncExists,
)
from tests.base import SparkTest


class ResampleUnitTests(SparkTest):
    def test_appendAggKey_freq_is_none(self):
        input_tsdf = self.get_data_as_tsdf("input_data")

        self.assertRaises(TypeError, _appendAggKey, input_tsdf)

    def test_appendAggKey_freq_microsecond(self):
        input_tsdf = self.get_data_as_tsdf("input_data")

        appendAggKey_tuple = _appendAggKey(input_tsdf, "1 MICROSECOND")
        appendAggKey_tsdf = appendAggKey_tuple[0]

        self.assertIsInstance(appendAggKey_tsdf, TSDF)
        self.assertIn("agg_key", appendAggKey_tsdf.df.columns)
        self.assertEqual(appendAggKey_tuple[1], "1")
        self.assertEqual(appendAggKey_tuple[2], "microseconds")

    def test_aggregate_floor(self):
        input_tsdf = self.get_data_as_tsdf("input_data")
        print("input_df")
        input_tsdf.show(truncate=False)
        expected_data = self.get_data_as_sdf("expected_data")
        print("expected_df")
        expected_data.show(truncate=False)

        aggregate_df = aggregate(input_tsdf, "1 DAY", "floor")
        print("aggregate_df")
        aggregate_df.show(truncate=False)

        self.assertDataFrameEquality(
            aggregate_df,
            expected_data,
        )

    def test_aggregate_average(self):
        pass
        # # TODO: DATE returns `null`
        # # DATE is being included in metricCols when metricCols is None
        #   # is this intentional?
        # # resample.py -> lines 86 to 87
        # # occurring in all `func` arguments but causing null values for "mean"
        # input_tsdf = self.get_data_as_tsdf("input_data")
        # expected_data = self.get_data_as_sdf("expected_data")
        #
        # aggregate_df = aggregate(input_tsdf, "1 DAY", "mean")
        #
        # self.assertDataFrameEquality(
        #     aggregate_df,
        #     expected_data,
        # )

    def test_aggregate_min(self):
        input_tsdf = self.get_data_as_tsdf("input_data")
        print("input_df")
        input_tsdf.show(truncate=False)
        expected_data = self.get_data_as_sdf("expected_data")
        print("expected_df")
        expected_data.show(truncate=False)

        aggregate_df = aggregate(input_tsdf, "1 DAY", "min")
        print("aggregate_df")
        aggregate_df.show(truncate=False)

        self.assertDataFrameEquality(
            aggregate_df,
            expected_data,
        )

    def test_aggregate_max(self):
        input_tsdf = self.get_data_as_tsdf("input_data")
        expected_data = self.get_data_as_sdf("expected_data")

        aggregate_df = aggregate(input_tsdf, "1 DAY", "max")

        self.assertDataFrameEquality(
            aggregate_df,
            expected_data,
        )

    def test_aggregate_ceiling(self):
        input_tsdf = self.get_data_as_tsdf("input_data")
        expected_data = self.get_data_as_sdf("expected_data")

        aggregate_df = aggregate(input_tsdf, "1 DAY", "ceil")
        print("aggregated:")
        aggregate_df.show(truncate=False)

        self.assertDataFrameEquality(
            aggregate_df,
            expected_data,
        )

    def test_aggregate_invalid_func_arg(self):
        # TODO : we should not be hitting an UnboundLocalError
        input_tsdf = self.get_data_as_tsdf("input_data")

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
