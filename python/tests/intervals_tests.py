from unittest import TestCase

import numpy as np
import pandas as pd
import pyspark.sql
import pyspark.sql.functions as f
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.utils import AnalysisException

from tempo.intervals import (
    Interval,
    IntervalsDF,
    OverlapResolver,
    identify_interval_overlaps,
    update_interval_boundary,
    merge_metric_columns_of_intervals,
    resolve_all_overlaps,
    add_as_disjoint,
    make_disjoint_wrap,
)
from tests.tsdf_tests import SparkTest


class IntervalsDFTests(SparkTest):
    union_tests_dict_input = [
        {
            "start_ts": "2020-08-01 00:00:09",
            "end_ts": "2020-08-01 00:00:14",
            "series_1": "v1",
            "metric_1": 5,
            "metric_2": None,
        },
        {
            "start_ts": "2020-08-01 00:00:09",
            "end_ts": "2020-08-01 00:00:11",
            "series_1": "v1",
            "metric_1": None,
            "metric_2": 0,
        },
        {
            "start_ts": "2020-08-01 00:00:09",
            "end_ts": "2020-08-01 00:00:12",
            "series_1": "v1",
            "metric_1": None,
            "metric_2": 4,
        },
        {
            "start_ts": "2020-08-01 00:00:09",
            "end_ts": "2020-08-01 00:00:14",
            "series_1": "v1",
            "metric_1": 5,
            "metric_2": None,
        },
        {
            "start_ts": "2020-08-01 00:00:09",
            "end_ts": "2020-08-01 00:00:11",
            "series_1": "v1",
            "metric_1": None,
            "metric_2": 0,
        },
        {
            "start_ts": "2020-08-01 00:00:09",
            "end_ts": "2020-08-01 00:00:12",
            "series_1": "v1",
            "metric_1": None,
            "metric_2": 4,
        },
    ]

    def test_init_series_str(self):
        df_input = self.get_test_df_builder("init").as_sdf()

        idf = IntervalsDF(df_input, "start_ts", "end_ts", "series_1")

        self.assertIsInstance(idf, IntervalsDF)
        self.assertIsInstance(idf.df, DataFrame)
        self.assertEqual(idf.start_ts, "start_ts")
        self.assertEqual(idf.end_ts, "end_ts")
        self.assertEqual(idf.interval_boundaries, ["start_ts", "end_ts"])
        self.assertCountEqual(idf.series_ids, ["series_1"])
        self.assertCountEqual(
            idf.structural_columns, ["start_ts", "end_ts", "series_1"]
        )
        self.assertCountEqual(idf.observational_columns, ["metric_1", "metric_2"])
        self.assertCountEqual(idf.metric_columns, ["metric_1", "metric_2"])

    def test_init_series_comma_seperated_str(self):
        df_input = self.get_test_df_builder("init").as_sdf()

        idf = IntervalsDF(df_input, "start_ts", "end_ts", "series_1, series_2")

        self.assertIsInstance(idf, IntervalsDF)
        self.assertIsInstance(idf.df, DataFrame)
        self.assertEqual(idf.start_ts, "start_ts")
        self.assertEqual(idf.end_ts, "end_ts")
        self.assertEqual(idf.interval_boundaries, ["start_ts", "end_ts"])
        self.assertCountEqual(idf.series_ids, ["series_1", "series_2"])
        self.assertCountEqual(
            idf.structural_columns, ["start_ts", "end_ts", "series_1", "series_2"]
        )
        self.assertCountEqual(idf.observational_columns, ["metric_1", "metric_2"])
        self.assertCountEqual(idf.metric_columns, ["metric_1", "metric_2"])

    def test_init_series_tuple(self):
        df_input = self.get_test_df_builder("init").as_sdf()

        idf = IntervalsDF(df_input, "start_ts", "end_ts", ("series_1",))

        self.assertIsInstance(idf, IntervalsDF)
        self.assertIsInstance(idf.df, DataFrame)
        self.assertEqual(idf.start_ts, "start_ts")
        self.assertEqual(idf.end_ts, "end_ts")
        self.assertEqual(idf.interval_boundaries, ["start_ts", "end_ts"])
        self.assertCountEqual(idf.series_ids, ["series_1"])
        self.assertCountEqual(
            idf.structural_columns, ["start_ts", "end_ts", "series_1"]
        )
        self.assertCountEqual(idf.observational_columns, ["metric_1", "metric_2"])
        self.assertCountEqual(idf.metric_columns, ["metric_1", "metric_2"])

    def test_init_series_list(self):
        df_input = self.get_test_df_builder("init").as_sdf()

        idf = IntervalsDF(df_input, "start_ts", "end_ts", ["series_1"])

        self.assertIsInstance(idf, IntervalsDF)
        self.assertIsInstance(idf.df, DataFrame)
        self.assertEqual(idf.start_ts, "start_ts")
        self.assertEqual(idf.end_ts, "end_ts")
        self.assertEqual(idf.interval_boundaries, ["start_ts", "end_ts"])
        self.assertCountEqual(idf.series_ids, ["series_1"])
        self.assertCountEqual(
            idf.structural_columns, ["start_ts", "end_ts", "series_1"]
        )
        self.assertCountEqual(idf.observational_columns, ["metric_1", "metric_2"])
        self.assertCountEqual(idf.metric_columns, ["metric_1", "metric_2"])

    def test_init_series_none(self):
        df_input = self.get_test_df_builder("init").as_sdf()

        idf = IntervalsDF(df_input, "start_ts", "end_ts", None)

        self.assertIsInstance(idf, IntervalsDF)
        self.assertIsInstance(idf.df, DataFrame)
        self.assertEqual(idf.start_ts, "start_ts")
        self.assertEqual(idf.end_ts, "end_ts")
        self.assertEqual(idf.interval_boundaries, ["start_ts", "end_ts"])
        self.assertCountEqual(idf.series_ids, [])
        self.assertCountEqual(idf.structural_columns, ["start_ts", "end_ts"])
        self.assertCountEqual(
            idf.observational_columns, ["series_1", "metric_1", "metric_2"]
        )
        self.assertCountEqual(idf.metric_columns, ["metric_1", "metric_2"])

    def test_init_series_int(self):
        df_input = self.get_test_df_builder("init").as_sdf()

        self.assertRaises(
            ValueError,
            IntervalsDF,
            df_input,
            "start_ts",
            "end_ts",
            1,
        )

    def test_window_property(self):
        idf: IntervalsDF = self.get_test_df_builder("init").as_idf()

        self.assertIsInstance(idf.window, pyspark.sql.window.WindowSpec)

    def test_fromStackedMetrics_series_str(self):
        df_input = self.get_test_df_builder("init").as_sdf()

        self.assertRaises(
            ValueError,
            IntervalsDF.fromStackedMetrics,
            df_input,
            "start_ts",
            "end_ts",
            "series_1",
            "metric_name",
            "metric_value",
        )

    def test_fromStackedMetrics_series_tuple(self):
        df_input = self.get_test_df_builder("init").as_sdf()

        self.assertRaises(
            ValueError,
            IntervalsDF.fromStackedMetrics,
            df_input,
            "start_ts",
            "end_ts",
            ("series_1",),
            "metric_name",
            "metric_value",
        )

    def test_fromStackedMetrics_series_list(self):
        df_input = self.get_test_df_builder("init").as_sdf()
        idf_expected = self.get_test_df_builder("expected").as_idf()

        df_input = df_input.withColumn(
            "start_ts", f.to_timestamp("start_ts")
        ).withColumn("end_ts", f.to_timestamp("end_ts"))

        idf = IntervalsDF.fromStackedMetrics(
            df_input,
            "start_ts",
            "end_ts",
            [
                "series_1",
            ],
            "metric_name",
            "metric_value",
        )

        self.assertDataFrameEquality(idf, idf_expected)

    def test_fromStackedMetrics_metric_names(self):
        df_input = self.get_test_df_builder("init").as_sdf()
        idf_expected = self.get_test_df_builder("expected").as_idf()

        df_input = df_input.withColumn(
            "start_ts", f.to_timestamp("start_ts")
        ).withColumn("end_ts", f.to_timestamp("end_ts"))

        idf = IntervalsDF.fromStackedMetrics(
            df_input,
            "start_ts",
            "end_ts",
            [
                "series_1",
            ],
            "metric_name",
            "metric_value",
            ["metric_1", "metric_2"],
        )

        self.assertDataFrameEquality(idf, idf_expected)

    def test_make_disjoint(self):
        idf_input = self.get_test_df_builder("init").as_idf()
        idf_expected = self.get_test_df_builder("expected").as_idf()

        idf_actual = idf_input.make_disjoint()

        self.assertDataFrameEquality(
            idf_expected, idf_actual, ignore_row_order=True
        )

    def test_make_disjoint_contains_interval_already_disjoint(self):
        idf_input = self.get_test_df_builder("init").as_idf()
        idf_expected = self.get_test_df_builder("expected").as_idf()
        print("expected")
        print(idf_expected.df.toPandas())

        idf_actual = idf_input.make_disjoint()

        self.assertDataFrameEquality(
            idf_expected, idf_actual, ignore_row_order=True
        )

    def test_make_disjoint_contains_intervals_equal(self):
        idf_input = self.get_test_df_builder("init").as_idf()
        idf_expected = self.get_test_df_builder("expected").as_idf()

        idf_actual = idf_input.make_disjoint()

        self.assertDataFrameEquality(
            idf_expected, idf_actual, ignore_row_order=True
        )

    def test_make_disjoint_intervals_same_start(self):
        idf_input = self.get_test_df_builder("init").as_idf()
        idf_expected = self.get_test_df_builder("expected").as_idf()

        idf_actual = idf_input.make_disjoint()

        self.assertDataFrameEquality(
            idf_expected, idf_actual, ignore_row_order=True
        )

    def test_make_disjoint_intervals_same_end(self):
        idf_input = self.get_test_df_builder("init").as_idf()
        idf_expected = self.get_test_df_builder("expected").as_idf()

        idf_actual = idf_input.make_disjoint()

        self.assertDataFrameEquality(
            idf_expected, idf_actual, ignore_row_order=True
        )

    def test_make_disjoint_multiple_series(self):
        idf_input = self.get_test_df_builder("init").as_idf()
        idf_expected = self.get_test_df_builder("expected").as_idf()

        idf_actual = idf_input.make_disjoint()

        self.assertDataFrameEquality(
            idf_expected, idf_actual, ignore_row_order=True
        )

    def test_make_disjoint_single_metric(self):
        idf_input = self.get_test_df_builder("init").as_idf()
        idf_expected = self.get_test_df_builder("expected").as_idf()

        idf_actual = idf_input.make_disjoint()

        self.assertDataFrameEquality(
            idf_expected, idf_actual, ignore_row_order=True
        )

    def test_make_disjoint_interval_is_subset(self):
        idf_input = self.get_test_df_builder("init").as_idf()
        idf_expected = self.get_test_df_builder("expected").as_idf()

        idf_actual = idf_input.make_disjoint()

        self.assertDataFrameEquality(
            idf_expected, idf_actual, ignore_row_order=True
        )

    def test_union_other_idf(self):
        idf_input_1 = self.get_test_df_builder("init").as_idf()
        idf_input_2 = self.get_test_df_builder("init").as_idf()

        count_idf_1 = idf_input_1.df.count()
        count_idf_2 = idf_input_2.df.count()

        union_idf = idf_input_1.union(idf_input_2)

        count_union = union_idf.df.count()

        self.assertEqual(count_idf_1 + count_idf_2, count_union)

    def test_union_other_df(self):
        idf_input = self.get_test_df_builder("init").as_idf()
        df_input = self.get_test_df_builder("init").as_sdf()

        self.assertRaises(TypeError, idf_input.union, df_input)

    def test_union_other_list_dicts(self):
        idf_input = self.get_test_df_builder("init").as_idf()

        self.assertRaises(
            TypeError, idf_input.union, IntervalsDFTests.union_tests_dict_input
        )

    def test_unionByName_other_idf(self):
        idf_input_1 = self.get_test_df_builder("init").as_idf()
        idf_input_2 = self.get_test_df_builder("init").as_idf()

        count_idf_1 = idf_input_1.df.count()
        count_idf_2 = idf_input_2.df.count()

        union_idf = idf_input_1.unionByName(idf_input_2)

        count_union_by_name = union_idf.df.count()

        self.assertEqual(count_idf_1 + count_idf_2, count_union_by_name)

    def test_unionByName_other_df(self):
        idf_input = self.get_test_df_builder("init").as_idf()
        df_input = self.get_test_df_builder("init").as_sdf()

        self.assertRaises(TypeError, idf_input.unionByName, df_input)

    def test_unionByName_other_list_dicts(self):
        idf_input = self.get_test_df_builder("init").as_idf()

        self.assertRaises(
            TypeError, idf_input.unionByName, IntervalsDFTests.union_tests_dict_input
        )

    def test_unionByName_extra_column(self):
        idf_extra_col = self.get_test_df_builder("init_extra_col").as_idf()
        idf_input = self.get_test_df_builder("init").as_idf()

        self.assertRaises(AnalysisException, idf_extra_col.unionByName, idf_input)

    def test_unionByName_other_extra_column(self):
        idf_input = self.get_test_df_builder("init").as_idf()
        idf_extra_col = self.get_test_df_builder("init_extra_col").as_idf()

        self.assertRaises(AnalysisException, idf_input.unionByName, idf_extra_col)

    def test_toDF(self):
        # NB: init is used for both since the expected df is the same
        idf_input = self.get_test_df_builder("init").as_idf()
        expected_df = self.get_test_df_builder("init").as_sdf()

        actual_df = idf_input.toDF()

        self.assertDataFrameEquality(actual_df, expected_df)

    def test_toDF_stack(self):
        idf_input = self.get_test_df_builder("init").as_idf()
        expected_df = self.get_test_df_builder("expected").as_sdf()

        expected_df = expected_df.withColumn(
            "start_ts", f.to_timestamp("start_ts")
        ).withColumn("end_ts", f.to_timestamp("end_ts"))

        actual_df = idf_input.toDF(stack=True)

        self.assertDataFrameEquality(actual_df, expected_df)

    def test_make_disjoint_issue_268(self):
        # https://github.com/databrickslabs/tempo/issues/268

        idf_input = self.get_test_df_builder("init").as_idf()
        idf_expected = self.get_test_df_builder("expected").as_idf()

        idf_actual = idf_input.make_disjoint()
        idf_actual.df.show(truncate=False)

        self.assertDataFrameEquality(
            idf_expected, idf_actual, ignore_row_order=True
        )


class PandasFunctionTests(TestCase):
    def test_identify_interval_overlaps_df_empty(self):
        df = pd.DataFrame()
        row = pd.Series({"start": "2023-01-01T00:00:01", "end": "2023-01-01T00:00:05"})
        result = identify_interval_overlaps(df, Interval(row, "start", "end"))
        self.assertTrue(result.empty)

    def test_identify_interval_overlaps_overlapping_intervals(self):
        df = pd.DataFrame(
            {
                "start": [
                    "2023-01-01T00:00:01",
                    "2023-01-01T00:00:04",
                    "2023-01-01T00:00:07",
                ],
                "end": [
                    "2023-01-01T00:00:05",
                    "2023-01-01T00:00:08",
                    "2023-01-01T00:00:10",
                ],
            }
        )
        row = pd.Series({"start": "2023-01-01T00:00:03", "end": "2023-01-01T00:00:06"})
        result = identify_interval_overlaps(df, Interval(row, "start", "end"))
        expected = pd.DataFrame(
            {
                "start": ["2023-01-01T00:00:01", "2023-01-01T00:00:04"],
                "end": ["2023-01-01T00:00:05", "2023-01-01T00:00:08"],
            }
        )
        self.assertEqual(len(result), 2)
        self.assertTrue(result.equals(expected))

    def test_identify_interval_overlaps_no_overlapping_intervals(self):
        df = pd.DataFrame(
            {
                "start": [
                    "2023-01-01T00:00:01",
                    "2023-01-01T00:00:02",
                    "2023-01-01T00:00:03",
                ],
                "end": [
                    "2023-01-01T00:00:02",
                    "2023-01-01T00:00:03",
                    "2023-01-01T00:00:04",
                ],
            }
        )
        row = pd.Series(
            {"start": "2023-01-01T00:00:04.1", "end": "2023-01-01T00:00:05"}
        )
        result = identify_interval_overlaps(df, Interval(row, "start", "end"))
        self.assertTrue(result.empty)

    def test_identify_interval_overlaps_interval_subset(self):
        df = pd.DataFrame(
            {
                "start": [
                    "2023-01-01T00:00:01",
                    "2023-01-01T00:00:05",
                    "2023-01-01T00:00:08",
                ],
                "end": [
                    "2023-01-01T00:00:10",
                    "2023-01-01T00:00:07",
                    "2023-01-01T00:00:11",
                ],
            }
        )
        row = pd.Series({"start": "2023-01-01T00:00:02", "end": "2023-01-01T00:00:04"})
        result = identify_interval_overlaps(df, Interval(row, "start", "end"))
        expected = pd.DataFrame(
            {"start": ["2023-01-01T00:00:01"], "end": ["2023-01-01T00:00:10"]}
        )
        self.assertEqual(len(result), 1)
        self.assertTrue(result.equals(expected))

    def test_identify_interval_overlaps_identical_start_end(self):
        df = pd.DataFrame(
            {"start": ["2023-01-01T00:00:02"], "end": ["2023-01-01T00:00:05"]}
        )
        row = pd.Series({"start": "2023-01-01T00:00:02", "end": "2023-01-01T00:00:05"})
        result = identify_interval_overlaps(df, Interval(row, "start", "end"))
        self.assertTrue(result.empty)

    # TODO: uncomment once this method is moved into the Intervals Class

    # def test_check_for_nan_values_pd_series_with_nan(self):
    #     s = pd.Series([1, 2, float("nan"), 4])
    #     self.assertTrue(check_for_nan_values(s))
    #
    # def test_check_for_nan_values_pd_series_without_nan(self):
    #     s = pd.Series([1, 2, 3, 4])
    #     self.assertFalse(check_for_nan_values(s))
    #
    # def test_check_for_nan_values_pd_df_with_nan(self):
    #     df = pd.DataFrame({"A": [1, 2, 3], "B": ["a", float("nan"), "c"]})
    #     self.assertTrue(check_for_nan_values(df))
    #
    # def test_check_for_nan_values_pd_df_without_nan(self):
    #     df = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})
    #     self.assertFalse(check_for_nan_values(df))
    #
    # def test_check_for_nan_values_np_array_with_nan(self):
    #     arr = np.array([1, 2, float("nan"), 4])
    #     self.assertTrue(check_for_nan_values(arr))
    #
    # def test_check_for_nan_values_np_array_without_nan(self):
    #     arr = np.array([1, 2, 3, 4])
    #     self.assertFalse(check_for_nan_values(arr))
    #
    # def test_check_for_nan_values_np_scalar_nan(self):
    #     scalar = np.float64(float("nan"))
    #     self.assertTrue(check_for_nan_values(scalar))
    #
    # def test_check_for_nan_values_np_scalar_value(self):
    #     scalar = np.float64(5.0)
    #     self.assertFalse(check_for_nan_values(scalar))
    #
    # def test_check_for_nan_values_str(self):
    #     string = "valid"
    #     self.assertFalse(check_for_nan_values(string))
    #
    # def test_check_for_nan_values_none(self):
    #     self.assertTrue(check_for_nan_values(None))

    def test_interval_starts_before_other(self):
        interval = Interval(pd.Series({"start": "2023-01-01T00:00:01", "end": "2023-01-01T00:00:03"}), "start", "end")
        other = Interval(pd.Series({"start": "2023-01-01T00:00:02", "end": "2023-01-01T00:00:04"}), "start", "end")
        resolver = OverlapResolver(interval, other)
        result = resolver.check_interval_starts_first()
        self.assertTrue(result)

    def test_interval_starts_same_as_other(self):
        interval = Interval(pd.Series({"start": "2023-01-01T00:00:01", "end": "2023-01-01T00:00:03"}), "start", "end")
        other = Interval(pd.Series({"start": "2023-01-01T00:00:01", "end": "2023-01-01T00:00:04"}), "start", "end")
        resolver = OverlapResolver(interval, other)
        result = resolver.check_interval_starts_first()
        self.assertFalse(result)

    # def test_interval_starts_after_other(self):
    #     other = Interval(pd.Series({"start": "2023-01-01T00:00:03", "end": "2023-01-01T00:00:04"}), "start", "end")
    #     interval = Interval(pd.Series({"start": "2023-01-01T00:00:02", "end": "2023-01-01T00:00:03"}), "start", "end")
    #     resolver = OverlapResolver(interval, other)
    #     result = resolver.does_interval_start_before()
    #     self.assertFalse(result)

    def test_start_nan_boundary(self):
        self.assertRaises(
            TypeError,
            Interval,
            pd.Series({"start": float("nan")}),
            "start",
            "end",
        )

    def test_end_nan_boundary(self):
        self.assertRaises(
            TypeError,
            Interval,
            pd.Series({"start": float("nan"), "end": "2023-01-01T00:00:03"}),
            "start",
            "end",
        )

    def test_none_end_boundary(self):
        self.assertRaises(
            TypeError,
            Interval,
            pd.Series(
                {"start": "2022-01-02", "end": None, "metric_1": 6, "metric_2": 11}
            ),
            "start",
            "end",

        )

    def test_none_start_boundary(self):
        self.assertRaises(
            TypeError,
            Interval,
            pd.Series(
                {"start": None, "end": "2022-01-01", "metric_1": 5, "metric_2": 10}
            ),
            "start",
            "end",

        )

    def test_interval_ends_before_other(self):
        interval = Interval(pd.Series({"start": "2023-01-01T00:00:00", "end": "2023-01-01T00:00:01"}), "start", "end")
        other = Interval(pd.Series({"start": "2023-01-01T00:00:00", "end": "2023-01-01T00:00:02"}), "start", "end")
        resolver = OverlapResolver(interval, other)
        result = resolver.check_interval_ends_first()
        self.assertTrue(result)

    def test_interval_ends_same_as_other(self):
        interval = Interval(pd.Series({"start": "2023-01-01T00:00:00", "end": "2023-01-01T00:00:01"}), "start", "end")
        other = Interval(pd.Series({"start": "2023-01-01T00:00:00", "end": "2023-01-01T00:00:01"}), "start", "end")
        resolver = OverlapResolver(interval, other)
        result = resolver.check_interval_ends_first(
        )
        self.assertFalse(result)

    def test_interval_ends_after_other(self):
        interval = Interval(pd.Series({"start": "2023-01-01T00:00:01", "end": "2023-01-01T00:00:03"}), "start", "end")
        other = Interval(pd.Series({"start": "2023-01-01T00:00:01", "end": "2023-01-01T00:00:02"}), "start", "end")
        resolver = OverlapResolver(interval, other)
        result = resolver.check_interval_ends_first()
        self.assertFalse(result)

    def test_interval_contained_in_other(self):
        interval = Interval(pd.Series(
            {"start": "2023-01-01T00:00:00", "end": "2023-01-01T03:00:00"}
        ), "start", "end")
        other = Interval(pd.Series(
            {"start": "2023-01-01T01:00:00", "end": "2023-01-01T02:00:00"}
        ), "start", "end")
        resolver = OverlapResolver(interval, other)
        result = resolver.check_interval_contained()
        self.assertFalse(result)

    def test_interval_starts_before_other_not_contained(self):
        interval = Interval(pd.Series(
            {"start": "2023-01-01T00:00:00", "end": "2023-01-01T01:30:00"}
        ), "start", "end")
        other = Interval(pd.Series(
            {"start": "2023-01-01T01:00:00", "end": "2023-01-01T03:00:00"}
        ), "start", "end")
        resolver = OverlapResolver(interval, other)
        result = resolver.check_interval_contained()
        self.assertFalse(result)

    def test_interval_ends_after_other_not_contained(self):
        interval = Interval(pd.Series(
            {"start": "2023-01-01T02:00:00", "end": "2023-01-01T04:00:00"}
        ), "start", "end")
        other = Interval(pd.Series(
            {"start": "2023-01-01T01:00:00", "end": "2023-01-01T05:00:00"}
        ), "start", "end")
        resolver = OverlapResolver(interval, other)
        result = resolver.check_interval_contained()
        self.assertFalse(result)

    def test_interval_outside_other_not_contained(self):
        interval = Interval(pd.Series(
            {"start": "2023-01-01T00:00:00", "end": "2023-01-01T01:00:00"}
        ), "start", "end")
        other = Interval(pd.Series(
            {"start": "2023-01-01T02:00:00", "end": "2023-01-01T03:00:00"}
        ), "start", "end")
        resolver = OverlapResolver(interval, other)
        result = resolver.check_interval_contained()
        self.assertFalse(result)

    def test_intervals_share_start_boundary(self):
        interval = Interval(pd.Series({"start": "2023-01-01T01:00:00", "end": "2023-01-01T01:30:00"}), "start", "end")
        other = Interval(pd.Series({"start": "2023-01-01T01:00:00", "end": "2023-01-01T01:30:00"}), "start", "end")
        resolver = OverlapResolver(interval, other)
        result = resolver.check_starts_equivalent()
        self.assertTrue(result)

    def test_intervals_share_start_boundary_with_different_start(self):
        interval = Interval(pd.Series({"start": "2023-01-01T01:00:00", "end": "2023-01-01T01:30:00"}), "start", "end")
        other = Interval(pd.Series({"start": "2023-01-01T02:00:00", "end": "2023-01-01T02:30:00"}), "start", "end")
        resolver = OverlapResolver(interval, other)
        result = resolver.check_starts_equivalent()
        self.assertFalse(result)

    def test_intervals_share_start_boundary_with_default_other_timestamp(self):
        interval = Interval(pd.Series({"start": "2023-01-01T01:00:00", "end": "2023-01-01T01:30:00"}), "start", "end")
        other = Interval(pd.Series({"start": "2023-01-01T01:00:00", "end": "2023-01-01T01:30:00"}), "start", "end")
        resolver = OverlapResolver(interval, other)
        result = resolver.check_starts_equivalent()
        self.assertTrue(result)

    def test_intervals_share_start_boundary_with_defined_other_timestamp(self):
        interval = Interval(pd.Series({"start": "2023-01-01T01:00:00", "end": "2023-01-01T01:30:00"}), "start", "end")
        other = Interval(pd.Series({"start": "2023-01-01T01:00:00", "end": "2023-01-01T01:30:00"}), "start", "end")
        resolver = OverlapResolver(interval, other)
        result = resolver.check_starts_equivalent()
        self.assertTrue(result)

    def test_intervals_share_end_boundary(self):
        interval = Interval(pd.Series({"start": "2023-01-01T00:00:00", "end": "2023-01-01T01:00:00"}), "start", "end")
        other = Interval(pd.Series({"start": "2023-01-01T00:30:00", "end": "2023-01-01T01:00:00"}), "start", "end")
        resolver = OverlapResolver(interval, other)
        result = resolver.check_ends_equivalent()
        self.assertTrue(result)

    def test_intervals_share_end_boundary_with_different_end(self):
        interval = Interval(pd.Series({"start": "2023-01-01T00:00:00", "end": "2023-01-01T01:00:00"}), "start", "end")
        other = Interval(pd.Series({"start": "2023-01-01T00:30:00", "end": "2023-01-01T02:00:00"}), "start", "end")
        resolver = OverlapResolver(interval, other)
        result = resolver.check_ends_equivalent()
        self.assertFalse(result)

    def test_intervals_share_end_boundary_with_default_other_timestamp(self):
        interval = Interval(pd.Series({"start": "2023-01-01T00:00:00", "end": "2023-01-01T01:00:00"}), "start", "end")
        other = Interval(pd.Series({"start": "2023-01-01T00:00:00", "end": "2023-01-01T01:00:00"}), "start", "end")
        resolver = OverlapResolver(interval, other)
        result = resolver.check_ends_equivalent()
        self.assertTrue(result)

    def test_intervals_share_end_boundary_with_defined_other_timestamp(self):
        interval = Interval(pd.Series({"start": "2023-01-01T00:00:00", "end": "2023-01-01T01:00:00"}), "start", "end")
        other = Interval(pd.Series({"start": "2023-01-01T00:00:00", "end": "2023-01-01T01:00:00"}), "start", "end")
        resolver = OverlapResolver(interval, other)
        result = resolver.check_ends_equivalent()
        self.assertTrue(result)

    def test_intervals_boundaries_are_equivalent(self):
        interval = Interval(pd.Series(
            {"start": "2023-01-01T01:00:00", "end": "2023-01-01T02:00:00"}
        ), "start", "end")
        other = Interval(pd.Series(
            {"start": "2023-01-01T01:00:00", "end": "2023-01-01T02:00:00"}
        ), "start", "end")
        resolver = OverlapResolver(interval, other)
        result = resolver.check_boundaries_equivalent()
        self.assertTrue(result)

    def test_intervals_boundaries_are_equivalent_with_different_start(self):
        interval = Interval(pd.Series(
            {"start": "2023-01-01T01:00:00", "end": "2023-01-01T02:00:00"}
        ), "start", "end")
        other = Interval(pd.Series(
            {"start": "2023-01-01T01:30:00", "end": "2023-01-01T02:00:00"}
        ), "start", "end")
        resolver = OverlapResolver(interval, other)
        result = resolver.check_boundaries_equivalent()
        self.assertFalse(result)

    def test_intervals_boundaries_are_equivalent_with_different_end(self):
        interval = Interval(pd.Series(
            {"start": "2023-01-01T01:00:00", "end": "2023-01-01T02:30:00"}
        ), "start", "end")
        other = Interval(pd.Series(
            {"start": "2023-01-01T01:00:00", "end": "2023-01-01T02:00:00"}
        ), "start", "end")
        resolver = OverlapResolver(interval, other)
        result = resolver.check_boundaries_equivalent()
        self.assertFalse(result)

    def test_intervals_boundaries_are_equivalent_with_default_other_timestamps(self):
        interval = Interval(pd.Series(
            {"start": "2023-01-01T01:00:00", "end": "2023-01-01T02:00:00"}
        ), "start", "end")
        other = Interval(pd.Series(
            {"start": "2023-01-01T01:00:00", "end": "2023-01-01T02:00:00"}
        ), "start", "end")
        resolver = OverlapResolver(interval, other)
        result = resolver.check_boundaries_equivalent()
        self.assertTrue(result)

    def test_intervals_boundaries_are_equivalent_with_defined_other_timestamps(self):
        interval = Interval(pd.Series(
            {"start": "2023-01-01T01:00:00", "end": "2023-01-01T02:00:00"}
        ), "start", "end")
        other = Interval(pd.Series(
            {"start": "2023-01-01T01:00:00", "end": "2023-01-01T02:00:00"}
        ), "start", "end")
        resolver = OverlapResolver(interval, other)
        result = resolver.check_boundaries_equivalent()
        self.assertTrue(result)

    def test_update_interval_boundary_start(self):
        interval = pd.Series(
            {"start": "2023-01-01T01:00:00", "end": "2023-01-01T02:00:00"}
        )
        updated = update_interval_boundary(
            interval=interval,
            boundary_key="start",
            update_value="2023-01-01T01:30:00",
        )
        self.assertEqual(updated["start"], "2023-01-01T01:30:00")

    def test_update_interval_boundary_end(self):
        interval = pd.Series(
            {"start": "2023-01-01T01:00:00", "end": "2023-01-01T02:00:00"}
        )
        updated = update_interval_boundary(
            interval=interval,
            boundary_key="end",
            update_value="2023-01-01T02:30:00",
        )
        self.assertEqual(updated["end"], "2023-01-01T02:30:00")

    def test_update_interval_boundary_return_new_copy(self):
        interval = pd.Series(
            {"start": "2023-01-01T01:00:00", "end": "2023-01-01T02:00:00"}
        )
        updated = update_interval_boundary(
            interval=interval,
            boundary_key="start",
            update_value="2023-01-01T01:30:00",
        )
        self.assertNotEqual(id(interval), id(updated))
        self.assertEqual(interval["start"], "2023-01-01T01:00:00")

    def test_update_interval_boundary_non_existent_boundary(self):
        interval = pd.Series(
            {"start": "2023-01-01T01:00:00", "end": "2023-01-01T02:00:00"}
        )
        self.assertRaises(
            KeyError,
            update_interval_boundary,
            interval=interval,
            boundary_key="not_a_boundary",
            update_value="2023-01-01T01:30:00",
        )

    # NB: `mertric_merge_method = False` is a placeholder to allow
    # user-defined merge methods in the future and is currently a
    # no-op.
    def test_merge_metric_columns_of_intervals_with_list_metric_merge_false(self):
        main = Interval(pd.Series({"start": "01:00", "end": "02:00", "value": 10}), "start", "end")
        child = Interval(pd.Series({"start": "01:00", "end": "02:00", "value": 20}), "start", "end")
        expected = pd.Series({"start": "01:00", "end": "02:00", "value": 10})
        merged = merge_metric_columns_of_intervals(
            interval=main,
            other=child,

        )
        self.assertTrue(merged.equals(expected))

    # NB: `mertric_merge_method = True` is a placeholder to allow
    # user-defined merge methods in the future and currently takes
    # metrics from the `child_interval` if they are not null or nan.
    def test_merge_metric_columns_of_intervals_with_list_metric_merge_true(self):
        main = Interval(pd.Series({"start": "01:00", "end": "02:00", "value": 10}), "start", "end",
                        metric_columns=["value"])
        child = Interval(pd.Series({"start": "01:00", "end": "02:00", "value": 20}), "start", "end",
                         metric_columns=["value"])
        expected = pd.Series({"start": "01:00", "end": "02:00", "value": 20})
        merged = merge_metric_columns_of_intervals(
            interval=main,
            other=child,

            metric_merge_method=True,
        )
        self.assertTrue(merged.equals(expected))

    def test_merge_metric_columns_of_intervals_with_string_metric_column(self):
        main = Interval(pd.Series({"start": "01:00", "end": "02:00", "value": 10}), "start", "end",
                        metric_columns=["value"])
        child = Interval(pd.Series({"start": "01:00", "end": "02:00", "value": 20}), "start", "end",
                         metric_columns=["value"])
        expected = pd.Series({"start": "01:00", "end": "02:00", "value": 20})
        merged = merge_metric_columns_of_intervals(
            interval=main,
            other=child,

            metric_merge_method=True,
        )
        self.assertTrue(merged.equals(expected))

    def test_merge_metric_columns_of_intervals_with_string_metric_columns(self):
        main = Interval(pd.Series({"start": "01:00", "end": "02:00", "value1": 10, "value2": 20}), "start", "end",
                        metric_columns=["value1", "value2"])
        child = Interval(pd.Series(
            {"start": "01:00", "end": "02:00", "value1": 20, "value2": 30}
        ), "start", "end", metric_columns=["value1", "value2"])
        expected = pd.Series(
            {"start": "01:00", "end": "02:00", "value1": 20, "value2": 30}
        )
        merged = merge_metric_columns_of_intervals(
            interval=main,
            other=child,

            metric_merge_method=True,
        )
        self.assertTrue(merged.equals(expected))

    def test_merge_metric_columns_of_intervals_return_new_copy(self):
        main = Interval(pd.Series({"start": "01:00", "end": "02:00", "value": 10}), "start", "end")
        child = Interval(pd.Series({"start": "01:00", "end": "02:00", "value": 20}), "start", "end")
        merged = merge_metric_columns_of_intervals(
            interval=main,
            other=child,
        )
        self.assertNotEqual(id(main), id(merged))

    # def test_merge_metric_columns_of_intervals_with_invalid_metric_columns_input(self):
    #     main = pd.Series({"start": "01:00", "end": "02:00", "value": 10})
    #     child = pd.Series({"start": "01:00", "end": "02:00", "value": 20})
    #     self.assertRaises(
    #         ValueError,
    #         IntervalContext,
    #         interval=main,
    #         other=child,
    #         interval_context=IntervalContext("start", "end", metric_columns=[10]),
    #         other_context=IntervalContext("start", "end", metric_columns=[10]),
    #     )

    def test_merge_metric_columns_of_intervals_handle_nan_in_child(self):
        main = Interval(pd.Series({"start": "01:00", "end": "02:00", "value": 10}), "start", "end")
        child = Interval(pd.Series({"start": "01:00", "end": "02:00", "value": float("nan")}), "start", "end")
        merged = merge_metric_columns_of_intervals(
            interval=main,
            other=child,

            metric_merge_method=True,
        )
        self.assertEqual(merged["value"], 10)

    def test_resolve_overlap_where_interval_other_have_equivalent_metric_cols(self):
        interval = Interval(pd.Series(
            {"start": "2022-01-02", "end": "2022-01-03", "metric_1": 5, "metric_2": 10}
        ), "start", "end")

        other = Interval(pd.Series(
            {"start": "2022-01-01", "end": "2022-01-04", "metric_1": 5, "metric_2": 10}
        ), "start", "end")
        resolver = OverlapResolver(interval, other)
        result = resolver.resolve_overlap()

        self.assertEqual(len(result), 1)

    def test_resolve_overlap_where_interval_is_contained_by_other(self):
        interval = Interval(pd.Series(
            {"start": "2022-01-02", "end": "2022-01-03", "metric_1": 5, "metric_2": 10}
        ), "start", "end", metric_columns=["metric_1", "metric_2"])

        other = Interval(pd.Series(
            {"start": "2022-01-01", "end": "2022-01-04", "metric_1": 6, "metric_2": 11}
        ), "start", "end", metric_columns=["metric_1", "metric_2"])
        resolver = OverlapResolver(interval, other)
        result = resolver.resolve_overlap()

        self.assertEqual(3, len(result))

    def test_resolve_overlap_where_shared_start_but_interval_ends_before_other(self):
        interval = Interval(pd.Series(
            {"start": "2022-01-01", "end": "2022-01-03", "metric_1": 5, "metric_2": 10}
        ), "start", "end", metric_columns=["metric_1", "metric_2"])

        other = Interval(pd.Series(
            {"start": "2022-01-01", "end": "2022-01-04", "metric_1": 6, "metric_2": 11}
        ), "start", "end", metric_columns=["metric_1", "metric_2"])
        resolver = OverlapResolver(interval, other)
        result = resolver.resolve_overlap()

        self.assertEqual(len(result), 2)

    def test_resolve_overlap_where_shared_start_but_interval_ends_after_other(self):
        interval = Interval(pd.Series(
            {"start": "2022-01-01", "end": "2022-01-04", "metric_1": 5, "metric_2": 10}
        ), "start", "end", metric_columns=["metric_1", "metric_2"])

        other = Interval(pd.Series(
            {"start": "2022-01-01", "end": "2022-01-03", "metric_1": 6, "metric_2": 11}
        ), "start", "end", metric_columns=["metric_1", "metric_2"])
        resolver = OverlapResolver(interval, other)
        result = resolver.resolve_overlap()

        self.assertEqual(len(result), 2)

    def test_resolve_overlap_where_shared_end_and_interval_starts_before_other(self):
        interval = Interval(pd.Series(
            {"start": "2022-01-01", "end": "2022-01-04", "metric_1": 5, "metric_2": 10}
        ), "start", "end", metric_columns=["metric_1", "metric_2"])

        other = Interval(pd.Series(
            {"start": "2022-01-02", "end": "2022-01-04", "metric_1": 6, "metric_2": 11}
        ), "start", "end", metric_columns=["metric_1", "metric_2"])
        resolver = OverlapResolver(interval, other)
        result = resolver.resolve_overlap()

        self.assertEqual(len(result), 2)

    def test_resolve_overlap_where_shared_end_and_interval_starts_after_other(self):
        interval = Interval(pd.Series(
            {"start": "2022-01-02", "end": "2022-01-04", "metric_1": 5, "metric_2": 10}
        ), "start", "end", metric_columns=["metric_1", "metric_2"])

        other = Interval(pd.Series(
            {"start": "2022-01-01", "end": "2022-01-04", "metric_1": 6, "metric_2": 11}
        ), "start", "end", metric_columns=["metric_1", "metric_2"])
        resolver = OverlapResolver(interval, other)
        result = resolver.resolve_overlap()

        self.assertEqual(len(result), 2)

    def test_resolve_overlap_shared_start_and_end(self):
        interval = Interval(pd.Series(
            {"start": "2022-01-01", "end": "2022-01-03", "metric_1": 5, "metric_2": 10}
        ), "start", "end")

        other = Interval(pd.Series(
            {"start": "2022-01-01", "end": "2022-01-03", "metric_1": 6, "metric_2": 11}
        ), "start", "end")
        resolver = OverlapResolver(interval, other)
        result = resolver.resolve_overlap()

        self.assertEqual(len(result), 1)

    def test_resolve_overlaps_where_interval_starts_first_partially_overlaps_other(
        self,
    ):
        interval = Interval(pd.Series(
            {"start": "2022-01-01", "end": "2022-01-03", "metric_1": 5, "metric_2": 10}
        ), "start", "end", metric_columns=["metric_1", "metric_2"])

        other = Interval(pd.Series(
            {"start": "2022-01-02", "end": "2022-01-04", "metric_1": 6, "metric_2": 11}
        ), "start", "end", metric_columns=["metric_1", "metric_2"])
        resolver = OverlapResolver(interval, other)
        result = resolver.resolve_overlap()

        self.assertEqual(len(result), 3)

    def test_resolve_overlaps_where_other_starts_first_partially_overlaps_interval(
        self,
    ):
        interval = Interval(pd.Series(
            {"start": "2022-01-01", "end": "2022-01-03", "metric_1": 6, "metric_2": 11}
        ), "start", "end", metric_columns=["metric_1", "metric_2"])
        other = Interval(pd.Series(
            {"start": "2022-01-02", "end": "2022-01-04", "metric_1": 5, "metric_2": 10}
        ), "start", "end", metric_columns=["metric_1", "metric_2"])
        resolver = OverlapResolver(interval, other)
        result = resolver.resolve_overlap()

        self.assertEqual(len(result), 3)

    def test_resolve_overlaps_where_different_series_id_col_names(self):
        interval = Interval(pd.Series(
            {
                "start": "2022-01-01",
                "end": "2022-01-03",
                "series_1": 1,
                "metric_1": 5,
                "metric_2": 10,
            }
        ), "start", "end", "series_1", ["metric_1", "metric_2"])

        other = Interval(pd.Series(
            {
                "start": "2022-01-02",
                "end": "2022-01-04",
                "wrong": 1,
                "metric_1": 6,
                "metric_2": 11,
            }
        ), "start", "end", "wrong", ["metric_1", "metric_2"])
        resolver = OverlapResolver(interval, other)
        self.assertRaises(
            ValueError,
            resolver.resolve_overlap

        )

    def test_resolve_overlaps_where_different_metric_col_names(self):
        interval = Interval(pd.Series(
            {
                "start": "2022-01-01",
                "end": "2022-01-03",
                "series_1": 1,
                "metric_1": 5,
                "metric_2": 10,
            }
        ), "start", "end", "series_1", ["metric_1", "metric_2"])

        other = Interval(pd.Series(
            {
                "start": "2022-01-02",
                "end": "2022-01-04",
                "series_1": 1,
                "wrong": 6,
                "metric_2": 11,
            }
        ), "start", "end", "series_1", ["wrong", "metric_2"])
        resolver = OverlapResolver(interval, other)
        self.assertRaises(
            ValueError,
            resolver.resolve_overlap

        )

    def test_resolve_overlaps_where_different_series_shapes(self):
        interval = Interval(pd.Series(
            {
                "start": "2022-01-01",
                "end": "2022-01-03",
                "series_1": 1,
                "metric_1": 5,
                "metric_2": 10,
            }
        ), "start", "end", ["series_1"], ["metric_1", "metric_2"])

        other = Interval(pd.Series(
            {"start": "2022-01-02", "end": "2022-01-04", "series_1": 1, "metric_2": 11}
        ), "start", "end", ["series_1"], ["metric_2"])
        resolver = OverlapResolver(interval, other)
        self.assertRaises(
            ValueError,
            resolver.resolve_overlap

        )

    def test_resolve_overlaps_where_no_overlaps(self):
        # Test 11: No overlap at all but still within the range of other series
        interval = Interval(pd.Series(
            {"start": "2022-01-01", "end": "2022-01-02", "metric_1": 5, "metric_2": 10}
        ), "start", "end")

        other = Interval(pd.Series(
            {"start": "2022-01-03", "end": "2022-01-04", "metric_1": 6, "metric_2": 11}
        ), "start", "end")
        resolver = OverlapResolver(interval, other)
        result = resolver.resolve_overlap()

        self.assertEqual(len(result), 2)


    def test_resolve_all_overlaps_basic(self):
        interval = Interval(pd.Series(
            {"start": "2023-01-01 00:00:00", "end": "2023-01-01 05:00:00", "value": 10}
        ), "start", "end", metric_columns="value")
        overlaps = pd.DataFrame(
            {
                "start": [
                    "2023-01-01 02:00:00",
                    "2023-01-01 03:00:00",
                    "2023-01-01 04:00:00",
                ],
                "end": [
                    "2023-01-01 04:00:00",
                    "2023-01-01 06:00:00",
                    "2023-01-01 07:00:00",
                ],
                "value": [5, 7, 8],
            }
        )

        result = resolve_all_overlaps(
            interval, overlaps
        )

        self.assertEqual(3, len(result))

    def test_resolve_all_overlaps_where_no_overlaps(self):
        interval = Interval(pd.Series(
            {"start": "2023-01-01 00:00:00", "end": "2023-01-01 05:00:00", "value": 10}
        ), "end", "start")
        overlaps = pd.DataFrame(
            {
                "start": [
                    "2023-01-01 06:00:00",
                    "2023-01-01 10:00:00",
                    "2023-01-01 12:00:00",
                ],
                "end": [
                    "2023-01-01 09:00:00",
                    "2023-01-01 11:00:00",
                    "2023-01-01 13:00:00",
                ],
                "value": [5, 7, 8],
            }
        )

        result = resolve_all_overlaps(
            interval, overlaps,
        )

        self.assertEqual(len(result), 4)

    def test_add_as_disjoint_where_basic_overlap(self):
        interval = Interval(pd.Series(
            {"start": "2023-01-01 00:00:00", "end": "2023-01-01 05:00:00", "value": 10}
        ), "start", "end", metric_columns="value")
        disjoint_set = pd.DataFrame(
            {
                "start": ["2023-01-01 03:00:00"],
                "end": ["2023-01-01 04:00:00"],
                "value": [5],
            }
        )

        result = add_as_disjoint(interval, disjoint_set)

        expected = pd.DataFrame(
            {
                "start": [
                    "2023-01-01 00:00:00",
                    "2023-01-01 03:00:00",
                    "2023-01-01 04:00:00",
                ],
                "end": [
                    "2023-01-01 03:00:00",
                    "2023-01-01 04:00:00",
                    "2023-01-01 05:00:00",
                ],
                "value": [10, 10, 10],
            }
        )

        self.assertTrue(np.array_equal(result.values, expected.values))

    def test_add_as_disjoint_where_no_overlap(self):
        interval = Interval(pd.Series(
            {"start": "2023-01-01 00:00:00", "end": "2023-01-01 05:00:00", "value": 10}
        ), "start", "end", metric_columns="value")
        disjoint_set = pd.DataFrame(
            {
                "start": ["2023-01-01 06:00:00"],
                "end": ["2023-01-01 07:00:00"],
                "value": [5],
            }
        )

        result = add_as_disjoint(interval, disjoint_set)

        expected = pd.concat([disjoint_set, pd.DataFrame([interval.data])])

        self.assertTrue(np.array_equal(result, expected))

    def test_add_as_disjoint_where_empty_disjoint_set(self):
        interval = Interval(pd.Series(
            {"start": "2023-01-01 00:00:00", "end": "2023-01-01 05:00:00", "value": 10}
        ), "start", "end", metric_columns="value")
        disjoint_set = pd.DataFrame()

        result = add_as_disjoint(interval, disjoint_set)

        expected = pd.DataFrame([interval.data])

        self.assertTrue(np.array_equal(result.values, expected.values))

    def test_add_as_disjoint_where_none_disjoint_set(self):
        interval = Interval(pd.Series(
            {"start": "2023-01-01 00:00:00", "end": "2023-01-01 05:00:00", "value": 10}
        ), "start", "end", metric_columns="value")

        result = add_as_disjoint(interval, None)

        expected = pd.DataFrame([interval.data])

        self.assertTrue(np.array_equal(result.values, expected.values))

    def test_add_as_disjoint_where_duplicate_interval(self):
        interval = Interval(pd.Series(
            {"start": "2023-01-01 00:00:00", "end": "2023-01-01 05:00:00", "value": 10}
        ), "start", "end")
        disjoint_set = pd.DataFrame(
            {
                "start": ["2023-01-01 00:00:00"],
                "end": ["2023-01-01 05:00:00"],
                "value": [10],
            }
        )

        result = add_as_disjoint(interval, disjoint_set)

        self.assertTrue(np.array_equal(result.values, disjoint_set.values))

    def test_add_as_disjoint_where_multiple_overlaps(self):
        interval = Interval(pd.Series(
            {"start": "2023-01-01 01:00:00", "end": "2023-01-01 05:00:00", "value": 10}
        ), "start", "end", metric_columns="value")
        disjoint_set = pd.DataFrame(
            {
                "start": [
                    "2023-01-01 00:00:00",
                    "2023-01-01 02:00:00",
                    "2023-01-01 03:00:00",
                ],
                "end": [
                    "2023-01-01 03:00:00",
                    "2023-01-01 04:00:00",
                    "2023-01-01 06:00:00",
                ],
                "value": [5, 10, 15],
            }
        )

        result = add_as_disjoint(interval, disjoint_set)

        expected = pd.DataFrame(
            {
                "start": [
                    "2023-01-01 00:00:00",
                    "2023-01-01 01:00:00",
                    "2023-01-01 03:00:00",
                    "2023-01-01 05:00:00",
                ],
                "end": [
                    "2023-01-01 01:00:00",
                    "2023-01-01 03:00:00",
                    "2023-01-01 05:00:00",
                    "2023-01-01 06:00:00",
                ],
                "value": [
                    5,
                    10,
                    10,
                    15,
                ],
            }
        )

        # NB: `pd.testing.assert_frame_equal` returns None or raises an Exception
        self.assertIsNone(
            pd.testing.assert_frame_equal(
                result.sort_values("start").reset_index(drop=True),
                expected.sort_values("start").reset_index(drop=True),
            )
        )

    def test_add_as_disjoint_where_all_records_overlap(self):
        interval = Interval(pd.Series(
            {"start": "2023-01-01 01:00:00", "end": "2023-01-01 05:00:00", "value": 10}
        ), "start", "end", metric_columns="value")
        disjoint_set = pd.DataFrame(
            {
                "start": ["2023-01-01 01:30:00", "2023-01-01 02:30:00"],
                "end": ["2023-01-01 02:30:00", "2023-01-01 03:30:00"],
                "value": [5, 10],
            }
        )

        result = add_as_disjoint(interval, disjoint_set)

        expected = pd.DataFrame(
            {
                "start": ["2023-01-01 01:00:00"],
                "end": ["2023-01-01 01:30:00"],
                "value": [10],
            }
        )

        # NB: `pd.testing.assert_frame_equal` returns None or raises an Exception
        self.assertIsNone(
            pd.testing.assert_frame_equal(
                result.sort_values("start").reset_index(drop=True),
                expected.sort_values("start").reset_index(drop=True),
            )
        )

    def test_make_disjoint_wrap_basic(self):
        start_ts = "start"
        end_ts = "end"
        series_ids = "id"
        metric_columns = "value"

        df = pd.DataFrame(
            {
                "start": ["2023-01-01 00:00", "2023-01-01 02:00"],
                "end": ["2023-01-01 03:00", "2023-01-01 04:00"],
                "id": [1, 2],
                "value": [10, 20],
            }
        )

        make_disjoint = make_disjoint_wrap(start_ts, end_ts, series_ids, metric_columns)
        result = make_disjoint(df)

        expected = pd.DataFrame(
            {
                "start": ["2023-01-01 00:00", "2023-01-01 02:00", "2023-01-01 03:00"],
                "end": ["2023-01-01 02:00", "2023-01-01 03:00", "2023-01-01 04:00"],
                "id": [1, 2, 2],
                "value": [10, 10, 20],
            }
        )

        # NB: `pd.testing.assert_frame_equal` returns None or raises an Exception
        self.assertIsNone(
            pd.testing.assert_frame_equal(
                result.sort_values(["id", "start"]).reset_index(drop=True),
                expected.sort_values(["id", "start"]).reset_index(drop=True),
            )
        )

    def test_make_disjoint_wrap_where_empty_dataframe(self):
        start_ts = "start"
        end_ts = "end"
        series_ids = "id"
        metric_columns = "value"

        df = pd.DataFrame(columns=[start_ts, end_ts, series_ids, metric_columns])

        make_disjoint = make_disjoint_wrap(start_ts, end_ts, series_ids, metric_columns)
        result = make_disjoint(df)

        expected = df  # Still an empty DataFrame

        # Check that result is empty
        self.assertTrue(result.empty, "The resulting DataFrame should be empty")
        self.assertListEqual(result.columns.tolist(), expected.columns.tolist())

    def test_make_disjoint_wrap_where_no_overlaps(self):
        start_ts = "start"
        end_ts = "end"
        series_ids = "id"
        metric_columns = "value"

        df = pd.DataFrame(
            {
                "start": ["2023-01-01 00:00", "2023-01-01 04:00"],
                "end": ["2023-01-01 02:00", "2023-01-01 06:00"],
                "id": [1, 2],
                "value": [10, 20],
            }
        )

        make_disjoint = make_disjoint_wrap(start_ts, end_ts, series_ids, metric_columns)
        result = make_disjoint(df)

        # No change in the result since all intervals are disjoint
        expected = df

        # NB: `pd.testing.assert_frame_equal` returns None or raises an Exception
        self.assertIsNone(pd.testing.assert_frame_equal(result, expected))

    def test_make_disjoint_duplicate_rows(self):
        start_ts = "start"
        end_ts = "end"
        series_ids = "id"
        metric_columns = "value"

        df = pd.DataFrame(
            {
                "start": ["2023-01-01 00:00", "2023-01-01 00:00"],
                "end": ["2023-01-01 02:00", "2023-01-01 02:00"],
                "id": [1, 1],
                "value": [10, 10],
            }
        )

        make_disjoint = make_disjoint_wrap(start_ts, end_ts, series_ids, metric_columns)
        result = make_disjoint(df)

        expected = pd.DataFrame(
            {
                "start": ["2023-01-01 00:00"],
                "end": ["2023-01-01 02:00"],
                "id": [1],
                "value": [10],
            }
        )

        # NB: `pd.testing.assert_frame_equal` returns None or raises an Exception
        self.assertIsNone(pd.testing.assert_frame_equal(result, expected))
