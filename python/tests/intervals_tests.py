from tempo.intervals import *
from tests.tsdf_tests import SparkTest
from pyspark.sql.utils import AnalysisException
import pyspark.sql.functions as Fn


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
        df_input = self.get_data_as_sdf("input")

        self.assertRaises(
            ValueError,
            IntervalsDF,
            df_input,
            "start_ts",
            "end_ts",
            "series_1",
        )

    def test_init_series_tuple(self):
        df_input = self.get_data_as_sdf("input")

        self.assertRaises(
            ValueError,
            IntervalsDF,
            df_input,
            "start_ts",
            "end_ts",
            ("series_1",),
        )

    def test_init_series_list(self):
        df_input = self.get_data_as_sdf("input")

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
        df_input = self.get_data_as_sdf("input")

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

    def test_fromStackedMetrics_series_str(self):
        df_input = self.get_data_as_sdf("input")

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
        df_input = self.get_data_as_sdf("input")

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
        df_input = self.get_data_as_sdf("input")
        idf_expected = self.get_data_as_idf("expected")

        df_input = df_input.withColumn(
            "start_ts", Fn.to_timestamp("start_ts")
        ).withColumn("end_ts", Fn.to_timestamp("end_ts"))

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

        self.assertDataFrameEquality(idf, idf_expected, from_idf=True)

    def test_fromStackedMetrics_metric_names(self):
        df_input = self.get_data_as_sdf("input")
        idf_expected = self.get_data_as_idf("expected")

        df_input = df_input.withColumn(
            "start_ts", Fn.to_timestamp("start_ts")
        ).withColumn("end_ts", Fn.to_timestamp("end_ts"))

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

        self.assertDataFrameEquality(idf, idf_expected, from_idf=True)

    def test_disjoint(self):
        idf_input = self.get_data_as_idf("input")
        idf_expected = self.get_data_as_idf("expected")

        idf_actual = idf_input.disjoint()

        self.assertDataFrameEquality(
            idf_expected, idf_actual, from_idf=True, ignore_row_order=True
        )

    def test_disjoint_contains_interval_already_disjoint(self):
        idf_input = self.get_data_as_idf("input")
        idf_expected = self.get_data_as_idf("expected")

        idf_actual = idf_input.disjoint()

        self.assertDataFrameEquality(
            idf_expected, idf_actual, from_idf=True, ignore_row_order=True
        )

    def test_disjoint_contains_intervals_equal(self):
        idf_input = self.get_data_as_idf("input")
        idf_expected = self.get_data_as_idf("expected")

        idf_actual = idf_input.disjoint()

        self.assertDataFrameEquality(
            idf_expected, idf_actual, from_idf=True, ignore_row_order=True
        )

    def test_disjoint_intervals_same_start(self):
        idf_input = self.get_data_as_idf("input")
        idf_expected = self.get_data_as_idf("expected")

        idf_actual = idf_input.disjoint()

        self.assertDataFrameEquality(
            idf_expected, idf_actual, from_idf=True, ignore_row_order=True
        )

    def test_disjoint_intervals_same_end(self):
        idf_input = self.get_data_as_idf("input")
        idf_expected = self.get_data_as_idf("expected")

        idf_actual = idf_input.disjoint()

        self.assertDataFrameEquality(
            idf_expected, idf_actual, from_idf=True, ignore_row_order=True
        )

    def test_disjoint_multiple_series(self):
        idf_input = self.get_data_as_idf("input")
        idf_expected = self.get_data_as_idf("expected")

        idf_actual = idf_input.disjoint()

        self.assertDataFrameEquality(
            idf_expected, idf_actual, from_idf=True, ignore_row_order=True
        )

    def test_disjoint_single_metric(self):
        idf_input = self.get_data_as_idf("input")
        idf_expected = self.get_data_as_idf("expected")

        idf_actual = idf_input.disjoint()

        self.assertDataFrameEquality(
            idf_expected, idf_actual, from_idf=True, ignore_row_order=True
        )

    def test_disjoint_interval_is_subset(self):
        idf_input = self.get_data_as_idf("input")
        idf_expected = self.get_data_as_idf("expected")

        idf_actual = idf_input.disjoint()

        self.assertDataFrameEquality(
            idf_expected, idf_actual, from_idf=True, ignore_row_order=True
        )

    def test_union_other_idf(self):
        idf_input_1 = self.get_data_as_idf("input")
        idf_input_2 = self.get_data_as_idf("input")

        count_idf_1 = idf_input_1.df.count()
        count_idf_2 = idf_input_2.df.count()

        union_idf = idf_input_1.union(idf_input_2)

        count_union = union_idf.df.count()

        self.assertEqual(count_idf_1 + count_idf_2, count_union)

    def test_union_other_df(self):
        idf_input = self.get_data_as_idf("input")
        df_input = self.get_data_as_sdf("input")

        self.assertRaises(TypeError, idf_input.union, df_input)

    def test_union_other_list_dicts(self):
        idf_input = self.get_data_as_idf("input")

        self.assertRaises(
            TypeError, idf_input.union, IntervalsDFTests.union_tests_dict_input
        )

    def test_unionByName_other_idf(self):
        idf_input_1 = self.get_data_as_idf("input")
        idf_input_2 = self.get_data_as_idf("input")

        count_idf_1 = idf_input_1.df.count()
        count_idf_2 = idf_input_2.df.count()

        union_idf = idf_input_1.unionByName(idf_input_2)

        count_union_by_name = union_idf.df.count()

        self.assertEqual(count_idf_1 + count_idf_2, count_union_by_name)

    def test_unionByName_other_df(self):
        idf_input = self.get_data_as_idf("input")
        df_input = self.get_data_as_sdf("input")

        self.assertRaises(TypeError, idf_input.unionByName, df_input)

    def test_unionByName_other_list_dicts(self):
        idf_input = self.get_data_as_idf("input")

        self.assertRaises(
            TypeError, idf_input.unionByName, IntervalsDFTests.union_tests_dict_input
        )

    def test_unionByName_extra_column(self):
        idf_extra_col = self.get_data_as_idf("input_extra_col")
        idf_input = self.get_data_as_idf("input")

        self.assertRaises(AnalysisException, idf_extra_col.unionByName, idf_input)

    def test_unionByName_other_extra_column(self):
        idf_input = self.get_data_as_idf("input")
        idf_extra_col = self.get_data_as_idf("input_extra_col")

        self.assertRaises(AnalysisException, idf_input.unionByName, idf_extra_col)

    def test_toDF(self):
        idf_input = self.get_data_as_idf("input")
        expected_df = self.get_data_as_sdf("input")

        actual_df = idf_input.toDF()

        self.assertDataFrameEquality(actual_df, expected_df)

    def test_toDF_stack(self):
        idf_input = self.get_data_as_idf("input")
        expected_df = self.get_data_as_sdf("expected")

        expected_df = expected_df.withColumn(
            "start_ts", Fn.to_timestamp("start_ts")
        ).withColumn("end_ts", Fn.to_timestamp("end_ts"))

        actual_df = idf_input.toDF(stack=True)

        self.assertDataFrameEquality(actual_df, expected_df)
