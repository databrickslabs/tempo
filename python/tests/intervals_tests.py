from tempo.intervals import *
from tests.tsdf_tests import SparkTest
from pyspark.sql.utils import AnalysisException
import pyspark.sql.functions as f


class IntervalsDFTests(SparkTest):
    union_tests_dict_input = [
        {
            "start_ts": "2020-08-01 00:00:09",
            "end_ts": "2020-08-01 00:00:14",
            "identifier_1": "v1",
            "series_1": 5,
            "series_2": None,
        },
        {
            "start_ts": "2020-08-01 00:00:09",
            "end_ts": "2020-08-01 00:00:11",
            "identifier_1": "v1",
            "series_1": None,
            "series_2": 0,
        },
        {
            "start_ts": "2020-08-01 00:00:09",
            "end_ts": "2020-08-01 00:00:12",
            "identifier_1": "v1",
            "series_1": None,
            "series_2": 4,
        },
        {
            "start_ts": "2020-08-01 00:00:09",
            "end_ts": "2020-08-01 00:00:14",
            "identifier_1": "v1",
            "series_1": 5,
            "series_2": None,
        },
        {
            "start_ts": "2020-08-01 00:00:09",
            "end_ts": "2020-08-01 00:00:11",
            "identifier_1": "v1",
            "series_1": None,
            "series_2": 0,
        },
        {
            "start_ts": "2020-08-01 00:00:09",
            "end_ts": "2020-08-01 00:00:12",
            "identifier_1": "v1",
            "series_1": None,
            "series_2": 4,
        },
    ]

    def test_init_identifier_series_str(self):
        df_input = self.get_data_as_sdf("input")

        self.assertRaises(
            ValueError,
            IntervalsDF,
            df_input,
            "start_ts",
            "end_ts",
            "identifier_1",
            "series_1",
        )

    def test_init_identifier_series_tuple(self):
        df_input = self.get_data_as_sdf("input")

        self.assertRaises(
            ValueError,
            IntervalsDF,
            df_input,
            "start_ts",
            "end_ts",
            ("identifier_1",),
            (
                "series_1",
                "series_2",
            ),
        )

    def test_init_identifier_series_list(self):
        df_input = self.get_data_as_sdf("input")

        idf = IntervalsDF(
            df_input,
            "start_ts",
            "end_ts",
            ["identifier_1"],
            ["series_1", "series_2"],
        )

        self.assertIsInstance(idf, IntervalsDF)
        self.assertIsInstance(idf.df, DataFrame)
        self.assertEqual(idf.start_ts, "start_ts")
        self.assertEqual(idf.end_ts, "end_ts")
        self.assertEqual(idf.identifiers, ["identifier_1"])
        self.assertEqual(idf.series_ids, ["series_1", "series_2"])

    def test_init_identifier_none(self):
        df_input = self.get_data_as_sdf("input")

        self.assertRaises(
            ValueError,
            IntervalsDF,
            df_input,
            "start_ts",
            "end_ts",
            None,
        )

    def test_init_identifier_truthiness(self):
        df_input = self.get_data_as_sdf("input")

        self.assertRaises(
            ValueError,
            IntervalsDF,
            df_input,
            "start_ts",
            "end_ts",
            [],
            None,
        )

    def test_init_series_none(self):
        df_input = self.get_data_as_sdf("input")

        idf = IntervalsDF(
            df_input,
            "start_ts",
            "end_ts",
            ["identifier_1"],
            None,
        )

        self.assertEqual(idf.series_ids, None)

    def test_init_series_truthiness(self):
        df_input = self.get_data_as_sdf("input")

        self.assertRaises(
            ValueError,
            IntervalsDF,
            df_input,
            "start_ts",
            "end_ts",
            ["identifier_1"],
            [],
        )

    def test_fromStackedSeries_identifier_str(self):
        df_input = self.get_data_as_sdf("input")

        self.assertRaises(
            ValueError,
            IntervalsDF.fromStackedSeries,
            df_input,
            "start_ts",
            "end_ts",
            "identifier_1",
            "series_name",
            "series_value",
        )

    def test_fromStackedSeries_identifier_tuple(self):
        df_input = self.get_data_as_sdf("input")

        self.assertRaises(
            ValueError,
            IntervalsDF.fromStackedSeries,
            df_input,
            "start_ts",
            "end_ts",
            ("identifier_1",),
            "series_name",
            "series_value",
        )

    def test_fromStackedSeries_identifier_list(self):
        df_input = self.get_data_as_sdf("input")
        idf_expected = self.get_data_as_idf("expected")

        df_input = df_input.withColumn(
            "start_ts", f.to_timestamp("start_ts")
        ).withColumn("end_ts", f.to_timestamp("end_ts"))

        idf = IntervalsDF.fromStackedSeries(
            df_input,
            "start_ts",
            "end_ts",
            [
                "identifier_1",
            ],
            "series_name",
            "series_value",
        )

        self.assertDataFramesEqual(idf.df, idf_expected.df)

    def test_disjoint_overlapping_intervals(self):
        idf_input = self.get_data_as_idf("input")
        idf_expected = self.get_data_as_idf("expected")

        idf_actual = idf_input.disjoint()

        self.assertDataFramesEqual(idf_expected.df, idf_actual.df)

    def test_disjoint_interval_no_overlap(self):
        ...

    def test_disjoint_intervals_same_start(self):
        ...

    def test_disjoint_intervals_same_end(self):
        ...

    def test_disjoint_active_interval(self):
        # end_ts is null since interval is current state
        ...

    def test_disjoint_multiple_identifiers(self):
        ...

    def test_disjoint_single_series(self):
        ...

    def test_disjoint_interval_is_subset_of_another(self):
        ...

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

        self.assertDataFramesEqual(actual_df, expected_df)

    def test_toDF_stack(self):
        idf_input = self.get_data_as_idf("input")
        expected_df = self.get_data_as_sdf("expected")

        expected_df = expected_df.withColumn(
            "start_ts", f.to_timestamp("start_ts")
        ).withColumn("end_ts", f.to_timestamp("end_ts"))

        actual_df = idf_input.toDF(stack=True)

        self.assertDataFramesEqual(actual_df, expected_df)