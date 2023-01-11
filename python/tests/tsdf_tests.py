from io import StringIO
import sys
import unittest
import os
from dateutil import parser as dt_parser
from unittest import mock
from pyspark.sql.column import Column
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.window import WindowSpec
import pyspark.sql.functions as Fn

from tempo.tsdf import TSDF
from tests.base import SparkTest


class TSDFBaseTests(SparkTest):
    def test_TSDF_init(self):
        tsdf_init = self.get_data_as_tsdf("init")

        self.assertIsInstance(tsdf_init.df, DataFrame)
        self.assertEqual(tsdf_init.ts_col, "event_ts")
        self.assertEqual(tsdf_init.series_ids, ["symbol"])

    def test_describe(self):
        """AS-OF Join without a time-partition test"""

        # Construct dataframes
        tsdf_init = self.get_data_as_tsdf("init")

        # generate description dataframe
        res = tsdf_init.describe()

        # joined dataframe should equal the expected dataframe
        # self.assertDataFrameEquality(res, dfExpected)
        assert res.count() == 7
        assert (
            res.filter(Fn.col("unique_time_series_count") != " ")
            .select(Fn.max(Fn.col("unique_time_series_count")))
            .collect()[0][0]
            == "1"
        )
        assert (
            res.filter(Fn.col("min_ts") != " ")
            .select(Fn.col("min_ts").cast("string"))
            .collect()[0][0]
            == "2020-08-01 00:00:10"
        )
        assert (
            res.filter(Fn.col("max_ts") != " ")
            .select(Fn.col("max_ts").cast("string"))
            .collect()[0][0]
            == "2020-09-01 00:19:12"
        )

    # TODO - will be moved to new test suite for asOfJoin
    # def test__getBytesFromPlan(self):
    #     init_tsdf = self.get_data_as_tsdf("init")
    #
    #     _bytes = init_tsdf._TSDF__getBytesFromPlan(init_tsdf.df, self.spark)
    #
    #     self.assertEqual(_bytes, 6.2)

    @staticmethod
    @mock.patch.dict(os.environ, {"TZ": "UTC"})
    def __timestamp_to_double(ts: str) -> float:
        return dt_parser.isoparse(ts).timestamp()

    @staticmethod
    def __tsdf_with_double_tscol(tsdf: TSDF) -> TSDF:
        with_double_tscol_df = tsdf.df.withColumn(
            tsdf.ts_col, Fn.col(tsdf.ts_col).cast("double")
        )
        return TSDF(
            with_double_tscol_df, ts_col=tsdf.ts_col, series_ids=tsdf.series_ids
        )

    # TODO - replace this with test code for TSDF.fromTimestampString with nano-second precision
    # def test__add_double_ts(self):
    #     init_tsdf = self.get_data_as_tsdf("init")
    #     df = init_tsdf._TSDF__add_double_ts()
    #
    #     schema_string = df.schema.simpleString()
    #
    #     self.assertIn("double_ts:double", schema_string)

    # TODO - replace with tests for TSDF.fromTimestampString
    # def test__validate_ts_string_valid(self):
    #     valid_timestamp_string = "2020-09-01 00:02:10"
    #
    #     self.assertIsNone(TSDF._TSDF__validate_ts_string(valid_timestamp_string))
    #
    # def test__validate_ts_string_alt_format_valid(self):
    #     valid_timestamp_string = "2020-09-01T00:02:10"
    #
    #     self.assertIsNone(TSDF._TSDF__validate_ts_string(valid_timestamp_string))
    #
    # def test__validate_ts_string_with_microseconds_valid(self):
    #     valid_timestamp_string = "2020-09-01 00:02:10.00000000"
    #
    #     self.assertIsNone(TSDF._TSDF__validate_ts_string(valid_timestamp_string))
    #
    # def test__validate_ts_string_alt_format_with_microseconds_valid(self):
    #     valid_timestamp_string = "2020-09-01T00:02:10.00000000"
    #
    #     self.assertIsNone(TSDF._TSDF__validate_ts_string(valid_timestamp_string))
    #
    # def test__validate_ts_string_invalid(self):
    #     invalid_timestamp_string = "this will not work"
    #
    #     self.assertRaises(
    #         ValueError, TSDF._TSDF__validate_ts_string, invalid_timestamp_string
    #     )

    # TODO - replace with tests of TSSchema validation
    # def test__validated_column_not_string(self):
    #     init_df = self.get_data_as_tsdf("init").df
    #
    #     self.assertRaises(TypeError, TSDF._TSDF__validated_column, init_df, 0)
    #
    # def test__validated_column_not_found(self):
    #     init_df = self.get_data_as_tsdf("init").df
    #
    #     self.assertRaises(
    #         ValueError,
    #         TSDF._TSDF__validated_column,
    #         init_df,
    #         "does not exist",
    #     )
    #
    # def test__validated_column(self):
    #     init_df = self.get_data_as_tsdf("init").df
    #
    #     self.assertEqual(
    #         TSDF._TSDF__validated_column(init_df, "symbol"),
    #         "symbol",
    #     )
    #
    # def test__validated_columns_string(self):
    #     init_tsdf = self.get_data_as_tsdf("init")
    #
    #     self.assertEqual(
    #         init_tsdf._TSDF__validated_columns(init_tsdf.df, "symbol"),
    #         ["symbol"],
    #     )
    #
    # def test__validated_columns_none(self):
    #     init_tsdf = self.get_data_as_tsdf("init")
    #
    #     self.assertEqual(
    #         init_tsdf._TSDF__validated_columns(init_tsdf.df, None),
    #         [],
    #     )
    #
    # def test__validated_columns_tuple(self):
    #     init_tsdf = self.get_data_as_tsdf("init")
    #
    #     self.assertRaises(
    #         TypeError,
    #         init_tsdf._TSDF__validated_columns,
    #         init_tsdf.df,
    #         ("symbol",),
    #     )
    #
    # def test__validated_columns_list_multiple_elems(self):
    #     init_tsdf = self.get_data_as_tsdf("init")
    #
    #     self.assertEqual(
    #         init_tsdf._TSDF__validated_columns(
    #             init_tsdf.df,
    #             ["symbol", "event_ts", "trade_pr"],
    #         ),
    #         ["symbol", "event_ts", "trade_pr"],
    #     )

    # TODO - replace with test code for refactored asOfJoin helpers
    # def test__checkPartitionCols(self):
    #     init_tsdf = self.get_data_as_tsdf("init")
    #     right_tsdf = self.get_data_as_tsdf("right_tsdf")
    #
    #     self.assertRaises(ValueError, init_tsdf._TSDF__checkPartitionCols, right_tsdf)
    #
    # def test__validateTsColMatch(self):
    #     init_tsdf = self.get_data_as_tsdf("init")
    #     right_tsdf = self.get_data_as_tsdf("right_tsdf")
    #
    #     self.assertRaises(ValueError, init_tsdf._TSDF__validateTsColMatch, right_tsdf)
    #
    # def test__addPrefixToColumns_non_empty_string(self):
    #     init_tsdf = self.get_data_as_tsdf("init")
    #
    #     df = init_tsdf._TSDF__addPrefixToColumns(["event_ts"], "prefix").df
    #
    #     schema_string = df.schema.simpleString()
    #
    #     self.assertIn("prefix_event_ts", schema_string)
    #
    # def test__addPrefixToColumns_empty_string(self):
    #     init_tsdf = self.get_data_as_tsdf("init")
    #
    #     df = init_tsdf._TSDF__addPrefixToColumns(["event_ts"], "").df
    #
    #     schema_string = df.schema.simpleString()
    #
    #     # comma included (,event_ts) to ensure we don't match if there is a prefix added
    #     self.assertIn(",event_ts", schema_string)
    #
    # def test__addColumnsFromOtherDF(self):
    #     init_tsdf = self.get_data_as_tsdf("init")
    #
    #     df = init_tsdf._TSDF__addColumnsFromOtherDF(["another_col"]).df
    #
    #     schema_string = df.schema.simpleString()
    #
    #     self.assertIn("another_col", schema_string)
    #
    # def test__combineTSDF(self):
    #     init1_tsdf = self.get_data_as_tsdf("init")
    #     init2_tsdf = self.get_data_as_tsdf("init")
    #
    #     union_tsdf = init1_tsdf._TSDF__combineTSDF(init2_tsdf, "combined_ts_col")
    #     df = union_tsdf.df
    #
    #     schema_string = df.schema.simpleString()
    #
    #     self.assertEqual(init1_tsdf.df.count() + init2_tsdf.df.count(), df.count())
    #     self.assertIn("combined_ts_col", schema_string)
    #
    # def test__getLastRightRow(self):
    #     # TODO: several errors and hard-coded columns that throw AnalysisException
    #     pass
    #
    # def test__getTimePartitions(self):
    #     init_tsdf = self.get_data_as_tsdf("init")
    #     expected_tsdf = self.get_data_as_tsdf("expected")
    #
    #     actual_tsdf = init_tsdf._TSDF__getTimePartitions(10)
    #
    #     self.assertDataFrameEquality(
    #         actual_tsdf,
    #         expected_tsdf,
    #         from_tsdf=True,
    #     )
    #
    # def test__getTimePartitions_with_fraction(self):
    #     init_tsdf = self.get_data_as_tsdf("init")
    #     expected_tsdf = self.get_data_as_tsdf("expected")
    #
    #     actual_tsdf = init_tsdf._TSDF__getTimePartitions(10, 0.25)
    #
    #     self.assertDataFrameEquality(
    #         actual_tsdf,
    #         expected_tsdf,
    #         from_tsdf=True,
    #     )

    def test_select_empty(self):
        # TODO: Can we narrow down to types of Exception?
        init_tsdf = self.get_data_as_tsdf("init")

        self.assertRaises(Exception, init_tsdf.select)

    def test_select_only_required_cols(self):
        init_tsdf = self.get_data_as_tsdf("init")

        tsdf = init_tsdf.select("event_ts", "symbol")

        self.assertEqual(tsdf.df.columns, ["event_ts", "symbol"])

    def test_select_all_cols(self):
        init_tsdf = self.get_data_as_tsdf("init")

        tsdf = init_tsdf.select("event_ts", "symbol", "trade_pr")

        self.assertEqual(tsdf.df.columns, ["event_ts", "symbol", "trade_pr"])

    def test_show(self):
        init_tsdf = self.get_data_as_tsdf("init")

        captured_output = StringIO()
        sys.stdout = captured_output
        init_tsdf.show()
        self.assertEqual(
            captured_output.getvalue(),
            (
                "+------+-------------------+--------+\n"
                "|symbol|           event_ts|trade_pr|\n"
                "+------+-------------------+--------+\n"
                "|    S1|2020-08-01 00:00:10|  349.21|\n"
                "|    S1|2020-08-01 00:01:12|  351.32|\n"
                "|    S1|2020-09-01 00:02:10|   361.1|\n"
                "|    S1|2020-09-01 00:19:12|   362.1|\n"
                "|    S2|2020-08-01 00:01:10|  743.01|\n"
                "|    S2|2020-08-01 00:01:24|  751.92|\n"
                "|    S2|2020-09-01 00:02:10|   761.1|\n"
                "|    S2|2020-09-01 00:20:42|  762.33|\n"
                "+------+-------------------+--------+\n"
                "\n"
            ),
        )

    def test_show_n_5(self):
        init_tsdf = self.get_data_as_tsdf("init")

        captured_output = StringIO()
        sys.stdout = captured_output
        init_tsdf.show(5)
        self.assertEqual(
            captured_output.getvalue(),
            (
                "+------+-------------------+--------+\n"
                "|symbol|           event_ts|trade_pr|\n"
                "+------+-------------------+--------+\n"
                "|    S1|2020-08-01 00:00:10|  349.21|\n"
                "|    S1|2020-08-01 00:01:12|  351.32|\n"
                "|    S1|2020-09-01 00:02:10|   361.1|\n"
                "|    S1|2020-09-01 00:19:12|   362.1|\n"
                "|    S2|2020-08-01 00:01:10|  743.01|\n"
                "+------+-------------------+--------+\n"
                "only showing top 5 rows\n"
                "\n"
            ),
        )

    def test_show_k_gt_n(self):
        init_tsdf = self.get_data_as_tsdf("init")

        captured_output = StringIO()
        sys.stdout = captured_output
        self.assertRaises(ValueError, init_tsdf.show, 5, 10)

    def test_show_truncate_false(self):
        init_tsdf = self.get_data_as_tsdf("init")

        captured_output = StringIO()
        sys.stdout = captured_output
        init_tsdf.show(truncate=False)
        self.assertEqual(
            captured_output.getvalue(),
            (
                "+------+-------------------+--------+\n"
                "|symbol|event_ts           |trade_pr|\n"
                "+------+-------------------+--------+\n"
                "|S1    |2020-08-01 00:00:10|349.21  |\n"
                "|S1    |2020-08-01 00:01:12|351.32  |\n"
                "|S1    |2020-09-01 00:02:10|361.1   |\n"
                "|S1    |2020-09-01 00:19:12|362.1   |\n"
                "|S2    |2020-08-01 00:01:10|743.01  |\n"
                "|S2    |2020-08-01 00:01:24|751.92  |\n"
                "|S2    |2020-09-01 00:02:10|761.1   |\n"
                "|S2    |2020-09-01 00:20:42|762.33  |\n"
                "+------+-------------------+--------+\n"
                "\n"
            ),
        )

    def test_show_vertical_true(self):
        init_tsdf = self.get_data_as_tsdf("init")

        captured_output = StringIO()
        sys.stdout = captured_output
        init_tsdf.show(vertical=True)
        self.assertEqual(
            captured_output.getvalue(),
            (
                "-RECORD 0-----------------------\n"
                " symbol   | S1                  \n"
                " event_ts | 2020-08-01 00:00:10 \n"
                " trade_pr | 349.21              \n"
                "-RECORD 1-----------------------\n"
                " symbol   | S1                  \n"
                " event_ts | 2020-08-01 00:01:12 \n"
                " trade_pr | 351.32              \n"
                "-RECORD 2-----------------------\n"
                " symbol   | S1                  \n"
                " event_ts | 2020-09-01 00:02:10 \n"
                " trade_pr | 361.1               \n"
                "-RECORD 3-----------------------\n"
                " symbol   | S1                  \n"
                " event_ts | 2020-09-01 00:19:12 \n"
                " trade_pr | 362.1               \n"
                "-RECORD 4-----------------------\n"
                " symbol   | S2                  \n"
                " event_ts | 2020-08-01 00:01:10 \n"
                " trade_pr | 743.01              \n"
                "-RECORD 5-----------------------\n"
                " symbol   | S2                  \n"
                " event_ts | 2020-08-01 00:01:24 \n"
                " trade_pr | 751.92              \n"
                "-RECORD 6-----------------------\n"
                " symbol   | S2                  \n"
                " event_ts | 2020-09-01 00:02:10 \n"
                " trade_pr | 761.1               \n"
                "-RECORD 7-----------------------\n"
                " symbol   | S2                  \n"
                " event_ts | 2020-09-01 00:20:42 \n"
                " trade_pr | 762.33              \n"
                "\n"
            ),
        )

    def test_show_vertical_true_n_5(self):
        init_tsdf = self.get_data_as_tsdf("init")

        captured_output = StringIO()
        sys.stdout = captured_output
        init_tsdf.show(5, vertical=True)
        self.assertEqual(
            captured_output.getvalue(),
            (
                "-RECORD 0-----------------------\n"
                " symbol   | S1                  \n"
                " event_ts | 2020-08-01 00:00:10 \n"
                " trade_pr | 349.21              \n"
                "-RECORD 1-----------------------\n"
                " symbol   | S1                  \n"
                " event_ts | 2020-08-01 00:01:12 \n"
                " trade_pr | 351.32              \n"
                "-RECORD 2-----------------------\n"
                " symbol   | S1                  \n"
                " event_ts | 2020-09-01 00:02:10 \n"
                " trade_pr | 361.1               \n"
                "-RECORD 3-----------------------\n"
                " symbol   | S1                  \n"
                " event_ts | 2020-09-01 00:19:12 \n"
                " trade_pr | 362.1               \n"
                "-RECORD 4-----------------------\n"
                " symbol   | S2                  \n"
                " event_ts | 2020-08-01 00:01:10 \n"
                " trade_pr | 743.01              \n"
                "only showing top 5 rows\n"
                "\n"
            ),
        )

    def test_show_truncate_false_vertical_true(self):
        init_tsdf = self.get_data_as_tsdf("init")

        captured_output = StringIO()
        sys.stdout = captured_output
        init_tsdf.show(truncate=False, vertical=True)
        self.assertEqual(
            captured_output.getvalue(),
            (
                "-RECORD 0-----------------------\n"
                " symbol   | S1                  \n"
                " event_ts | 2020-08-01 00:00:10 \n"
                " trade_pr | 349.21              \n"
                "-RECORD 1-----------------------\n"
                " symbol   | S1                  \n"
                " event_ts | 2020-08-01 00:01:12 \n"
                " trade_pr | 351.32              \n"
                "-RECORD 2-----------------------\n"
                " symbol   | S1                  \n"
                " event_ts | 2020-09-01 00:02:10 \n"
                " trade_pr | 361.1               \n"
                "-RECORD 3-----------------------\n"
                " symbol   | S1                  \n"
                " event_ts | 2020-09-01 00:19:12 \n"
                " trade_pr | 362.1               \n"
                "-RECORD 4-----------------------\n"
                " symbol   | S2                  \n"
                " event_ts | 2020-08-01 00:01:10 \n"
                " trade_pr | 743.01              \n"
                "-RECORD 5-----------------------\n"
                " symbol   | S2                  \n"
                " event_ts | 2020-08-01 00:01:24 \n"
                " trade_pr | 751.92              \n"
                "-RECORD 6-----------------------\n"
                " symbol   | S2                  \n"
                " event_ts | 2020-09-01 00:02:10 \n"
                " trade_pr | 761.1               \n"
                "-RECORD 7-----------------------\n"
                " symbol   | S2                  \n"
                " event_ts | 2020-09-01 00:20:42 \n"
                " trade_pr | 762.33              \n"
                "\n"
            ),
        )

    def test_at_string_timestamp(self):
        """
        Test of time-slicing at(..) function using a string timestamp
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        target_ts = "2020-09-01 00:02:10"
        at_tsdf = init_tsdf.at(target_ts)

        self.assertDataFrameEquality(at_tsdf, expected_tsdf, from_tsdf=True)

    def test_at_numeric_timestamp(self):
        """
        Test of time-slicint at(..) function using a numeric timestamp
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        # test with numeric ts_col
        init_dbl_tsdf = self.__tsdf_with_double_tscol(init_tsdf)
        expected_dbl_tsdf = self.__tsdf_with_double_tscol(expected_tsdf)

        target_ts = "2020-09-01 00:02:10"
        target_dbl = self.__timestamp_to_double(target_ts)
        at_dbl_tsdf = init_dbl_tsdf.at(target_dbl)

        self.assertDataFrameEquality(at_dbl_tsdf, expected_dbl_tsdf, from_tsdf=True)

    def test_before_string_timestamp(self):
        """
        Test of time-slicing before(..) function
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        target_ts = "2020-09-01 00:02:10"
        before_tsdf = init_tsdf.before(target_ts)

        self.assertDataFrameEquality(before_tsdf, expected_tsdf, from_tsdf=True)

    def test_before_numeric_timestamp(self):
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        # test with numeric ts_col
        init_dbl_tsdf = self.__tsdf_with_double_tscol(init_tsdf)
        expected_dbl_tsdf = self.__tsdf_with_double_tscol(expected_tsdf)

        target_ts = "2020-09-01 00:02:10"
        target_dbl = self.__timestamp_to_double(target_ts)
        before_dbl_tsdf = init_dbl_tsdf.before(target_dbl)

        self.assertDataFrameEquality(before_dbl_tsdf, expected_dbl_tsdf, from_tsdf=True)

    def test_atOrBefore_string_timestamp(self):
        """
        Test of time-slicing atOrBefore(..) function
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        target_ts = "2020-09-01 00:02:10"
        before_tsdf = init_tsdf.atOrBefore(target_ts)

        self.assertDataFrameEquality(before_tsdf, expected_tsdf, from_tsdf=True)

    def test_atOrBefore_numeric_timestamp(self):
        """
        Test of time-slicing atOrBefore(..) function
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        target_ts = "2020-09-01 00:02:10"

        # test with numeric ts_col
        init_dbl_tsdf = self.__tsdf_with_double_tscol(init_tsdf)
        expected_dbl_tsdf = self.__tsdf_with_double_tscol(expected_tsdf)

        target_dbl = self.__timestamp_to_double(target_ts)
        before_dbl_tsdf = init_dbl_tsdf.atOrBefore(target_dbl)

        self.assertDataFrameEquality(before_dbl_tsdf, expected_dbl_tsdf, from_tsdf=True)

    def test_after_string_timestamp(self):
        """
        Test of time-slicing after(..) function
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        target_ts = "2020-09-01 00:02:10"
        after_tsdf = init_tsdf.after(target_ts)

        self.assertDataFrameEquality(after_tsdf, expected_tsdf, from_tsdf=True)

    def test_after_numeric_timestamp(self):
        """
        Test of time-slicing after(..) function
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        target_ts = "2020-09-01 00:02:10"

        # test with numeric ts_col
        init_dbl_tsdf = self.__tsdf_with_double_tscol(init_tsdf)
        expected_dbl_tsdf = self.__tsdf_with_double_tscol(expected_tsdf)

        target_dbl = self.__timestamp_to_double(target_ts)
        after_dbl_tsdf = init_dbl_tsdf.after(target_dbl)

        self.assertDataFrameEquality(after_dbl_tsdf, expected_dbl_tsdf, from_tsdf=True)

    def test_atOrAfter_string_timestamp(self):
        """
        Test of time-slicing atOrAfter(..) function
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        target_ts = "2020-09-01 00:02:10"
        after_tsdf = init_tsdf.atOrAfter(target_ts)

        self.assertDataFrameEquality(after_tsdf, expected_tsdf, from_tsdf=True)

    def test_atOrAfter_numeric_timestamp(self):
        """
        Test of time-slicing atOrAfter(..) function
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        target_ts = "2020-09-01 00:02:10"

        # test with numeric ts_col
        init_dbl_tsdf = self.__tsdf_with_double_tscol(init_tsdf)
        expected_dbl_tsdf = self.__tsdf_with_double_tscol(expected_tsdf)

        target_dbl = self.__timestamp_to_double(target_ts)
        after_dbl_tsdf = init_dbl_tsdf.atOrAfter(target_dbl)

        self.assertDataFrameEquality(after_dbl_tsdf, expected_dbl_tsdf, from_tsdf=True)

    def test_between_string_timestamp(self):
        """
        Test of time-slicing between(..) function
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        ts1 = "2020-08-01 00:01:10"
        ts2 = "2020-09-01 00:18:00"
        between_tsdf = init_tsdf.between(ts1, ts2)

        self.assertDataFrameEquality(between_tsdf, expected_tsdf, from_tsdf=True)

    def test_between_numeric_timestamp(self):
        """
        Test of time-slicing between(..) function
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        ts1 = "2020-08-01 00:01:10"
        ts2 = "2020-09-01 00:18:00"

        # test with numeric ts_col
        init_dbl_tsdf = self.__tsdf_with_double_tscol(init_tsdf)
        expected_dbl_tsdf = self.__tsdf_with_double_tscol(expected_tsdf)

        ts1_dbl = self.__timestamp_to_double(ts1)
        ts2_dbl = self.__timestamp_to_double(ts2)
        between_dbl_tsdf = init_dbl_tsdf.between(ts1_dbl, ts2_dbl)

        self.assertDataFrameEquality(
            between_dbl_tsdf, expected_dbl_tsdf, from_tsdf=True
        )

    def test_between_exclusive_string_timestamp(self):
        """
        Test of time-slicing between(..) function
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        ts1 = "2020-08-01 00:01:10"
        ts2 = "2020-09-01 00:18:00"
        between_tsdf = init_tsdf.between(ts1, ts2, inclusive=False)

        self.assertDataFrameEquality(between_tsdf, expected_tsdf, from_tsdf=True)

    def test_between_exclusive_numeric_timestamp(self):
        """
        Test of time-slicing between(..) function
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        ts1 = "2020-08-01 00:01:10"
        ts2 = "2020-09-01 00:18:00"

        # test with numeric ts_col
        init_dbl_tsdf = self.__tsdf_with_double_tscol(init_tsdf)
        expected_dbl_tsdf = self.__tsdf_with_double_tscol(expected_tsdf)

        ts1_dbl = self.__timestamp_to_double(ts1)
        ts2_dbl = self.__timestamp_to_double(ts2)
        between_dbl_tsdf = init_dbl_tsdf.between(ts1_dbl, ts2_dbl, inclusive=False)

        self.assertDataFrameEquality(
            between_dbl_tsdf, expected_dbl_tsdf, from_tsdf=True
        )

    def test_earliest_string_timestamp(self):
        """
        Test of time-slicing earliest(..) function
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        earliest_tsdf = init_tsdf.earliest(n=3)

        self.assertDataFrameEquality(earliest_tsdf, expected_tsdf, from_tsdf=True)

    def test_earliest_numeric_timestamp(self):
        """
        Test of time-slicing earliest(..) function
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        # test with numeric ts_col
        init_dbl_tsdf = self.__tsdf_with_double_tscol(init_tsdf)
        expected_dbl_tsdf = self.__tsdf_with_double_tscol(expected_tsdf)

        earliest_dbl_tsdf = init_dbl_tsdf.earliest(n=3)

        self.assertDataFrameEquality(
            earliest_dbl_tsdf, expected_dbl_tsdf, from_tsdf=True
        )

    def test_latest_string_timestamp(self):
        """
        Test of time-slicing latest(..) function
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        latest_tsdf = init_tsdf.latest(n=3)

        self.assertDataFrameEquality(
            latest_tsdf, expected_tsdf, ignore_row_order=True, from_tsdf=True
        )

    def test_latest_numeric_timestamp(self):
        """
        Test of time-slicing latest(..) function
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        # test with numeric ts_col
        init_dbl_tsdf = self.__tsdf_with_double_tscol(init_tsdf)
        expected_dbl_tsdf = self.__tsdf_with_double_tscol(expected_tsdf)

        latest_dbl_tsdf = init_dbl_tsdf.latest(n=3)

        self.assertDataFrameEquality(
            latest_dbl_tsdf, expected_dbl_tsdf, ignore_row_order=True, from_tsdf=True
        )

    def test_priorTo_string_timestamp(self):
        """
        Test of time-slicing priorTo(..) function
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        target_ts = "2020-09-01 00:02:00"
        prior_tsdf = init_tsdf.priorTo(target_ts)

        self.assertDataFrameEquality(prior_tsdf, expected_tsdf, from_tsdf=True)

    def test_priorTo_numeric_timestamp(self):
        """
        Test of time-slicing priorTo(..) function
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        target_ts = "2020-09-01 00:02:00"

        # test with numeric ts_col
        init_dbl_tsdf = self.__tsdf_with_double_tscol(init_tsdf)
        expected_dbl_tsdf = self.__tsdf_with_double_tscol(expected_tsdf)

        target_dbl = self.__timestamp_to_double(target_ts)
        prior_dbl_tsdf = init_dbl_tsdf.priorTo(target_dbl)

        self.assertDataFrameEquality(prior_dbl_tsdf, expected_dbl_tsdf, from_tsdf=True)

    def test_subsequentTo_string_timestamp(self):
        """
        Test of time-slicing subsequentTo(..) function
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        target_ts = "2020-09-01 00:02:00"
        subsequent_tsdf = init_tsdf.subsequentTo(target_ts)

        self.assertDataFrameEquality(subsequent_tsdf, expected_tsdf, from_tsdf=True)

    def test_subsequentTo_numeric_timestamp(self):
        """
        Test of time-slicing subsequentTo(..) function
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        target_ts = "2020-09-01 00:02:00"

        # test with numeric ts_col
        init_dbl_tsdf = self.__tsdf_with_double_tscol(init_tsdf)
        expected_dbl_tsdf = self.__tsdf_with_double_tscol(expected_tsdf)

        target_dbl = self.__timestamp_to_double(target_ts)
        subsequent_dbl_tsdf = init_dbl_tsdf.subsequentTo(target_dbl)

        self.assertDataFrameEquality(
            subsequent_dbl_tsdf, expected_dbl_tsdf, from_tsdf=True
        )

    def test__rowsBetweenWindow(self):
        init_tsdf = self.get_data_as_tsdf("init")

        self.assertIsInstance(init_tsdf._TSDF__rowsBetweenWindow(1, 1), WindowSpec)

    # def test_withPartitionCols(self):
    #     init_tsdf = self.get_data_as_tsdf("init")
    #
    #     actual_tsdf = init_tsdf.withPartitionCols(["symbol"])
    #
    #     self.assertEqual(init_tsdf.partitionCols, [])
    #     self.assertEqual(actual_tsdf.partitionCols, ["symbol"])


class FourierTransformTest(SparkTest):
    def test_fourier_transform(self):
        """Test of fourier transform functionality in TSDF objects"""

        # construct dataframes
        tsdf_init = self.get_data_as_tsdf("init")
        dfExpected = self.get_data_as_sdf("expected")

        # convert to TSDF
        result_tsdf = tsdf_init.fourier_transform(1, "val")

        # should be equal to the expected dataframe
        self.assertDataFrameEquality(result_tsdf.df, dfExpected)


class RangeStatsTest(SparkTest):
    def test_range_stats(self):
        """Test of range stats for 20 minute rolling window"""

        # construct dataframes
        tsdf_init = self.get_data_as_tsdf("init")
        dfExpected = self.get_data_as_sdf("expected")

        # convert to TSDF

        # using lookback of 20 minutes
        featured_df = tsdf_init.withRangeStats(rangeBackWindowSecs=1200).df

        # cast to decimal with precision in cents for simplicity
        featured_df = featured_df.select(
            Fn.col("symbol"),
            Fn.col("event_ts"),
            Fn.col("mean_trade_pr").cast("decimal(5, 2)"),
            Fn.col("count_trade_pr"),
            Fn.col("min_trade_pr").cast("decimal(5,2)"),
            Fn.col("max_trade_pr").cast("decimal(5,2)"),
            Fn.col("sum_trade_pr").cast("decimal(5,2)"),
            Fn.col("stddev_trade_pr").cast("decimal(5,2)"),
            Fn.col("zscore_trade_pr").cast("decimal(5,2)"),
        )

        # cast to decimal with precision in cents for simplicity
        dfExpected = dfExpected.select(
            Fn.col("symbol"),
            Fn.col("event_ts"),
            Fn.col("mean_trade_pr").cast("decimal(5, 2)"),
            Fn.col("count_trade_pr"),
            Fn.col("min_trade_pr").cast("decimal(5,2)"),
            Fn.col("max_trade_pr").cast("decimal(5,2)"),
            Fn.col("sum_trade_pr").cast("decimal(5,2)"),
            Fn.col("stddev_trade_pr").cast("decimal(5,2)"),
            Fn.col("zscore_trade_pr").cast("decimal(5,2)"),
        )

        # should be equal to the expected dataframe
        self.assertDataFrameEquality(featured_df, dfExpected)

    def test_group_stats(self):
        """Test of range stats for 20 minute rolling window"""

        # construct dataframes
        tsdf_init = self.get_data_as_tsdf("init")
        dfExpected = self.get_data_as_sdf("expected")

        # using lookback of 20 minutes
        featured_df = tsdf_init.withGroupedStats(freq="1 min").df

        # cast to decimal with precision in cents for simplicity
        featured_df = featured_df.select(
            Fn.col("symbol"),
            Fn.col("event_ts"),
            Fn.col("mean_trade_pr").cast("decimal(5, 2)"),
            Fn.col("count_trade_pr"),
            Fn.col("min_trade_pr").cast("decimal(5,2)"),
            Fn.col("max_trade_pr").cast("decimal(5,2)"),
            Fn.col("sum_trade_pr").cast("decimal(5,2)"),
            Fn.col("stddev_trade_pr").cast("decimal(5,2)"),
        )

        # cast to decimal with precision in cents for simplicity
        dfExpected = dfExpected.select(
            Fn.col("symbol"),
            Fn.col("event_ts"),
            Fn.col("mean_trade_pr").cast("decimal(5, 2)"),
            Fn.col("count_trade_pr"),
            Fn.col("min_trade_pr").cast("decimal(5,2)"),
            Fn.col("max_trade_pr").cast("decimal(5,2)"),
            Fn.col("sum_trade_pr").cast("decimal(5,2)"),
            Fn.col("stddev_trade_pr").cast("decimal(5,2)"),
        )

        # should be equal to the expected dataframe
        self.assertDataFrameEquality(featured_df, dfExpected)


class ResampleTest(SparkTest):
    def test_resample(self):
        """Test of range stats for 20 minute rolling window"""

        # construct dataframes
        tsdf_input = self.get_data_as_tsdf("input")
        dfExpected = self.get_data_as_sdf("expected")
        expected_30s_df = self.get_data_as_sdf("expected30m")
        barsExpected = self.get_data_as_sdf("expectedbars")

        # 1 minute aggregation
        featured_df = tsdf_input.resample(freq="min", func="floor", prefix="floor").df
        # 30 minute aggregation
        resample_30m = tsdf_input.resample(freq="5 minutes", func="mean").df.withColumn(
            "trade_pr", Fn.round(Fn.col("trade_pr"), 2)
        )

        bars = tsdf_input.calc_bars(
            freq="min", metricCols=["trade_pr", "trade_pr_2"]
        ).df

        # should be equal to the expected dataframe
        self.assertDataFrameEquality(featured_df, dfExpected)
        self.assertDataFrameEquality(resample_30m, expected_30s_df)

        # test bars summary
        self.assertDataFrameEquality(bars, barsExpected)

    def test_resample_millis(self):
        """Test of resampling for millisecond windows"""

        # construct dataframes
        tsdf_init = self.get_data_as_tsdf("init")
        dfExpected = self.get_data_as_sdf("expectedms")

        # 30 minute aggregation
        resample_ms = tsdf_init.resample(freq="ms", func="mean").df.withColumn(
            "trade_pr", Fn.round(Fn.col("trade_pr"), 2)
        )

        self.assertDataFrameEquality(resample_ms, dfExpected)

    def test_upsample(self):
        """Test of range stats for 20 minute rolling window"""

        # construct dataframes
        tsdf_input = self.get_data_as_tsdf("input")
        expected_30s_df = self.get_data_as_sdf("expected30m")
        barsExpected = self.get_data_as_sdf("expectedbars")

        resample_30m = tsdf_input.resample(
            freq="5 minutes", func="mean", fill=True
        ).df.withColumn("trade_pr", Fn.round(Fn.col("trade_pr"), 2))

        bars = tsdf_input.calc_bars(
            freq="min", metricCols=["trade_pr", "trade_pr_2"]
        ).df

        upsampled = resample_30m.filter(
            Fn.col("event_ts").isin(
                "2020-08-01 00:00:00",
                "2020-08-01 00:05:00",
                "2020-09-01 00:00:00",
                "2020-09-01 00:15:00",
            )
        )

        # test upsample summary
        self.assertDataFrameEquality(upsampled, expected_30s_df)

        # test bars summary
        self.assertDataFrameEquality(bars, barsExpected)


class ExtractStateIntervalsTest(SparkTest):
    """Test of finding time ranges for metrics with constant state."""

    def test_eq_0(self):
        # construct dataframes
        input_tsdf: TSDF = self.get_data_as_tsdf("input")
        expected_df: DataFrame = self.get_data_as_sdf("expected")

        # call extractStateIntervals method
        intervals_eq_1_df: DataFrame = input_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3"
        )
        intervals_eq_2_df: DataFrame = input_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition="=="
        )

        # test extractStateIntervals_tsdf summary
        self.assertDataFrameEquality(intervals_eq_1_df, expected_df)
        self.assertDataFrameEquality(intervals_eq_2_df, expected_df)

    def test_eq_1(self):
        # construct dataframes
        input_tsdf: TSDF = self.get_data_as_tsdf("input")
        expected_df: DataFrame = self.get_data_as_sdf("expected")

        # call extractStateIntervals method
        intervals_eq_1_df: DataFrame = input_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3"
        )
        intervals_eq_2_df: DataFrame = input_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition="=="
        )

        # test extractStateIntervals_tsdf summary
        self.assertDataFrameEquality(intervals_eq_1_df, expected_df)
        self.assertDataFrameEquality(intervals_eq_2_df, expected_df)

    def test_ne_0(self):
        # construct dataframes
        input_tsdf: TSDF = self.get_data_as_tsdf("input")
        expected_df: DataFrame = self.get_data_as_sdf("expected")

        # call extractStateIntervals method
        intervals_ne_0_df: DataFrame = input_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition="!="
        )
        intervals_ne_1_df: DataFrame = input_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition="<>"
        )

        # test extractStateIntervals_tsdf summary
        self.assertDataFrameEquality(intervals_ne_0_df, expected_df)
        self.assertDataFrameEquality(intervals_ne_1_df, expected_df)

    def test_ne_1(self):
        # construct dataframes
        input_tsdf: TSDF = self.get_data_as_tsdf("input")
        expected_df: DataFrame = self.get_data_as_sdf("expected")

        # call extractStateIntervals method
        intervals_ne_0_df: DataFrame = input_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition="!="
        )
        intervals_ne_1_df: DataFrame = input_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition="<>"
        )

        # test extractStateIntervals_tsdf summary
        self.assertDataFrameEquality(intervals_ne_0_df, expected_df)
        self.assertDataFrameEquality(intervals_ne_1_df, expected_df)

    def test_gt_0(self):
        # construct dataframes
        input_tsdf: TSDF = self.get_data_as_tsdf("input")
        expected_df: DataFrame = self.get_data_as_sdf("expected")

        # call extractStateIntervals method
        intervals_gt_df: DataFrame = input_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition=">"
        )

        self.assertDataFrameEquality(intervals_gt_df, expected_df)

    def test_gt_1(self):
        # construct dataframes
        input_tsdf: TSDF = self.get_data_as_tsdf("input")
        expected_df: DataFrame = self.get_data_as_sdf("expected")

        # call extractStateIntervals method
        intervals_gt_df: DataFrame = input_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition=">"
        )

        self.assertDataFrameEquality(intervals_gt_df, expected_df)

    def test_lt_0(self):
        # construct dataframes
        input_tsdf: TSDF = self.get_data_as_tsdf("input")
        expected_df: DataFrame = self.get_data_as_sdf("expected")

        # call extractStateIntervals method
        intervals_lt_df: DataFrame = input_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition="<"
        )

        # test extractStateIntervals_tsdf summary
        self.assertDataFrameEquality(intervals_lt_df, expected_df)

    def test_lt_1(self):
        # construct dataframes
        input_tsdf: TSDF = self.get_data_as_tsdf("input")
        expected_df: DataFrame = self.get_data_as_sdf("expected")

        # call extractStateIntervals method
        intervals_lt_df: DataFrame = input_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition="<"
        )

        # test intervals_tsdf summary
        self.assertDataFrameEquality(intervals_lt_df, expected_df)

    def test_gte_0(self):
        # construct dataframes
        input_tsdf: TSDF = self.get_data_as_tsdf("input")
        expected_df: DataFrame = self.get_data_as_sdf("expected")

        # call extractStateIntervals method
        intervals_gt_df: DataFrame = input_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition=">="
        )

        self.assertDataFrameEquality(intervals_gt_df, expected_df)

    def test_gte_1(self):
        # construct dataframes
        input_tsdf: TSDF = self.get_data_as_tsdf("input")
        expected_df: DataFrame = self.get_data_as_sdf("expected")

        # call extractStateIntervals method
        intervals_gt_df: DataFrame = input_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition=">="
        )

        self.assertDataFrameEquality(intervals_gt_df, expected_df)

    def test_lte_0(self):
        # construct dataframes
        input_tsdf: TSDF = self.get_data_as_tsdf("input")
        expected_df: DataFrame = self.get_data_as_sdf("expected")

        # call extractStateIntervals method
        intervals_lte_df: DataFrame = input_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition="<="
        )

        # test intervals_tsdf summary
        self.assertDataFrameEquality(intervals_lte_df, expected_df)

    def test_lte_1(self):
        # construct dataframes
        input_tsdf: TSDF = self.get_data_as_tsdf("input")
        expected_df: DataFrame = self.get_data_as_sdf("expected")

        # call extractStateIntervals method
        intervals_lte_df: DataFrame = input_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition="<="
        )

        # test extractStateIntervals_tsdf summary
        self.assertDataFrameEquality(intervals_lte_df, expected_df)

    def test_threshold_fn(self):
        # construct dataframes
        input_tsdf: TSDF = self.get_data_as_tsdf("input")
        expected_df: DataFrame = self.get_data_as_sdf("expected")

        # threshold state function
        def threshold_fn(a: Column, b: Column) -> Column:
            return Fn.abs(a - b) < Fn.lit(0.5)

        # call extractStateIntervals method
        extracted_intervals_df: DataFrame = input_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition=threshold_fn
        )

        # test extractStateIntervals_tsdf summary
        self.assertDataFrameEquality(extracted_intervals_df, expected_df)

    def test_null_safe_eq_0(self):
        # construct dataframes
        input_tsdf: TSDF = self.get_data_as_tsdf("input")
        expected_df: DataFrame = self.get_data_as_sdf("expected")

        intervals_eq_df: DataFrame = input_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition="<=>"
        )

        # test extractStateIntervals_tsdf summary
        self.assertDataFrameEquality(
            intervals_eq_df, expected_df, ignore_nullable=False
        )

    def test_null_safe_eq_1(self):
        # construct dataframes
        input_tsdf: TSDF = self.get_data_as_tsdf("input")
        expected_df: DataFrame = self.get_data_as_sdf("expected")

        intervals_eq_df: DataFrame = input_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition="<=>"
        )

        # test extractStateIntervals_tsdf summary
        self.assertDataFrameEquality(
            intervals_eq_df, expected_df, ignore_nullable=False
        )

    def test_adjacent_intervals(self):
        # construct dataframes
        input_tsdf: TSDF = self.get_data_as_tsdf("input")
        expected_df: DataFrame = self.get_data_as_sdf("expected")

        intervals_eq_df: DataFrame = input_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3"
        )

        # test extractStateIntervals_tsdf summary
        self.assertDataFrameEquality(intervals_eq_df, expected_df)

    def test_invalid_state_definition_str(self):
        # construct dataframes
        input_tsdf: TSDF = self.get_data_as_tsdf("input")

        try:
            input_tsdf.extractStateIntervals(
                "metric_1", "metric_2", "metric_3", state_definition="N/A"
            )
        except ValueError as e:
            self.assertEqual(type(e), ValueError)

    def test_invalid_state_definition_type(self):
        # construct dataframes
        input_tsdf: TSDF = self.get_data_as_tsdf("input")

        try:
            input_tsdf.extractStateIntervals(
                "metric_1", "metric_2", "metric_3", state_definition=0
            )
        except TypeError as e:
            self.assertEqual(type(e), TypeError)


# MAIN
if __name__ == "__main__":
    unittest.main()
