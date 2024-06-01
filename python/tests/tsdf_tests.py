from parameterized import parameterized

from tempo.tsdf import TSDF
from tempo.tsschema import (
    SimpleTimestampIndex,
    SimpleDateIndex,
    OrdinalTSIndex,
    ParsedTimestampIndex,
    ParsedDateIndex,
    TSSchema,
)
from tests.base import TestDataFrameBuilder, SparkTest


class TSDFBaseTests(SparkTest):
    @parameterized.expand([
        ("simple_ts_idx", SimpleTimestampIndex),
        ("simple_ts_no_series", SimpleTimestampIndex),
        ("simple_date_idx", SimpleDateIndex),
        ("ordinal_double_index", OrdinalTSIndex),
        ("ordinal_int_index", OrdinalTSIndex),
        ("parsed_ts_idx", ParsedTimestampIndex),
        ("parsed_date_idx", ParsedDateIndex),
    ])
    def test_tsdf_constructor(self, init_tsdf_id, expected_idx_class):
        # create TSDF
        init_tsdf = self.get_test_df_builder(init_tsdf_id).as_tsdf()
        # check that TSDF was created correctly
        self.assertIsNotNone(init_tsdf)
        self.assertIsInstance(init_tsdf, TSDF)
        # validate the TSSchema
        self.assertIsNotNone(init_tsdf.ts_schema)
        self.assertIsInstance(init_tsdf.ts_schema, TSSchema)
        # validate the TSIndex
        self.assertIsNotNone(init_tsdf.ts_index)
        self.assertIsInstance(init_tsdf.ts_index, expected_idx_class)

    @parameterized.expand([
        ("simple_ts_idx", ["symbol"]),
        ("simple_ts_no_series", []),
        ("simple_date_idx", ["station"]),
        ("ordinal_double_index", ["symbol"]),
        ("ordinal_int_index", ["symbol"]),
        ("parsed_ts_idx", ["symbol"]),
        ("parsed_date_idx", ["station"]),
    ])
    def test_series_ids(self, init_tsdf_id, expected_series_ids):
        # load TSDF
        tsdf = self.get_test_df_builder(init_tsdf_id).as_tsdf()
        # validate series ids
        self.assertEqual(set(tsdf.series_ids), set(expected_series_ids))

    @parameterized.expand([
        ("simple_ts_idx", ["event_ts", "symbol"]),
        ("simple_ts_no_series", ["event_ts"]),
        ("simple_date_idx", ["date", "station"]),
        ("ordinal_double_index", ["event_ts_dbl", "symbol"]),
        ("ordinal_int_index", ["order", "symbol"]),
        ("parsed_ts_idx", ["ts_idx", "symbol"]),
        ("parsed_date_idx", ["ts_idx", "station"]),
    ])
    def test_structural_cols(self, init_tsdf_id, expected_structural_cols):
        # load TSDF
        tsdf = self.get_test_df_builder(init_tsdf_id).as_tsdf()
        # validate structural cols
        self.assertEqual(set(tsdf.structural_cols), set(expected_structural_cols))

    @parameterized.expand([
        ("simple_ts_idx", ["trade_pr"]),
        ("simple_ts_no_series", ["trade_pr"]),
        ("simple_date_idx", ["temp"]),
        ("ordinal_double_index", ["trade_pr"]),
        ("ordinal_int_index", ["trade_pr"]),
        ("parsed_ts_idx", ["trade_pr"]),
        ("parsed_date_idx", ["temp"]),
    ])
    def test_obs_cols(self, init_tsdf_id, expected_obs_cols):
        # load TSDF
        tsdf = self.get_test_df_builder(init_tsdf_id).as_tsdf()
        # validate obs cols
        self.assertEqual(set(tsdf.observational_cols), set(expected_obs_cols))

    @parameterized.expand([
        ("simple_ts_idx", ["trade_pr"]),
        ("simple_ts_no_series", ["trade_pr"]),
        ("simple_date_idx", ["temp"]),
        ("ordinal_double_index", ["trade_pr"]),
        ("ordinal_int_index", ["trade_pr"]),
        ("parsed_ts_idx", ["trade_pr"]),
        ("parsed_date_idx", ["temp"]),
    ])
    def test_metric_cols(self, init_tsdf_id, expected_metric_cols):
        # load TSDF
        tsdf = self.get_test_df_builder(init_tsdf_id).as_tsdf()
        # validate metric cols
        self.assertEqual(set(tsdf.metric_cols), set(expected_metric_cols))


class TimeSlicingTests(SparkTest):
    @parameterized.expand([
        (
            "simple_ts_idx",
            "2020-09-01 00:02:10",
            {
                "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, event_ts string, trade_pr float",
                    "ts_convert": ["event_ts"],
                    "data": [
                        ["S1", "2020-09-01 00:02:10", 361.1],
                        ["S2", "2020-09-01 00:02:10", 761.10],
                    ],
                },
            },
        ),
        (
            "simple_ts_no_series",
            "2020-09-01 00:19:12",
            {
                "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                "df": {
                    "schema": "event_ts string, trade_pr float",
                    "ts_convert": ["event_ts"],
                    "data": [
                        ["2020-09-01 00:19:12", 362.1],
                    ],
                },
            },
        ),
        (
            "simple_date_idx",
            "2020-08-02",
            {
                "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                "df": {
                    "schema": "station string, date string, temp float",
                    "date_convert": ["date"],
                    "data": [
                        ["LGA", "2020-08-02", 28.79],
                        ["YYZ", "2020-08-02", 22.25],
                    ],
                },
            },
        ),
        (
            "ordinal_double_index",
            10.0,
            {
                "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, event_ts_dbl double, trade_pr float",
                    "data": [
                        ["S1", 10.0, 361.1],
                        ["S2", 10.0, 762.33],
                    ],
                },
            },
        ),
        (
            "ordinal_int_index",
            1,
            {
                "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, order int, trade_pr float",
                    "data": [
                        ["S1", 1, 349.21],
                        ["S2", 1, 751.92],
                    ],
                },
            },
        ),
        (
            "parsed_ts_idx",
            "2020-09-01 00:02:10.032",
            {
                "ts_idx": {
                    "ts_col": "event_ts",
                    "series_ids": ["symbol"],
                    "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                },
                "tsdf_constructor": "fromStringTimestamp",
                "df": {
                    "schema": "symbol string, event_ts string, trade_pr float",
                    "data": [
                        ["S1", "2020-09-01 00:02:10.032", 361.1],
                    ],
                },
            },
        ),
        (
            "parsed_date_idx",
            "2020-08-04",
            {
                "ts_idx": {
                    "ts_col": "date",
                    "series_ids": ["station"],
                    "ts_fmt": "yyyy-MM-dd",
                },
                "tsdf_constructor": "fromStringTimestamp",
                "df": {
                    "schema": "station string, date string, temp float",
                    "data": [
                        ["LGA", "2020-08-04", 25.57],
                        ["YYZ", "2020-08-04", 20.65],
                    ],
                },
            },
        ),
    ])
    def test_at(self, init_tsdf_id, ts, expected_tsdf_dict):
        # load TSDF
        tsdf = self.get_test_df_builder(init_tsdf_id).as_tsdf()
        # slice at timestamp
        at_tsdf = tsdf.at(ts)
        # validate the slice
        expected_tsdf = TestDataFrameBuilder(self.spark, expected_tsdf_dict).as_tsdf()
        self.assertDataFrameEquality(at_tsdf, expected_tsdf)

    @parameterized.expand([
        (
            "simple_ts_idx",
            "2020-09-01 00:02:10",
            {
                "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, event_ts string, trade_pr float",
                    "ts_convert": ["event_ts"],
                    "data": [
                        ["S1", "2020-08-01 00:00:10", 349.21],
                        ["S1", "2020-08-01 00:01:12", 351.32],
                        ["S2", "2020-08-01 00:01:10", 743.01],
                        ["S2", "2020-08-01 00:01:24", 751.92],
                    ],
                },
            },
        ),
        (
            "simple_ts_no_series",
            "2020-09-01 00:19:12",
            {
                "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                "df": {
                    "schema": "event_ts string, trade_pr float",
                    "ts_convert": ["event_ts"],
                    "data": [
                        ["2020-08-01 00:00:10", 349.21],
                        ["2020-08-01 00:01:10", 743.01],
                        ["2020-08-01 00:01:12", 351.32],
                        ["2020-08-01 00:01:24", 751.92],
                        ["2020-09-01 00:02:10", 361.1],
                    ],
                },
            },
        ),
        (
            "simple_date_idx",
            "2020-08-03",
            {
                "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                "df": {
                    "schema": "station string, date string, temp float",
                    "date_convert": ["date"],
                    "data": [
                        ["LGA", "2020-08-01", 27.58],
                        ["LGA", "2020-08-02", 28.79],
                        ["YYZ", "2020-08-01", 24.16],
                        ["YYZ", "2020-08-02", 22.25],
                    ],
                },
            },
        ),
        (
            "ordinal_double_index",
            10.0,
            {
                "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, event_ts_dbl double, trade_pr float",
                    "data": [
                        ["S1", 0.13, 349.21],
                        ["S1", 1.207, 351.32],
                        ["S2", 0.005, 743.01],
                        ["S2", 0.1, 751.92],
                        ["S2", 1.0, 761.10],
                    ],
                },
            },
        ),
        (
            "ordinal_int_index",
            1,
            {
                "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, order int, trade_pr float",
                    "data": [["S2", 0, 743.01]],
                },
            },
        ),
        (
            "parsed_ts_idx",
            "2020-09-01 00:02:10.000",
            {
                "ts_idx": {
                    "ts_col": "event_ts",
                    "series_ids": ["symbol"],
                    "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                },
                "tsdf_constructor": "fromStringTimestamp",
                "df": {
                    "schema": "symbol string, event_ts string, trade_pr float",
                    "data": [
                        ["S1", "2020-08-01 00:00:10.010", 349.21],
                        ["S1", "2020-08-01 00:01:12.021", 351.32],
                        ["S2", "2020-08-01 00:01:10.054", 743.01],
                        ["S2", "2020-08-01 00:01:24.065", 751.92],
                    ],
                },
            },
        ),
        (
            "parsed_date_idx",
            "2020-08-03",
            {
                "ts_idx": {
                    "ts_col": "date",
                    "series_ids": ["station"],
                    "ts_fmt": "yyyy-MM-dd",
                },
                "tsdf_constructor": "fromStringTimestamp",
                "df": {
                    "schema": "station string, date string, temp float",
                    "data": [
                        ["LGA", "2020-08-01", 27.58],
                        ["LGA", "2020-08-02", 28.79],
                        ["YYZ", "2020-08-01", 24.16],
                        ["YYZ", "2020-08-02", 22.25],
                    ],
                },
            },
        ),
    ])
    def test_before(self, init_tsdf_id, ts, expected_tsdf_dict):
        # load TSDF
        tsdf = self.get_test_df_builder(init_tsdf_id).as_tsdf()
        # slice at timestamp
        before_tsdf = tsdf.before(ts)
        # validate the slice
        expected_tsdf = TestDataFrameBuilder(self.spark, expected_tsdf_dict).as_tsdf()
        self.assertDataFrameEquality(before_tsdf, expected_tsdf)

    @parameterized.expand([
        (
            "simple_ts_idx",
            "2020-09-01 00:02:10",
            {
                "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, event_ts string, trade_pr float",
                    "ts_convert": ["event_ts"],
                    "data": [
                        ["S1", "2020-08-01 00:00:10", 349.21],
                        ["S1", "2020-08-01 00:01:12", 351.32],
                        ["S1", "2020-09-01 00:02:10", 361.1],
                        ["S2", "2020-08-01 00:01:10", 743.01],
                        ["S2", "2020-08-01 00:01:24", 751.92],
                        ["S2", "2020-09-01 00:02:10", 761.10],
                    ],
                },
            },
        ),
        (
            "simple_ts_no_series",
            "2020-09-01 00:19:12",
            {
                "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                "df": {
                    "schema": "event_ts string, trade_pr float",
                    "ts_convert": ["event_ts"],
                    "data": [
                        ["2020-08-01 00:00:10", 349.21],
                        ["2020-08-01 00:01:10", 743.01],
                        ["2020-08-01 00:01:12", 351.32],
                        ["2020-08-01 00:01:24", 751.92],
                        ["2020-09-01 00:02:10", 361.1],
                        ["2020-09-01 00:19:12", 362.1],
                    ],
                },
            },
        ),
        (
            "simple_date_idx",
            "2020-08-03",
            {
                "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                "df": {
                    "schema": "station string, date string, temp float",
                    "date_convert": ["date"],
                    "data": [
                        ["LGA", "2020-08-01", 27.58],
                        ["LGA", "2020-08-02", 28.79],
                        ["LGA", "2020-08-03", 28.53],
                        ["YYZ", "2020-08-01", 24.16],
                        ["YYZ", "2020-08-02", 22.25],
                        ["YYZ", "2020-08-03", 20.62],
                    ],
                },
            },
        ),
        (
            "ordinal_double_index",
            10.0,
            {
                "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, event_ts_dbl double, trade_pr float",
                    "data": [
                        ["S1", 0.13, 349.21],
                        ["S1", 1.207, 351.32],
                        ["S1", 10.0, 361.1],
                        ["S2", 0.005, 743.01],
                        ["S2", 0.1, 751.92],
                        ["S2", 1.0, 761.10],
                        ["S2", 10.0, 762.33],
                    ],
                },
            },
        ),
        (
            "ordinal_int_index",
            1,
            {
                "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, order int, trade_pr float",
                    "data": [["S1", 1, 349.21], ["S2", 0, 743.01], ["S2", 1, 751.92]],
                },
            },
        ),
        (
            "parsed_ts_idx",
            "2020-09-01 00:02:10.000",
            {
                "ts_idx": {
                    "ts_col": "event_ts",
                    "series_ids": ["symbol"],
                    "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                },
                "tsdf_constructor": "fromStringTimestamp",
                "df": {
                    "schema": "symbol string, event_ts string, trade_pr float",
                    "data": [
                        ["S1", "2020-08-01 00:00:10.010", 349.21],
                        ["S1", "2020-08-01 00:01:12.021", 351.32],
                        ["S2", "2020-08-01 00:01:10.054", 743.01],
                        ["S2", "2020-08-01 00:01:24.065", 751.92],
                    ],
                },
            },
        ),
        (
            "parsed_date_idx",
            "2020-08-03",
            {
                "ts_idx": {
                    "ts_col": "date",
                    "series_ids": ["station"],
                    "ts_fmt": "yyyy-MM-dd",
                },
                "tsdf_constructor": "fromStringTimestamp",
                "df": {
                    "schema": "station string, date string, temp float",
                    "data": [
                        ["LGA", "2020-08-01", 27.58],
                        ["LGA", "2020-08-02", 28.79],
                        ["LGA", "2020-08-03", 28.53],
                        ["YYZ", "2020-08-01", 24.16],
                        ["YYZ", "2020-08-02", 22.25],
                        ["YYZ", "2020-08-03", 20.62],
                    ],
                },
            },
        ),
    ])
    def test_atOrBefore(self, init_tsdf_id, ts, expected_tsdf_dict):
        # load TSDF
        tsdf = self.get_test_df_builder(init_tsdf_id).as_tsdf()
        # slice at timestamp
        at_before_tsdf = tsdf.atOrBefore(ts)
        # validate the slice
        expected_tsdf = TestDataFrameBuilder(self.spark, expected_tsdf_dict).as_tsdf()
        self.assertDataFrameEquality(at_before_tsdf, expected_tsdf)

    @parameterized.expand([
        (
            "simple_ts_idx",
            "2020-09-01 00:02:10",
            {
                "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, event_ts string, trade_pr float",
                    "ts_convert": ["event_ts"],
                    "data": [
                        ["S1", "2020-09-01 00:19:12", 362.1],
                        ["S2", "2020-09-01 00:20:42", 762.33],
                    ],
                },
            },
        ),
        (
            "simple_ts_no_series",
            "2020-09-01 00:08:12",
            {
                "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                "df": {
                    "schema": "event_ts string, trade_pr float",
                    "ts_convert": ["event_ts"],
                    "data": [
                        ["2020-09-01 00:19:12", 362.1],
                        ["2020-09-01 00:20:42", 762.33],
                    ],
                },
            },
        ),
        (
            "simple_date_idx",
            "2020-08-02",
            {
                "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                "df": {
                    "schema": "station string, date string, temp float",
                    "date_convert": ["date"],
                    "data": [
                        ["LGA", "2020-08-03", 28.53],
                        ["LGA", "2020-08-04", 25.57],
                        ["YYZ", "2020-08-03", 20.62],
                        ["YYZ", "2020-08-04", 20.65],
                    ],
                },
            },
        ),
        (
            "ordinal_double_index",
            1.0,
            {
                "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, event_ts_dbl double, trade_pr float",
                    "data": [
                        ["S1", 1.207, 351.32],
                        ["S1", 10.0, 361.1],
                        ["S1", 24.357, 362.1],
                        ["S2", 10.0, 762.33],
                    ],
                },
            },
        ),
        (
            "ordinal_int_index",
            10,
            {
                "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, order int, trade_pr float",
                    "data": [
                        ["S1", 20, 351.32],
                        ["S1", 127, 361.1],
                        ["S1", 243, 362.1],
                        ["S2", 100, 762.33],
                    ],
                },
            },
        ),
        (
            "parsed_ts_idx",
            "2020-09-01 00:02:10.000",
            {
                "ts_idx": {
                    "ts_col": "event_ts",
                    "series_ids": ["symbol"],
                    "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                },
                "tsdf_constructor": "fromStringTimestamp",
                "df": {
                    "schema": "symbol string, event_ts string, trade_pr float",
                    "data": [
                        ["S1", "2020-09-01 00:02:10.032", 361.1],
                        ["S1", "2020-09-01 00:19:12.043", 362.1],
                        ["S2", "2020-09-01 00:02:10.076", 761.10],
                        ["S2", "2020-09-01 00:20:42.087", 762.33],
                    ],
                },
            },
        ),
        (
            "parsed_date_idx",
            "2020-08-03",
            {
                "ts_idx": {
                    "ts_col": "date",
                    "series_ids": ["station"],
                    "ts_fmt": "yyyy-MM-dd",
                },
                "tsdf_constructor": "fromStringTimestamp",
                "df": {
                    "schema": "station string, date string, temp float",
                    "data": [
                        ["LGA", "2020-08-04", 25.57],
                        ["YYZ", "2020-08-04", 20.65],
                    ],
                },
            },
        ),
    ])
    def test_after(self, init_tsdf_id, ts, expected_tsdf_dict):
        # load TSDF
        tsdf = self.get_test_df_builder(init_tsdf_id).as_tsdf()
        # slice at timestamp
        after_tsdf = tsdf.after(ts)
        # validate the slice
        expected_tsdf = TestDataFrameBuilder(self.spark, expected_tsdf_dict).as_tsdf()
        self.assertDataFrameEquality(after_tsdf, expected_tsdf)

    @parameterized.expand([
        (
            "simple_ts_idx",
            "2020-09-01 00:02:10",
            {
                "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, event_ts string, trade_pr float",
                    "ts_convert": ["event_ts"],
                    "data": [
                        ["S1", "2020-09-01 00:02:10", 361.1],
                        ["S1", "2020-09-01 00:19:12", 362.1],
                        ["S2", "2020-09-01 00:02:10", 761.10],
                        ["S2", "2020-09-01 00:20:42", 762.33],
                    ],
                },
            },
        ),
        (
            "simple_ts_no_series",
            "2020-08-01 00:01:24",
            {
                "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                "df": {
                    "schema": "event_ts string, trade_pr float",
                    "ts_convert": ["event_ts"],
                    "data": [
                        ["2020-08-01 00:01:24", 751.92],
                        ["2020-09-01 00:02:10", 361.1],
                        ["2020-09-01 00:19:12", 362.1],
                        ["2020-09-01 00:20:42", 762.33],
                    ],
                },
            },
        ),
        (
            "simple_date_idx",
            "2020-08-03",
            {
                "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                "df": {
                    "schema": "station string, date string, temp float",
                    "date_convert": ["date"],
                    "data": [
                        ["LGA", "2020-08-03", 28.53],
                        ["LGA", "2020-08-04", 25.57],
                        ["YYZ", "2020-08-03", 20.62],
                        ["YYZ", "2020-08-04", 20.65],
                    ],
                },
            },
        ),
        (
            "ordinal_double_index",
            10.0,
            {
                "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, event_ts_dbl double, trade_pr float",
                    "data": [
                        ["S1", 10.0, 361.1],
                        ["S1", 24.357, 362.1],
                        ["S2", 10.0, 762.33],
                    ],
                },
            },
        ),
        (
            "ordinal_int_index",
            10,
            {
                "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, order int, trade_pr float",
                    "data": [
                        ["S1", 20, 351.32],
                        ["S1", 127, 361.1],
                        ["S1", 243, 362.1],
                        ["S2", 10, 761.10],
                        ["S2", 100, 762.33],
                    ],
                },
            },
        ),
        (
            "parsed_ts_idx",
            "2020-09-01 00:02:10.000",
            {
                "ts_idx": {
                    "ts_col": "event_ts",
                    "series_ids": ["symbol"],
                    "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                },
                "tsdf_constructor": "fromStringTimestamp",
                "df": {
                    "schema": "symbol string, event_ts string, trade_pr float",
                    "data": [
                        ["S1", "2020-09-01 00:02:10.032", 361.1],
                        ["S1", "2020-09-01 00:19:12.043", 362.1],
                        ["S2", "2020-09-01 00:02:10.076", 761.10],
                        ["S2", "2020-09-01 00:20:42.087", 762.33],
                    ],
                },
            },
        ),
        (
            "parsed_date_idx",
            "2020-08-03",
            {
                "ts_idx": {
                    "ts_col": "date",
                    "series_ids": ["station"],
                    "ts_fmt": "yyyy-MM-dd",
                },
                "tsdf_constructor": "fromStringTimestamp",
                "df": {
                    "schema": "station string, date string, temp float",
                    "data": [
                        ["LGA", "2020-08-03", 28.53],
                        ["LGA", "2020-08-04", 25.57],
                        ["YYZ", "2020-08-03", 20.62],
                        ["YYZ", "2020-08-04", 20.65],
                    ],
                },
            },
        ),
    ])
    def test_atOrAfter(self, init_tsdf_id, ts, expected_tsdf_dict):
        # load TSDF
        tsdf = self.get_test_df_builder(init_tsdf_id).as_tsdf()
        # slice at timestamp
        at_after_tsdf = tsdf.atOrAfter(ts)
        # validate the slice
        expected_tsdf = TestDataFrameBuilder(self.spark, expected_tsdf_dict).as_tsdf()
        self.assertDataFrameEquality(at_after_tsdf, expected_tsdf)

    @parameterized.expand([
        (
            "simple_ts_idx",
            "2020-08-01 00:01:10",
            "2020-09-01 00:02:10",
            {
                "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, event_ts string, trade_pr float",
                    "ts_convert": ["event_ts"],
                    "data": [
                        ["S1", "2020-08-01 00:01:12", 351.32],
                        ["S2", "2020-08-01 00:01:24", 751.92],
                    ],
                },
            },
        ),
        (
            "simple_ts_no_series",
            "2020-08-01 00:01:10",
            "2020-09-01 00:02:10",
            {
                "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                "df": {
                    "schema": "event_ts string, trade_pr float",
                    "ts_convert": ["event_ts"],
                    "data": [
                        ["2020-08-01 00:01:12", 351.32],
                        ["2020-08-01 00:01:24", 751.92],
                    ],
                },
            },
        ),
        (
            "simple_date_idx",
            "2020-08-01",
            "2020-08-03",
            {
                "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                "df": {
                    "schema": "station string, date string, temp float",
                    "date_convert": ["date"],
                    "data": [
                        ["LGA", "2020-08-02", 28.79],
                        ["YYZ", "2020-08-02", 22.25],
                    ],
                },
            },
        ),
        (
            "ordinal_double_index",
            0.1,
            10.0,
            {
                "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, event_ts_dbl double, trade_pr float",
                    "data": [
                        ["S1", 0.13, 349.21],
                        ["S1", 1.207, 351.32],
                        ["S2", 1.0, 761.10],
                    ],
                },
            },
        ),
        (
            "ordinal_int_index",
            1,
            100,
            {
                "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, order int, trade_pr float",
                    "data": [["S1", 20, 351.32], ["S2", 10, 761.10]],
                },
            },
        ),
        (
            "parsed_ts_idx",
            "2020-08-01 00:00:10.010",
            "2020-09-01 00:02:10.076",
            {
                "ts_idx": {
                    "ts_col": "event_ts",
                    "series_ids": ["symbol"],
                    "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                },
                "tsdf_constructor": "fromStringTimestamp",
                "df": {
                    "schema": "symbol string, event_ts string, trade_pr float",
                    "data": [
                        ["S1", "2020-08-01 00:01:12.021", 351.32],
                        ["S1", "2020-09-01 00:02:10.032", 361.1],
                        ["S2", "2020-08-01 00:01:10.054", 743.01],
                        ["S2", "2020-08-01 00:01:24.065", 751.92],
                    ],
                },
            },
        ),
        (
            "parsed_date_idx",
            "2020-08-01",
            "2020-08-03",
            {
                "ts_idx": {
                    "ts_col": "date",
                    "series_ids": ["station"],
                    "ts_fmt": "yyyy-MM-dd",
                },
                "tsdf_constructor": "fromStringTimestamp",
                "df": {
                    "schema": "station string, date string, temp float",
                    "data": [
                        ["LGA", "2020-08-02", 28.79],
                        ["YYZ", "2020-08-02", 22.25],
                    ],
                },
            },
        ),
    ])
    def test_between_non_inclusive(
        self, init_tsdf_id, start_ts, end_ts, expected_tsdf_dict
    ):
        # load TSDF
        tsdf = self.get_test_df_builder(init_tsdf_id).as_tsdf()
        # slice at timestamp
        between_tsdf = tsdf.between(start_ts, end_ts, inclusive=False)
        # validate the slice
        expected_tsdf = TestDataFrameBuilder(self.spark, expected_tsdf_dict).as_tsdf()
        self.assertDataFrameEquality(between_tsdf, expected_tsdf)

    @parameterized.expand([
        (
            "simple_ts_idx",
            "2020-08-01 00:01:10",
            "2020-09-01 00:02:10",
            {
                "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, event_ts string, trade_pr float",
                    "ts_convert": ["event_ts"],
                    "data": [
                        ["S1", "2020-08-01 00:01:12", 351.32],
                        ["S1", "2020-09-01 00:02:10", 361.1],
                        ["S2", "2020-08-01 00:01:10", 743.01],
                        ["S2", "2020-08-01 00:01:24", 751.92],
                        ["S2", "2020-09-01 00:02:10", 761.10],
                    ],
                },
            },
        ),
        (
            "simple_ts_no_series",
            "2020-08-01 00:01:10",
            "2020-09-01 00:02:10",
            {
                "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                "df": {
                    "schema": "event_ts string, trade_pr float",
                    "ts_convert": ["event_ts"],
                    "data": [
                        ["2020-08-01 00:01:10", 743.01],
                        ["2020-08-01 00:01:12", 351.32],
                        ["2020-08-01 00:01:24", 751.92],
                        ["2020-09-01 00:02:10", 361.1],
                    ],
                },
            },
        ),
        (
            "simple_date_idx",
            "2020-08-01",
            "2020-08-03",
            {
                "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                "df": {
                    "schema": "station string, date string, temp float",
                    "date_convert": ["date"],
                    "data": [
                        ["LGA", "2020-08-01", 27.58],
                        ["LGA", "2020-08-02", 28.79],
                        ["LGA", "2020-08-03", 28.53],
                        ["YYZ", "2020-08-01", 24.16],
                        ["YYZ", "2020-08-02", 22.25],
                        ["YYZ", "2020-08-03", 20.62],
                    ],
                },
            },
        ),
        (
            "ordinal_double_index",
            0.1,
            10.0,
            {
                "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, event_ts_dbl double, trade_pr float",
                    "data": [
                        ["S1", 0.13, 349.21],
                        ["S1", 1.207, 351.32],
                        ["S1", 10.0, 361.1],
                        ["S2", 0.1, 751.92],
                        ["S2", 1.0, 761.10],
                        ["S2", 10.0, 762.33],
                    ],
                },
            },
        ),
        (
            "ordinal_int_index",
            1,
            100,
            {
                "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, order int, trade_pr float",
                    "data": [
                        ["S1", 1, 349.21],
                        ["S1", 20, 351.32],
                        ["S2", 1, 751.92],
                        ["S2", 10, 761.10],
                        ["S2", 100, 762.33],
                    ],
                },
            },
        ),
        (
            "parsed_ts_idx",
            "2020-08-01 00:00:10.010",
            "2020-09-01 00:02:10.076",
            {
                "ts_idx": {
                    "ts_col": "event_ts",
                    "series_ids": ["symbol"],
                    "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                },
                "tsdf_constructor": "fromStringTimestamp",
                "df": {
                    "schema": "symbol string, event_ts string, trade_pr float",
                    "data": [
                        ["S1", "2020-08-01 00:00:10.010", 349.21],
                        ["S1", "2020-08-01 00:01:12.021", 351.32],
                        ["S1", "2020-09-01 00:02:10.032", 361.1],
                        ["S2", "2020-08-01 00:01:10.054", 743.01],
                        ["S2", "2020-08-01 00:01:24.065", 751.92],
                        ["S2", "2020-09-01 00:02:10.076", 761.10],
                    ],
                },
            },
        ),
        (
            "parsed_date_idx",
            "2020-08-01",
            "2020-08-03",
            {
                "ts_idx": {
                    "ts_col": "date",
                    "series_ids": ["station"],
                    "ts_fmt": "yyyy-MM-dd",
                },
                "tsdf_constructor": "fromStringTimestamp",
                "df": {
                    "schema": "station string, date string, temp float",
                    "data": [
                        ["LGA", "2020-08-01", 27.58],
                        ["LGA", "2020-08-02", 28.79],
                        ["LGA", "2020-08-03", 28.53],
                        ["YYZ", "2020-08-01", 24.16],
                        ["YYZ", "2020-08-02", 22.25],
                        ["YYZ", "2020-08-03", 20.62],
                    ],
                },
            },
        ),
    ])
    def test_between_inclusive(
        self, init_tsdf_id, start_ts, end_ts, expected_tsdf_dict
    ):
        # load TSDF
        tsdf = self.get_test_df_builder(init_tsdf_id).as_tsdf()
        # slice at timestamp
        between_tsdf = tsdf.between(start_ts, end_ts, inclusive=True)
        # validate the slice
        expected_tsdf = TestDataFrameBuilder(self.spark, expected_tsdf_dict).as_tsdf()
        self.assertDataFrameEquality(between_tsdf, expected_tsdf)

    @parameterized.expand([
        (
            "simple_ts_idx",
            2,
            {
                "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, event_ts string, trade_pr float",
                    "ts_convert": ["event_ts"],
                    "data": [
                        ["S1", "2020-08-01 00:00:10", 349.21],
                        ["S1", "2020-08-01 00:01:12", 351.32],
                        ["S2", "2020-08-01 00:01:10", 743.01],
                        ["S2", "2020-08-01 00:01:24", 751.92],
                    ],
                },
            },
        ),
        (
            "simple_ts_no_series",
            2,
            {
                "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                "df": {
                    "schema": "event_ts string, trade_pr float",
                    "ts_convert": ["event_ts"],
                    "data": [
                        ["2020-08-01 00:00:10", 349.21],
                        ["2020-08-01 00:01:10", 743.01],
                    ],
                },
            },
        ),
        (
            "simple_date_idx",
            2,
            {
                "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                "df": {
                    "schema": "station string, date string, temp float",
                    "date_convert": ["date"],
                    "data": [
                        ["LGA", "2020-08-01", 27.58],
                        ["LGA", "2020-08-02", 28.79],
                        ["YYZ", "2020-08-01", 24.16],
                        ["YYZ", "2020-08-02", 22.25],
                    ],
                },
            },
        ),
        (
            "ordinal_double_index",
            2,
            {
                "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, event_ts_dbl double, trade_pr float",
                    "data": [
                        ["S1", 0.13, 349.21],
                        ["S1", 1.207, 351.32],
                        ["S2", 0.005, 743.01],
                        ["S2", 0.1, 751.92],
                    ],
                },
            },
        ),
        (
            "ordinal_int_index",
            2,
            {
                "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, order int, trade_pr float",
                    "data": [
                        ["S1", 1, 349.21],
                        ["S1", 20, 351.32],
                        ["S2", 0, 743.01],
                        ["S2", 1, 751.92],
                    ],
                },
            },
        ),
        (
            "parsed_ts_idx",
            2,
            {
                "ts_idx": {
                    "ts_col": "event_ts",
                    "series_ids": ["symbol"],
                    "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                },
                "tsdf_constructor": "fromStringTimestamp",
                "df": {
                    "schema": "symbol string, event_ts string, trade_pr float",
                    "data": [
                        ["S1", "2020-08-01 00:00:10.010", 349.21],
                        ["S1", "2020-08-01 00:01:12.021", 351.32],
                        ["S2", "2020-08-01 00:01:10.054", 743.01],
                        ["S2", "2020-08-01 00:01:24.065", 751.92],
                    ],
                },
            },
        ),
        (
            "parsed_date_idx",
            2,
            {
                "ts_idx": {
                    "ts_col": "date",
                    "series_ids": ["station"],
                    "ts_fmt": "yyyy-MM-dd",
                },
                "tsdf_constructor": "fromStringTimestamp",
                "df": {
                    "schema": "station string, date string, temp float",
                    "data": [
                        ["LGA", "2020-08-01", 27.58],
                        ["LGA", "2020-08-02", 28.79],
                        ["YYZ", "2020-08-01", 24.16],
                        ["YYZ", "2020-08-02", 22.25],
                    ],
                },
            },
        ),
    ])
    def test_earliest(self, init_tsdf_id, num_records, expected_tsdf_dict):
        # load TSDF
        tsdf = self.get_test_df_builder(init_tsdf_id).as_tsdf()
        # get earliest timestamp
        earliest_ts = tsdf.earliest(n=num_records)
        # validate the timestamp
        expected_tsdf = TestDataFrameBuilder(self.spark, expected_tsdf_dict).as_tsdf()
        self.assertDataFrameEquality(earliest_ts, expected_tsdf)

    @parameterized.expand([
        (
            "simple_ts_idx",
            2,
            {
                "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, event_ts string, trade_pr float",
                    "ts_convert": ["event_ts"],
                    "data": [
                        ["S1", "2020-09-01 00:19:12", 362.1],
                        ["S1", "2020-09-01 00:02:10", 361.1],
                        ["S2", "2020-09-01 00:20:42", 762.33],
                        ["S2", "2020-09-01 00:02:10", 761.10],
                    ],
                },
            },
        ),
        (
            "simple_ts_no_series",
            4,
            {
                "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                "df": {
                    "schema": "event_ts string, trade_pr float",
                    "ts_convert": ["event_ts"],
                    "data": [
                        ["2020-09-01 00:20:42", 762.33],
                        ["2020-09-01 00:19:12", 362.1],
                        ["2020-09-01 00:02:10", 361.1],
                        ["2020-08-01 00:01:24", 751.92],
                    ],
                },
            },
        ),
        (
            "simple_date_idx",
            3,
            {
                "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                "df": {
                    "schema": "station string, date string, temp float",
                    "date_convert": ["date"],
                    "data": [
                        ["LGA", "2020-08-04", 25.57],
                        ["LGA", "2020-08-03", 28.53],
                        ["LGA", "2020-08-02", 28.79],
                        ["YYZ", "2020-08-04", 20.65],
                        ["YYZ", "2020-08-03", 20.62],
                        ["YYZ", "2020-08-02", 22.25],
                    ],
                },
            },
        ),
        (
            "ordinal_double_index",
            1,
            {
                "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, event_ts_dbl double, trade_pr float",
                    "data": [["S1", 24.357, 362.1], ["S2", 10.0, 762.33]],
                },
            },
        ),
        (
            "ordinal_int_index",
            3,
            {
                "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, order int, trade_pr float",
                    "data": [
                        ["S1", 243, 362.1],
                        ["S1", 127, 361.1],
                        ["S1", 20, 351.32],
                        ["S2", 100, 762.33],
                        ["S2", 10, 761.10],
                        ["S2", 1, 751.92],
                    ],
                },
            },
        ),
        (
            "parsed_ts_idx",
            3,
            {
                "ts_idx": {
                    "ts_col": "event_ts",
                    "series_ids": ["symbol"],
                    "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                },
                "tsdf_constructor": "fromStringTimestamp",
                "df": {
                    "schema": "symbol string, event_ts string, trade_pr float",
                    "data": [
                        ["S1", "2020-09-01 00:19:12.043", 362.1],
                        ["S1", "2020-09-01 00:02:10.032", 361.1],
                        ["S1", "2020-08-01 00:01:12.021", 351.32],
                        ["S2", "2020-09-01 00:20:42.087", 762.33],
                        ["S2", "2020-09-01 00:02:10.076", 761.10],
                        ["S2", "2020-08-01 00:01:24.065", 751.92],
                    ],
                },
            },
        ),
        (
            "parsed_date_idx",
            1,
            {
                "ts_idx": {
                    "ts_col": "date",
                    "series_ids": ["station"],
                    "ts_fmt": "yyyy-MM-dd",
                },
                "tsdf_constructor": "fromStringTimestamp",
                "df": {
                    "schema": "station string, date string, temp float",
                    "data": [
                        ["LGA", "2020-08-04", 25.57],
                        ["YYZ", "2020-08-04", 20.65],
                    ],
                },
            },
        ),
    ])
    def test_latest(self, init_tsdf_id, num_records, expected_tsdf_dict):
        # load TSDF
        tsdf = self.get_test_df_builder(init_tsdf_id).as_tsdf()
        # get earliest timestamp
        latest_ts = tsdf.latest(n=num_records)
        # validate the timestamp
        expected_tsdf = TestDataFrameBuilder(self.spark, expected_tsdf_dict).as_tsdf()
        self.assertDataFrameEquality(latest_ts, expected_tsdf)

    @parameterized.expand([
        (
            "simple_ts_idx",
            "2020-09-01 00:02:10",
            2,
            {
                "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, event_ts string, trade_pr float",
                    "ts_convert": ["event_ts"],
                    "data": [
                        ["S1", "2020-09-01 00:02:10", 361.1],
                        ["S1", "2020-08-01 00:01:12", 351.32],
                        ["S2", "2020-09-01 00:02:10", 761.10],
                        ["S2", "2020-08-01 00:01:24", 751.92],
                    ],
                },
            },
        ),
        (
            "simple_ts_no_series",
            "2020-09-01 00:19:12",
            3,
            {
                "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                "df": {
                    "schema": "event_ts string, trade_pr float",
                    "ts_convert": ["event_ts"],
                    "data": [
                        ["2020-09-01 00:19:12", 362.1],
                        ["2020-09-01 00:02:10", 361.1],
                        ["2020-08-01 00:01:24", 751.92],
                    ],
                },
            },
        ),
        (
            "simple_date_idx",
            "2020-08-03",
            2,
            {
                "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                "df": {
                    "schema": "station string, date string, temp float",
                    "date_convert": ["date"],
                    "data": [
                        ["LGA", "2020-08-03", 28.53],
                        ["LGA", "2020-08-02", 28.79],
                        ["YYZ", "2020-08-03", 20.62],
                        ["YYZ", "2020-08-02", 22.25],
                    ],
                },
            },
        ),
        (
            "ordinal_double_index",
            10.0,
            4,
            {
                "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, event_ts_dbl double, trade_pr float",
                    "data": [
                        ["S1", 10.0, 361.1],
                        ["S1", 1.207, 351.32],
                        ["S1", 0.13, 349.21],
                        ["S2", 10.0, 762.33],
                        ["S2", 1.0, 761.10],
                        ["S2", 0.1, 751.92],
                        ["S2", 0.005, 743.01],
                    ],
                },
            },
        ),
        (
            "ordinal_int_index",
            1,
            1,
            {
                "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, order int, trade_pr float",
                    "data": [["S1", 1, 349.21], ["S2", 1, 751.92]],
                },
            },
        ),
        (
            "parsed_ts_idx",
            "2020-09-01 00:02:10.000",
            2,
            {
                "ts_idx": {
                    "ts_col": "event_ts",
                    "series_ids": ["symbol"],
                    "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                },
                "tsdf_constructor": "fromStringTimestamp",
                "df": {
                    "schema": "symbol string, event_ts string, trade_pr float",
                    "data": [
                        ["S1", "2020-08-01 00:01:12.021", 351.32],
                        ["S1", "2020-08-01 00:00:10.010", 349.21],
                        ["S2", "2020-08-01 00:01:24.065", 751.92],
                        ["S2", "2020-08-01 00:01:10.054", 743.01],
                    ],
                },
            },
        ),
        (
            "parsed_date_idx",
            "2020-08-03",
            3,
            {
                "ts_idx": {
                    "ts_col": "date",
                    "series_ids": ["station"],
                    "ts_fmt": "yyyy-MM-dd",
                },
                "tsdf_constructor": "fromStringTimestamp",
                "df": {
                    "schema": "station string, date string, temp float",
                    "data": [
                        ["LGA", "2020-08-03", 28.53],
                        ["LGA", "2020-08-02", 28.79],
                        ["LGA", "2020-08-01", 27.58],
                        ["YYZ", "2020-08-03", 20.62],
                        ["YYZ", "2020-08-02", 22.25],
                        ["YYZ", "2020-08-01", 24.16],
                    ],
                },
            },
        ),
    ])
    def test_priorTo(self, init_tsdf_id, ts, n, expected_tsdf_dict):
        # load TSDF
        tsdf = self.get_test_df_builder(init_tsdf_id).as_tsdf()
        # slice at timestamp
        prior_tsdf = tsdf.priorTo(ts, n=n)
        # validate the slice
        expected_tsdf = TestDataFrameBuilder(self.spark, expected_tsdf_dict).as_tsdf()
        self.assertDataFrameEquality(prior_tsdf, expected_tsdf)

    @parameterized.expand([
        (
            "simple_ts_idx",
            "2020-09-01 00:02:10",
            1,
            {
                "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, event_ts string, trade_pr float",
                    "ts_convert": ["event_ts"],
                    "data": [
                        ["S1", "2020-09-01 00:02:10", 361.1],
                        ["S2", "2020-09-01 00:02:10", 761.10],
                    ],
                },
            },
        ),
        (
            "simple_ts_no_series",
            "2020-08-01 00:01:24",
            3,
            {
                "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                "df": {
                    "schema": "event_ts string, trade_pr float",
                    "ts_convert": ["event_ts"],
                    "data": [
                        ["2020-08-01 00:01:24", 751.92],
                        ["2020-09-01 00:02:10", 361.1],
                        ["2020-09-01 00:19:12", 362.1],
                    ],
                },
            },
        ),
        (
            "simple_date_idx",
            "2020-08-02",
            5,
            {
                "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                "df": {
                    "schema": "station string, date string, temp float",
                    "date_convert": ["date"],
                    "data": [
                        ["LGA", "2020-08-02", 28.79],
                        ["LGA", "2020-08-03", 28.53],
                        ["LGA", "2020-08-04", 25.57],
                        ["YYZ", "2020-08-02", 22.25],
                        ["YYZ", "2020-08-03", 20.62],
                        ["YYZ", "2020-08-04", 20.65],
                    ],
                },
            },
        ),
        (
            "ordinal_double_index",
            10.0,
            2,
            {
                "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, event_ts_dbl double, trade_pr float",
                    "data": [
                        ["S1", 10.0, 361.1],
                        ["S1", 24.357, 362.1],
                        ["S2", 10.0, 762.33],
                    ],
                },
            },
        ),
        (
            "ordinal_int_index",
            10,
            2,
            {
                "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                "df": {
                    "schema": "symbol string, order int, trade_pr float",
                    "data": [
                        ["S1", 20, 351.32],
                        ["S1", 127, 361.1],
                        ["S2", 10, 761.10],
                        ["S2", 100, 762.33],
                    ],
                },
            },
        ),
        (
            "parsed_ts_idx",
            "2020-09-01 00:02:10",
            3,
            {
                "ts_idx": {
                    "ts_col": "event_ts",
                    "series_ids": ["symbol"],
                    "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                },
                "tsdf_constructor": "fromStringTimestamp",
                "df": {
                    "schema": "symbol string, event_ts string, trade_pr float",
                    "data": [
                        ["S1", "2020-09-01 00:02:10.032", 361.1],
                        ["S1", "2020-09-01 00:19:12.043", 362.1],
                        ["S2", "2020-09-01 00:02:10.076", 761.10],
                        ["S2", "2020-09-01 00:20:42.087", 762.33],
                    ],
                },
            },
        ),
        (
            "parsed_date_idx",
            "2020-08-03",
            2,
            {
                "ts_idx": {
                    "ts_col": "date",
                    "series_ids": ["station"],
                    "ts_fmt": "yyyy-MM-dd",
                },
                "tsdf_constructor": "fromStringTimestamp",
                "df": {
                    "schema": "station string, date string, temp float",
                    "data": [
                        ["LGA", "2020-08-03", 28.53],
                        ["LGA", "2020-08-04", 25.57],
                        ["YYZ", "2020-08-03", 20.62],
                        ["YYZ", "2020-08-04", 20.65],
                    ],
                },
            },
        ),
    ])
    def test_subsequentTo(self, init_tsdf_id, ts, n, expected_tsdf_dict):
        # load TSDF
        tsdf = self.get_test_df_builder(init_tsdf_id).as_tsdf()
        # slice at timestamp
        subseq_tsdf = tsdf.subsequentTo(ts, n=n)
        # validate the slice
        expected_tsdf = TestDataFrameBuilder(self.spark, expected_tsdf_dict).as_tsdf()
        self.assertDataFrameEquality(subseq_tsdf, expected_tsdf)
