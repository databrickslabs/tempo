import unittest
from abc import ABC
from typing import List

from parameterized import parameterized_class
from pyspark.sql import Column, WindowSpec
from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from tempo.tsschema import (
    OrdinalTSIndex,
    ParsedDateIndex,
    ParsedTimestampIndex,
    SimpleDateIndex,
    SimpleTimestampIndex,
    StandardTimeUnits,
    SubMicrosecondPrecisionTimestampIndex,
    SubsequenceTSIndex,
    TSIndex,
    TSSchema,
)
from tests.base import SparkTest


class TSIndexTester(unittest.TestCase, ABC):
    def _test_index(self, ts_idx: TSIndex):
        # must be a valid TSIndex object
        self.assertIsNotNone(ts_idx)
        self.assertIsInstance(ts_idx, self.idx_class)
        # must have the correct field name and type
        self.assertEqual(ts_idx.colname, self.ts_field.name)
        self.assertEqual(ts_idx.dataType, self.ts_field.dataType)
        # validate the unit
        if self.ts_unit is None:
            self.assertFalse(ts_idx.has_unit)
        else:
            self.assertTrue(ts_idx.has_unit)
            self.assertEqual(ts_idx.unit, self.ts_unit)


@parameterized_class(
    (
        "name",
        "ts_field",
        "idx_class",
        "extra_constr_args",
        "ts_unit",
        "expected_comp_expr",
        "expected_range_expr",
    ),
    [
        (
            "simple_timestamp_index",
            StructField("event_ts", TimestampType()),
            SimpleTimestampIndex,
            None,
            StandardTimeUnits.SECONDS,
            "Column<'event_ts'>",
            "Column<'CAST(event_ts AS DOUBLE)'>",
        ),
        (
            "ordinal_double_index",
            StructField("event_ts_dbl", DoubleType()),
            OrdinalTSIndex,
            None,
            None,
            "Column<'event_ts_dbl'>",
            None,
        ),
        (
            "ordinal_int_index",
            StructField("order", IntegerType()),
            OrdinalTSIndex,
            None,
            None,
            "Column<'order'>",
            None,
        ),
        (
            "simple_date_index",
            StructField("date", DateType()),
            SimpleDateIndex,
            None,
            StandardTimeUnits.DAYS,
            "Column<'date'>",
            "Column<'datediff(date, CAST(1970-01-01 AS DATE))'>",
        ),
        (
            "parsed_timestamp_index",
            StructField(
                "ts_idx",
                StructType(
                    [
                        StructField("parsed_ts", TimestampType(), True),
                        StructField("src_str", StringType(), True),
                    ]
                ),
                True,
            ),
            ParsedTimestampIndex,
            {"parsed_ts_field": "parsed_ts", "src_str_field": "src_str"},
            StandardTimeUnits.SECONDS,
            "[Column<'ts_idx.parsed_ts'>]",
            "Column<'CAST(ts_idx.parsed_ts AS DOUBLE)'>",
        ),
        (
            "parsed_date_index",
            StructField(
                "ts_idx",
                StructType(
                    [
                        StructField("parsed_date", DateType(), True),
                        StructField("src_str", StringType(), True),
                    ]
                ),
                True,
            ),
            ParsedDateIndex,
            {"parsed_ts_field": "parsed_date", "src_str_field": "src_str"},
            StandardTimeUnits.DAYS,
            "[Column<'ts_idx.parsed_date'>]",
            "Column<'datediff(ts_idx.parsed_date, CAST(1970-01-01 AS DATE))'>",
        ),
        (
            "sub_ms_index",
            StructField(
                "ts_idx",
                StructType(
                    [
                        StructField("double_ts", DoubleType(), True),
                        StructField("parsed_ts", TimestampType(), True),
                        StructField("src_str", StringType(), True),
                    ]
                ),
                True,
            ),
            SubMicrosecondPrecisionTimestampIndex,
            {
                "double_ts_field": "double_ts",
                "secondary_parsed_ts_field": "parsed_ts",
                "src_str_field": "src_str",
            },
            StandardTimeUnits.SECONDS,
            "[Column<'ts_idx.double_ts'>]",
            "Column<'ts_idx.double_ts'>",
        ),
        (
            "subsequence_timestamp_index",
            StructField(
                "ts_idx",
                StructType(
                    [
                        StructField("event_ts", TimestampType(), True),
                        StructField("seq_num", IntegerType(), True),
                    ]
                ),
                True,
            ),
            SubsequenceTSIndex,
            {
                "ts_col": "event_ts",
                "subsequence_col": "seq_num",
            },
            StandardTimeUnits.SECONDS,
            "[Column<'ts_idx.event_ts'>, Column<'ts_idx.seq_num'>]",
            "Column<'unix_timestamp(ts_idx.event_ts, yyyy-MM-dd HH:mm:ss)'>",
        ),
        (
            "subsequence_date_index",
            StructField(
                "ts_idx",
                StructType(
                    [
                        StructField("event_date", DateType(), True),
                        StructField("seq_num", IntegerType(), True),
                    ]
                ),
                True,
            ),
            SubsequenceTSIndex,
            {
                "ts_col": "event_date",
                "subsequence_col": "seq_num",
            },
            StandardTimeUnits.DAYS,
            "[Column<'ts_idx.event_date'>, Column<'ts_idx.seq_num'>]",
            "Column<'unix_timestamp(ts_idx.event_date, yyyy-MM-dd HH:mm:ss)'>",
        ),
    ],
)
class TSIndexTests(SparkTest, TSIndexTester):
    def _create_index(self) -> TSIndex:
        if self.extra_constr_args:
            return self.idx_class(self.ts_field, **self.extra_constr_args)
        return self.idx_class(self.ts_field)

    def test_index(self):
        # create a timestamp index
        ts_idx = self._create_index()
        # test the index
        self._test_index(ts_idx)

    def test_comparable_expression(self):
        # create a timestamp index
        ts_idx = self._create_index()
        # get the expressions
        compbl_expr = ts_idx.comparableExpr()
        # validate the expression
        self.assertIsNotNone(compbl_expr)
        self.assertIsInstance(compbl_expr, (Column, List))
        self.assertEqual(repr(compbl_expr), self.expected_comp_expr)

    def test_orderby_expression(self):
        # create a timestamp index
        ts_idx = self._create_index()
        # get the expressions
        orderby_expr = ts_idx.orderByExpr()
        # validate the expression
        self.assertIsNotNone(orderby_expr)
        self.assertIsInstance(orderby_expr, (Column, List))
        self.assertEqual(repr(orderby_expr), self.expected_comp_expr)

    def test_range_expression(self):
        # create a timestamp index
        ts_idx = self._create_index()
        # get the expressions
        if isinstance(ts_idx, OrdinalTSIndex):
            self.assertRaises(NotImplementedError, ts_idx.rangeExpr)
        else:
            range_expr = ts_idx.rangeExpr()
            # validate the expression
            self.assertIsNotNone(range_expr)
            self.assertIsInstance(range_expr, Column)
            self.assertEqual(repr(range_expr), self.expected_range_expr)


@parameterized_class(
    (
        "name",
        "df_schema",
        "constr_method",
        "constr_args",
        "idx_class",
        "ts_unit",
        "expected_ts_field",
        "expected_series_ids",
        "expected_structural_cols",
        "expected_obs_cols",
        "expected_metric_cols",
    ),
    [
        (
            "simple_timestamp_index",
            StructType(
                [
                    StructField("symbol", StringType(), True),
                    StructField("event_ts", TimestampType(), True),
                    StructField("trade_pr", DoubleType(), True),
                    StructField("trade_vol", IntegerType(), True),
                ]
            ),
            "fromDFSchema",
            {"ts_col": "event_ts", "series_ids": ["symbol"]},
            SimpleTimestampIndex,
            StandardTimeUnits.SECONDS,
            "event_ts",
            ["symbol"],
            ["event_ts", "symbol"],
            ["trade_pr", "trade_vol"],
            ["trade_pr", "trade_vol"],
        ),
        (
            "simple_ts_no_series",
            StructType(
                [
                    StructField("event_ts", TimestampType(), True),
                    StructField("trade_pr", DoubleType(), True),
                    StructField("trade_vol", IntegerType(), True),
                ]
            ),
            "fromDFSchema",
            {"ts_col": "event_ts", "series_ids": []},
            SimpleTimestampIndex,
            StandardTimeUnits.SECONDS,
            "event_ts",
            [],
            ["event_ts"],
            ["trade_pr", "trade_vol"],
            ["trade_pr", "trade_vol"],
        ),
        (
            "ordinal_double_index",
            StructType(
                [
                    StructField("symbol", StringType(), True),
                    StructField("event_ts_dbl", DoubleType(), True),
                    StructField("trade_pr", DoubleType(), True),
                ]
            ),
            "fromDFSchema",
            {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
            OrdinalTSIndex,
            None,
            "event_ts_dbl",
            ["symbol"],
            ["event_ts_dbl", "symbol"],
            ["trade_pr"],
            ["trade_pr"],
        ),
        (
            "ordinal_int_index",
            StructType(
                [
                    StructField("symbol", StringType(), True),
                    StructField("order", IntegerType(), True),
                    StructField("trade_pr", DoubleType(), True),
                ]
            ),
            "fromDFSchema",
            {"ts_col": "order", "series_ids": ["symbol"]},
            OrdinalTSIndex,
            None,
            "order",
            ["symbol"],
            ["order", "symbol"],
            ["trade_pr"],
            ["trade_pr"],
        ),
        (
            "simple_date_index",
            StructType(
                [
                    StructField("symbol", StringType(), True),
                    StructField("date", DateType(), True),
                    StructField("trade_pr", DoubleType(), True),
                ]
            ),
            "fromDFSchema",
            {"ts_col": "date", "series_ids": ["symbol"]},
            SimpleDateIndex,
            StandardTimeUnits.DAYS,
            "date",
            ["symbol"],
            ["date", "symbol"],
            ["trade_pr"],
            ["trade_pr"],
        ),
        (
            "parsed_timestamp_index",
            StructType(
                [
                    StructField("symbol", StringType(), True),
                    StructField(
                        "ts_idx",
                        StructType(
                            [
                                StructField("parsed_ts", TimestampType(), True),
                                StructField("src_str", StringType(), True),
                            ]
                        ),
                        True,
                    ),
                    StructField("trade_pr", DoubleType(), True),
                    StructField("trade_vol", IntegerType(), True),
                ]
            ),
            "fromParsedTimestamp",
            {
                "ts_col": "ts_idx",
                "parsed_field": "parsed_ts",
                "src_str_field": "src_str",
                "series_ids": ["symbol"],
            },
            ParsedTimestampIndex,
            StandardTimeUnits.SECONDS,
            "ts_idx",
            ["symbol"],
            ["ts_idx", "symbol"],
            ["trade_pr", "trade_vol"],
            ["trade_pr", "trade_vol"],
        ),
        (
            "parsed_ts_no_series",
            StructType(
                [
                    StructField(
                        "ts_idx",
                        StructType(
                            [
                                StructField("parsed_ts", TimestampType(), True),
                                StructField("src_str", StringType(), True),
                            ]
                        ),
                        True,
                    ),
                    StructField("trade_pr", DoubleType(), True),
                    StructField("trade_vol", IntegerType(), True),
                ]
            ),
            "fromParsedTimestamp",
            {
                "ts_col": "ts_idx",
                "parsed_field": "parsed_ts",
                "src_str_field": "src_str",
            },
            ParsedTimestampIndex,
            StandardTimeUnits.SECONDS,
            "ts_idx",
            [],
            ["ts_idx"],
            ["trade_pr", "trade_vol"],
            ["trade_pr", "trade_vol"],
        ),
        (
            "parsed_date_index",
            StructType(
                [
                    StructField(
                        "ts_idx",
                        StructType(
                            [
                                StructField("parsed_date", DateType(), True),
                                StructField("src_str", StringType(), True),
                            ]
                        ),
                        True,
                    ),
                    StructField("symbol", StringType(), True),
                    StructField("trade_pr", DoubleType(), True),
                ]
            ),
            "fromParsedTimestamp",
            {
                "ts_col": "ts_idx",
                "parsed_field": "parsed_date",
                "src_str_field": "src_str",
                "series_ids": ["symbol"],
            },
            ParsedDateIndex,
            StandardTimeUnits.DAYS,
            "ts_idx",
            ["symbol"],
            ["ts_idx", "symbol"],
            ["trade_pr"],
            ["trade_pr"],
        ),
        (
            "sub_ms_index",
            StructType(
                [
                    StructField(
                        "ts_idx",
                        StructType(
                            [
                                StructField("double_ts", DoubleType(), True),
                                StructField("parsed_ts", TimestampType(), True),
                                StructField("src_str", StringType(), True),
                            ]
                        ),
                        True,
                    ),
                    StructField("symbol", StringType(), True),
                    StructField("trade_pr", DoubleType(), True),
                ]
            ),
            "fromParsedTimestamp",
            {
                "ts_col": "ts_idx",
                "parsed_field": "double_ts",
                "src_str_field": "src_str",
                "secondary_parsed_field": "parsed_ts",
                "series_ids": ["symbol"],
            },
            SubMicrosecondPrecisionTimestampIndex,
            StandardTimeUnits.SECONDS,
            "ts_idx",
            ["symbol"],
            ["ts_idx", "symbol"],
            ["trade_pr"],
            ["trade_pr"],
        ),
    ],
)
class TSSchemaTests(SparkTest, TSIndexTester):
    def _create_ts_schema(self) -> TSSchema:
        return getattr(TSSchema, self.constr_method)(self.df_schema, **self.constr_args)

    def setUp(self) -> None:
        super().setUp()
        self.ts_col = self.constr_args["ts_col"]
        self.ts_field = self.df_schema[self.ts_col]
        self.ts_schema = self._create_ts_schema()

    def test_schema(self):
        # make sure it's a valid TSSchema instance
        self.assertIsNotNone(self.ts_schema)
        self.assertIsInstance(self.ts_schema, TSSchema)
        # test the index
        self._test_index(self.ts_schema.ts_idx)
        # validate the index
        self.ts_schema.validate(self.df_schema)

    def test_series_ids(self):
        # test the series ids
        self.assertEqual(self.ts_schema.series_ids, self.expected_series_ids)

    def test_structural_cols(self):
        # test the structural columns
        self.assertEqual(
            set(self.ts_schema.structural_columns), set(self.expected_structural_cols)
        )

    def test_observational_cols(self):
        # test the observational columns
        self.assertEqual(
            set(self.ts_schema.find_observational_columns(self.df_schema)),
            set(self.expected_obs_cols),
        )

    def test_metric_cols(self):
        # test the metric columns
        self.assertEqual(
            set(self.ts_schema.find_metric_columns(self.df_schema)),
            set(self.expected_metric_cols),
        )

    def test_base_window(self):
        # create a TSSchema
        ts_schema = self._create_ts_schema()
        # test the base window
        bw = ts_schema.baseWindow()
        self.assertIsNotNone(bw)
        self.assertIsInstance(bw, WindowSpec)
        # test it in reverse
        bw_rev = ts_schema.baseWindow(reverse=True)
        self.assertIsNotNone(bw_rev)
        self.assertIsInstance(bw_rev, WindowSpec)

    def test_rows_window(self):
        # create a TSSchema
        ts_schema = self._create_ts_schema()
        # test the base window
        rows_win = ts_schema.rowsBetweenWindow(0, 10)
        self.assertIsNotNone(rows_win)
        self.assertIsInstance(rows_win, WindowSpec)
        # test it in reverse
        rows_win_rev = ts_schema.rowsBetweenWindow(0, 10, reverse=True)
        self.assertIsNotNone(rows_win_rev)
        self.assertIsInstance(rows_win_rev, WindowSpec)

    def test_range_window(self):
        # create a TSSchema
        ts_schema = self._create_ts_schema()
        # test the base window
        if ts_schema.ts_idx.has_unit:
            range_win = ts_schema.rangeBetweenWindow(0, 10)
            self.assertIsNotNone(range_win)
            self.assertIsInstance(range_win, WindowSpec)
        else:
            self.assertRaises(NotImplementedError, ts_schema.rangeBetweenWindow, 0, 10)
        # test it in reverse
        if ts_schema.ts_idx.has_unit:
            range_win_rev = ts_schema.rangeBetweenWindow(0, 10, reverse=True)
            self.assertIsNotNone(range_win_rev)
            self.assertIsInstance(range_win_rev, WindowSpec)
        else:
            self.assertRaises(
                NotImplementedError, ts_schema.rangeBetweenWindow, 0, 10, reverse=True
            )


class TSSchemaValidationTests(SparkTest):
    """Test validation of required parameters for parsed timestamp indices"""

    def test_parsed_timestamp_missing_src_str_field(self):
        """Test that ParsedTimestampIndex raises ValueError when src_str_field is None"""
        df_schema = StructType(
            [
                StructField("symbol", StringType(), True),
                StructField(
                    "ts_idx",
                    StructType(
                        [
                            StructField("parsed_ts", TimestampType(), True),
                            StructField("src_str", StringType(), True),
                        ]
                    ),
                    True,
                ),
                StructField("value", DoubleType(), True),
            ]
        )

        with self.assertRaises(ValueError) as context:
            TSSchema.fromParsedTimestamp(
                df_schema,
                ts_col="ts_idx",
                parsed_field="parsed_ts",
                src_str_field=None,  # This should raise an error
                series_ids=["symbol"],
            )

        self.assertIn(
            "src_str_field is required for ParsedTimestampIndex", str(context.exception)
        )

    def test_parsed_date_missing_src_str_field(self):
        """Test that ParsedDateIndex raises ValueError when src_str_field is None"""
        df_schema = StructType(
            [
                StructField("symbol", StringType(), True),
                StructField(
                    "ts_idx",
                    StructType(
                        [
                            StructField("parsed_date", DateType(), True),
                            StructField("src_str", StringType(), True),
                        ]
                    ),
                    True,
                ),
                StructField("value", DoubleType(), True),
            ]
        )

        with self.assertRaises(ValueError) as context:
            TSSchema.fromParsedTimestamp(
                df_schema,
                ts_col="ts_idx",
                parsed_field="parsed_date",
                src_str_field=None,  # This should raise an error
                series_ids=["symbol"],
            )

        self.assertIn(
            "src_str_field is required for ParsedDateIndex", str(context.exception)
        )

    def test_submicrosecond_missing_src_str_field(self):
        """Test that SubMicrosecondPrecisionTimestampIndex raises ValueError when src_str_field is None"""
        df_schema = StructType(
            [
                StructField("symbol", StringType(), True),
                StructField(
                    "ts_idx",
                    StructType(
                        [
                            StructField("double_ts", DoubleType(), True),
                            StructField("parsed_ts", TimestampType(), True),
                            StructField("src_str", StringType(), True),
                        ]
                    ),
                    True,
                ),
                StructField("value", DoubleType(), True),
            ]
        )

        with self.assertRaises(ValueError) as context:
            TSSchema.fromParsedTimestamp(
                df_schema,
                ts_col="ts_idx",
                parsed_field="double_ts",
                src_str_field=None,  # This should raise an error
                secondary_parsed_field="parsed_ts",
                series_ids=["symbol"],
            )

        self.assertIn(
            "src_str_field is required for SubMicrosecondPrecisionTimestampIndex",
            str(context.exception),
        )

    def test_submicrosecond_missing_secondary_parsed_field(self):
        """Test that SubMicrosecondPrecisionTimestampIndex raises ValueError when secondary_parsed_field is None"""
        df_schema = StructType(
            [
                StructField("symbol", StringType(), True),
                StructField(
                    "ts_idx",
                    StructType(
                        [
                            StructField("double_ts", DoubleType(), True),
                            StructField("parsed_ts", TimestampType(), True),
                            StructField("src_str", StringType(), True),
                        ]
                    ),
                    True,
                ),
                StructField("value", DoubleType(), True),
            ]
        )

        with self.assertRaises(ValueError) as context:
            TSSchema.fromParsedTimestamp(
                df_schema,
                ts_col="ts_idx",
                parsed_field="double_ts",
                src_str_field="src_str",
                secondary_parsed_field=None,  # This should raise an error
                series_ids=["symbol"],
            )

        self.assertIn(
            "secondary_parsed_field is required for SubMicrosecondPrecisionTimestampIndex",
            str(context.exception),
        )

    def test_valid_parsed_timestamp_creation(self):
        """Test that valid parameters create ParsedTimestampIndex successfully"""
        df_schema = StructType(
            [
                StructField("symbol", StringType(), True),
                StructField(
                    "ts_idx",
                    StructType(
                        [
                            StructField("parsed_ts", TimestampType(), True),
                            StructField("src_str", StringType(), True),
                        ]
                    ),
                    True,
                ),
                StructField("value", DoubleType(), True),
            ]
        )

        # This should work without errors
        ts_schema = TSSchema.fromParsedTimestamp(
            df_schema,
            ts_col="ts_idx",
            parsed_field="parsed_ts",
            src_str_field="src_str",
            series_ids=["symbol"],
        )

        self.assertIsInstance(ts_schema, TSSchema)
        self.assertIsInstance(ts_schema.ts_idx, ParsedTimestampIndex)

    def test_valid_submicrosecond_creation(self):
        """Test that valid parameters create SubMicrosecondPrecisionTimestampIndex successfully"""
        df_schema = StructType(
            [
                StructField("symbol", StringType(), True),
                StructField(
                    "ts_idx",
                    StructType(
                        [
                            StructField("double_ts", DoubleType(), True),
                            StructField("parsed_ts", TimestampType(), True),
                            StructField("src_str", StringType(), True),
                        ]
                    ),
                    True,
                ),
                StructField("value", DoubleType(), True),
            ]
        )

        # This should work without errors
        ts_schema = TSSchema.fromParsedTimestamp(
            df_schema,
            ts_col="ts_idx",
            parsed_field="double_ts",
            src_str_field="src_str",
            secondary_parsed_field="parsed_ts",
            series_ids=["symbol"],
        )

        self.assertIsInstance(ts_schema, TSSchema)
        self.assertIsInstance(ts_schema.ts_idx, SubMicrosecondPrecisionTimestampIndex)
