import unittest
from abc import ABC, abstractmethod
from parameterized import parameterized, parameterized_class

from pyspark.sql import Column, WindowSpec
from pyspark.sql import functions as sfn
from pyspark.sql.types import (
    StructField,
    StructType,
    StringType,
    TimestampType,
    DoubleType,
    IntegerType,
    DateType,
    NumericType,
)

from tempo.tsschema import (
    TSIndex,
    SimpleTimestampIndex,
    OrdinalTSIndex,
    SimpleDateIndex,
    StandardTimeUnits,
    ParsedTimestampIndex,
    ParsedDateIndex,
    SubMicrosecondPrecisionTimestampIndex,
    TSSchema,
)
from tests.base import SparkTest


class TSIndexTests(SparkTest, ABC):
    @abstractmethod
    def _create_index(self) -> TSIndex:
        pass

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
                StructType([
                    StructField("parsed_ts", TimestampType(), True),
                    StructField("src_str", StringType(), True),
                ]),
                True,
            ),
            ParsedTimestampIndex,
            {"parsed_ts_field": "parsed_ts", "src_str_field": "src_str"},
            StandardTimeUnits.SECONDS,
            "Column<'ts_idx.parsed_ts'>",
            "Column<'CAST(ts_idx.parsed_ts AS DOUBLE)'>",
        ),
        (
            "parsed_date_index",
            StructField(
                "ts_idx",
                StructType([
                    StructField("parsed_date", DateType(), True),
                    StructField("src_str", StringType(), True),
                ]),
                True,
            ),
            ParsedDateIndex,
            {"parsed_ts_field": "parsed_date", "src_str_field": "src_str"},
            StandardTimeUnits.DAYS,
            "Column<'ts_idx.parsed_date'>",
            "Column<'datediff(ts_idx.parsed_date, CAST(1970-01-01 AS DATE))'>",
        ),
        (
            "sub_ms_index",
            StructField(
                "ts_idx",
                StructType([
                    StructField("double_ts", DoubleType(), True),
                    StructField("parsed_ts", TimestampType(), True),
                    StructField("src_str", StringType(), True),
                ]),
                True,
            ),
            SubMicrosecondPrecisionTimestampIndex,
            {
                "double_ts_field": "double_ts",
                "secondary_parsed_ts_field": "parsed_ts",
                "src_str_field": "src_str",
            },
            StandardTimeUnits.NANOSECONDS,
            "Column<'ts_idx.double_ts'>",
            "Column<'ts_idx.double_ts'>",
        ),
    ],
)
class SimpleTSIndexTests(TSIndexTests):
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
        self.assertIsInstance(compbl_expr, Column)
        self.assertEqual(repr(compbl_expr), self.expected_comp_expr)

    def test_orderby_expression(self):
        # create a timestamp index
        ts_idx = self._create_index()
        # get the expressions
        orderby_expr = ts_idx.orderByExpr()
        # validate the expression
        self.assertIsNotNone(orderby_expr)
        self.assertIsInstance(orderby_expr, Column)
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
        "ts_field",
        "constr_args",
        "idx_class",
        "ts_unit",
        "expected_comp_expr",
        "expected_range_expr",
    ),
    [
        (
            "parsed_timestamp_index",
            StructField(
                "ts_idx",
                StructType([
                    StructField("parsed_ts", TimestampType(), True),
                    StructField("src_str", StringType(), True),
                ]),
                True,
            ),
            {"parsed_ts_field": "parsed_ts", "src_str_field": "src_str"},
            ParsedTimestampIndex,
            StandardTimeUnits.SECONDS,
            "Column<'ts_idx.parsed_ts'>",
            "Column<'CAST(ts_idx.parsed_ts AS DOUBLE)'>",
        ),
        (
            "parsed_date_index",
            StructField(
                "ts_idx",
                StructType([
                    StructField("parsed_date", DateType(), True),
                    StructField("src_str", StringType(), True),
                ]),
                True,
            ),
            {"parsed_ts_field": "parsed_date", "src_str_field": "src_str"},
            ParsedDateIndex,
            StandardTimeUnits.DAYS,
            "Column<'ts_idx.parsed_date'>",
            "Column<'datediff(ts_idx.parsed_date, CAST(1970-01-01 AS DATE))'>",
        ),
        (
            "sub_ms_index",
            StructField(
                "ts_idx",
                StructType([
                    StructField("double_ts", DoubleType(), True),
                    StructField("parsed_ts", TimestampType(), True),
                    StructField("src_str", StringType(), True),
                ]),
                True,
            ),
            {
                "double_ts_field": "double_ts",
                "secondary_parsed_ts_field": "parsed_ts",
                "src_str_field": "src_str",
            },
            SubMicrosecondPrecisionTimestampIndex,
            StandardTimeUnits.NANOSECONDS,
            "Column<'ts_idx.double_ts'>",
            "Column<'ts_idx.double_ts'>",
        ),
    ],
)
class ParsedTSIndexTests(TSIndexTests):
    def _create_index(self):
        return self.idx_class(ts_struct=self.ts_field, **self.constr_args)

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
        self.assertIsInstance(compbl_expr, Column)
        self.assertEqual(repr(compbl_expr), self.expected_comp_expr)

    def test_orderby_expression(self):
        # create a timestamp index
        ts_idx = self._create_index()
        # get the expressions
        orderby_expr = ts_idx.orderByExpr()
        # validate the expression
        self.assertIsNotNone(orderby_expr)
        self.assertIsInstance(orderby_expr, Column)
        self.assertEqual(repr(orderby_expr), self.expected_comp_expr)

    def test_range_expression(self):
        # create a timestamp index
        ts_idx = self._create_index()
        # get the expressions
        range_expr = ts_idx.rangeExpr()
        # validate the expression
        self.assertIsNotNone(range_expr)
        self.assertIsInstance(range_expr, Column)
        self.assertEqual(repr(range_expr), self.expected_range_expr)


class TSSchemaTests(TSIndexTests, ABC):
    @abstractmethod
    def _create_ts_schema(self) -> TSSchema:
        pass


@parameterized_class(
    ("name", "df_schema", "ts_col", "series_ids", "idx_class", "ts_unit"),
    [
        (
            "simple_timestamp_index",
            StructType([
                StructField("symbol", StringType(), True),
                StructField("event_ts", TimestampType(), True),
                StructField("trade_pr", DoubleType(), True),
                StructField("trade_vol", IntegerType(), True),
            ]),
            "event_ts",
            ["symbol"],
            SimpleTimestampIndex,
            StandardTimeUnits.SECONDS,
        ),
        (
            "simple_ts_no_series",
            StructType([
                StructField("event_ts", TimestampType(), True),
                StructField("trade_pr", DoubleType(), True),
                StructField("trade_vol", IntegerType(), True),
            ]),
            "event_ts",
            [],
            SimpleTimestampIndex,
            StandardTimeUnits.SECONDS,
        ),
        (
            "ordinal_double_index",
            StructType([
                StructField("symbol", StringType(), True),
                StructField("event_ts_dbl", DoubleType(), True),
                StructField("trade_pr", DoubleType(), True),
            ]),
            "event_ts_dbl",
            ["symbol"],
            OrdinalTSIndex,
            None,
        ),
        (
            "ordinal_int_index",
            StructType([
                StructField("symbol", StringType(), True),
                StructField("order", IntegerType(), True),
                StructField("trade_pr", DoubleType(), True),
            ]),
            "order",
            ["symbol"],
            OrdinalTSIndex,
            None,
        ),
        (
            "simple_date_index",
            StructType([
                StructField("symbol", StringType(), True),
                StructField("date", DateType(), True),
                StructField("trade_pr", DoubleType(), True),
            ]),
            "date",
            ["symbol"],
            SimpleDateIndex,
            StandardTimeUnits.DAYS,
        ),
    ],
)
class SimpleIndexTSSchemaTests(TSSchemaTests):
    def _create_index(self) -> TSIndex:
        pass

    def _create_ts_schema(self) -> TSSchema:
        return TSSchema.fromDFSchema(self.df_schema, self.ts_col, self.series_ids)

    def setUp(self) -> None:
        super().setUp()
        self.ts_field = self.df_schema[self.ts_col]

    def test_schema(self):
        print(self.id())
        # create a TSSchema
        ts_schema = self._create_ts_schema()
        # make sure it's a valid TSSchema instance
        self.assertIsNotNone(ts_schema)
        self.assertIsInstance(ts_schema, TSSchema)
        # test the index
        self._test_index(ts_schema.ts_idx)
        # test the series ids
        self.assertEqual(ts_schema.series_ids, self.series_ids)
        # validate the index
        ts_schema.validate(self.df_schema)

    def test_structural_cols(self):
        # create a TSSchema
        ts_schema = self._create_ts_schema()
        # test the structural columns
        struct_cols = [self.ts_col] + self.series_ids
        self.assertEqual(set(ts_schema.structural_columns), set(struct_cols))

    def test_observational_cols(self):
        # create a TSSchema
        ts_schema = self._create_ts_schema()
        # test the structural columns
        struct_cols = [self.ts_col] + self.series_ids
        obs_cols = set(self.df_schema.fieldNames()) - set(struct_cols)
        self.assertEqual(
            ts_schema.find_observational_columns(self.df_schema), list(obs_cols)
        )

    def test_metric_cols(self):
        # create a TSSchema
        ts_schema = self._create_ts_schema()
        # test the metric columns
        struct_cols = [self.ts_col] + self.series_ids
        obs_cols = set(self.df_schema.fieldNames()) - set(struct_cols)
        metric_cols = {
            mc
            for mc in obs_cols
            if isinstance(self.df_schema[mc].dataType, NumericType)
        }
        self.assertEqual(
            set(ts_schema.find_metric_columns(self.df_schema)), metric_cols
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


class ParsedIndexTSSchemaTests(TSSchemaTests):
    def _create_index(self) -> TSIndex:
        pass

    def _create_ts_schema(self) -> TSSchema:
        pass
