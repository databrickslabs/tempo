import unittest
from parameterized import parameterized_class

from pyspark.sql import Column
from pyspark.sql import functions as sfn
from pyspark.sql.types import (
    StructField,
    StructType,
    StringType,
    TimestampType,
    DoubleType,
    IntegerType,
    DateType,
)

from tempo.tsschema import (
    TSIndex,
    SimpleTimestampIndex,
    OrdinalTSIndex,
    SimpleDateIndex,
    StandardTimeUnits,
    ParsedTimestampIndex,
    ParsedDateIndex
)
from tests.base import SparkTest


@parameterized_class(
    (
        "name",
        "ts_field",
        "idx_class",
        "ts_unit",
        "expected_comp_expr",
        "expected_range_expr",
    ),
    [
        (
            "simple_timestamp_index",
            StructField("event_ts", TimestampType()),
            SimpleTimestampIndex,
            StandardTimeUnits.SECONDS,
            "Column<'event_ts'>",
            "Column<'CAST(event_ts AS DOUBLE)'>",
        ),
        (
            "ordinal_double_index",
            StructField("event_ts_dbl", DoubleType()),
            OrdinalTSIndex,
            None,
            "Column<'event_ts_dbl'>",
            None,
        ),
        (
            "ordinal_int_index",
            StructField("order", IntegerType()),
            OrdinalTSIndex,
            None,
            "Column<'order'>",
            None,
        ),
        (
            "simple_date_index",
            StructField("date", DateType()),
            SimpleDateIndex,
            StandardTimeUnits.DAYS,
            "Column<'date'>",
            "Column<'datediff(date, CAST(1970-01-01 AS DATE))'>",
        ),
    ],
)
class SimpleTSIndexTests(SparkTest):
    def test_constructor(self):
        # create a timestamp index
        ts_idx = self.idx_class(self.ts_field)
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

    def test_comparable_expression(self):
        # create a timestamp index
        ts_idx: TSIndex = self.idx_class(self.ts_field)
        # get the expressions
        compbl_expr = ts_idx.comparableExpr()
        # validate the expression
        self.assertIsNotNone(compbl_expr)
        self.assertIsInstance(compbl_expr, Column)
        self.assertEqual(repr(compbl_expr), self.expected_comp_expr)

    def test_orderby_expression(self):
        # create a timestamp index
        ts_idx: TSIndex = self.idx_class(self.ts_field)
        # get the expressions
        orderby_expr = ts_idx.orderByExpr()
        # validate the expression
        self.assertIsNotNone(orderby_expr)
        self.assertIsInstance(orderby_expr, Column)
        self.assertEqual(repr(orderby_expr), self.expected_comp_expr)

    def test_range_expression(self):
        # create a timestamp index
        ts_idx = self.idx_class(self.ts_field)
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
        "expected_range_expr"
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
            {"parsed_ts_col": "parsed_ts", "src_str_col": "src_str"},
            ParsedTimestampIndex,
            StandardTimeUnits.SECONDS,
            "Column<'ts_idx.parsed_ts'>",
            "Column<'CAST(ts_idx.parsed_ts AS DOUBLE)'>"
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
            {"parsed_ts_col": "parsed_date", "src_str_col": "src_str"},
            ParsedDateIndex,
            StandardTimeUnits.DAYS,
            "Column<'ts_idx.parsed_date'>",
            "Column<'datediff(ts_idx.parsed_date, CAST(1970-01-01 AS DATE))'>"
        ),
        (
            "sub_ms_index",
            StructField(
                "ts_idx",
                StructType([
                    StructField("double_ts", TimestampType(), True),
                    StructField("parsed_ts", TimestampType(), True),
                    StructField("src_str", StringType(), True),
                ]),
                True,
            ),
            {"double_ts_col": "double_ts", "parsed_ts_col": "parsed_ts", "src_str_col": "src_str"},
            ParsedTimestampIndex,
            StandardTimeUnits.SECONDS,
            "Column<'ts_idx.parsed_ts'>",
            "Column<'CAST(ts_idx.parsed_ts AS DOUBLE)'>"
        ),
    ])
class ParsedTSIndexTests(SparkTest):
    def test_constructor(self):
        # create a timestamp index
        ts_idx = self.idx_class(ts_struct=self.ts_field, **self.constr_args)
        # must be a valid TSIndex object
        self.assertIsNotNone(ts_idx)
        self.assertIsInstance(ts_idx, self.idx_class)
        # must have the correct field name and type
        self.assertEqual(ts_idx.colname, self.ts_field.name)
        self.assertEqual(ts_idx.dataType, self.ts_field.dataType)
        # validate the unit
        self.assertTrue(ts_idx.has_unit)
        self.assertEqual(ts_idx.unit, self.ts_unit)

    def test_comparable_expression(self):
        # create a timestamp index
        ts_idx = self.idx_class(ts_struct=self.ts_field, **self.constr_args)
        # get the expressions
        compbl_expr = ts_idx.comparableExpr()
        # validate the expression
        self.assertIsNotNone(compbl_expr)
        self.assertIsInstance(compbl_expr, Column)
        self.assertEqual(repr(compbl_expr), self.expected_comp_expr)

    def test_orderby_expression(self):
        # create a timestamp index
        ts_idx = self.idx_class(ts_struct=self.ts_field, **self.constr_args)
        # get the expressions
        orderby_expr = ts_idx.orderByExpr()
        # validate the expression
        self.assertIsNotNone(orderby_expr)
        self.assertIsInstance(orderby_expr, Column)
        self.assertEqual(repr(orderby_expr), self.expected_comp_expr)

    def test_range_expression(self):
        # create a timestamp index
        ts_idx = self.idx_class(ts_struct=self.ts_field, **self.constr_args)
        # get the expressions
        range_expr = ts_idx.rangeExpr()
        # validate the expression
        self.assertIsNotNone(range_expr)
        self.assertIsInstance(range_expr, Column)
        self.assertEqual(repr(range_expr), self.expected_range_expr)


# class TSSchemaTests(SparkTest):
#     def test_simple_tsIndex(self):
#         schema_str = "event_ts timestamp, symbol string, trade_pr double"
#         schema = _parse_datatype_string(schema_str)
#         ts_idx = TSSchema.fromDFSchema(schema, "event_ts", ["symbol"])
#
#         print(ts_idx)
