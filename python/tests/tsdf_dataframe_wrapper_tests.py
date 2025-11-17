"""
Tests for TSDF DataFrame wrapper methods to improve code coverage.
Targets select, withColumn, where methods.
"""

import unittest

from pyspark.sql import functions as F

from tempo.tsdf import TSDF
from tests.base import SparkTest


class TSDFDataFrameWrapperTests(SparkTest):
    """Test TSDF DataFrame wrapper methods for better coverage."""

    def test_select_single_column(self):
        """Test select() method with single column"""
        tsdf = self.get_data_as_tsdf("init")

        result = tsdf.select("symbol", "price")

        self.assertIn("symbol", result.df.columns)
        self.assertIn("price", result.df.columns)
        self.assertNotIn("volume", result.df.columns)
        self.assertEqual(result.ts_col, tsdf.ts_col)

    def test_select_all_columns(self):
        """Test select() method with * wildcard"""
        tsdf = self.get_data_as_tsdf("init")

        result = tsdf.select("*")

        self.assertEqual(len(result.df.columns), len(tsdf.df.columns))

    def test_with_column_add_new(self):
        """Test withColumn() method adding a new column"""
        tsdf = self.get_data_as_tsdf("init")

        result = tsdf.withColumn("price_doubled", F.col("price") * 2)

        self.assertIn("price_doubled", result.df.columns)
        self.assertEqual(result.df.count(), tsdf.df.count())
        collected = result.df.collect()
        self.assertEqual(collected[0]["price_doubled"], collected[0]["price"] * 2)

    def test_with_column_replace_existing(self):
        """Test withColumn() method replacing an existing column"""
        tsdf = self.get_data_as_tsdf("init")

        result = tsdf.withColumn("price", F.col("price") + 10)

        self.assertIn("price", result.df.columns)
        self.assertEqual(result.df.count(), tsdf.df.count())

    def test_where_string_condition(self):
        """Test where() method with string condition"""
        tsdf = self.get_data_as_tsdf("init")

        result = tsdf.where("price > 100")

        self.assertLess(result.df.count(), tsdf.df.count())
        for row in result.df.collect():
            self.assertGreater(row["price"], 100)

    def test_where_column_condition(self):
        """Test where() method with Column condition"""
        tsdf = self.get_data_as_tsdf("init")

        result = tsdf.where(F.col("symbol") == "A")

        self.assertLess(result.df.count(), tsdf.df.count())
        for row in result.df.collect():
            self.assertEqual(row["symbol"], "A")

    def test_where_multiple_conditions(self):
        """Test where() method with multiple conditions"""
        tsdf = self.get_data_as_tsdf("init")

        result = tsdf.where((F.col("symbol") == "A") & (F.col("price") > 100))

        for row in result.df.collect():
            self.assertEqual(row["symbol"], "A")
            self.assertGreater(row["price"], 100)


if __name__ == "__main__":
    unittest.main()
