"""
Tests for TSDF stats and union methods to improve code coverage.
Targets describe, metricSummary, union, unionByName methods.
"""

import unittest

from tempo.tsdf import TSDF
from tests.base import SparkTest


class TSDFStatsTests(SparkTest):
    """Test TSDF statistics methods for better coverage."""

    def test_describe_default(self):
        """Test describe() method with default parameters"""
        tsdf = self.get_data_as_tsdf("init")

        result = tsdf.describe()

        self.assertIsNotNone(result)
        self.assertIn("summary", result.columns)

    def test_describe_specific_cols(self):
        """Test describe() method with specific columns"""
        tsdf = self.get_data_as_tsdf("init")

        result = tsdf.describe("price")

        self.assertIsNotNone(result)
        self.assertIn("summary", result.columns)
        self.assertIn("price", result.columns)


class TSDFUnionTests(SparkTest):
    """Test TSDF union methods for better coverage."""

    def test_union(self):
        """Test union() method"""
        tsdf1 = self.get_data_as_tsdf("init")
        tsdf2 = self.get_data_as_tsdf("other")

        result = tsdf1.union(tsdf2)

        self.assertEqual(result.df.count(), 4)
        self.assertEqual(result.ts_col, tsdf1.ts_col)
        self.assertEqual(result.series_ids, tsdf1.series_ids)

    def test_union_by_name(self):
        """Test unionByName() method"""
        tsdf1 = self.get_data_as_tsdf("init")
        tsdf2 = self.get_data_as_tsdf("other")

        result = tsdf1.unionByName(tsdf2)

        self.assertEqual(result.df.count(), 4)
        self.assertEqual(result.ts_col, tsdf1.ts_col)

    def test_union_by_name_with_missing_columns(self):
        """Test unionByName() with allowMissingColumns=True"""
        tsdf1 = self.get_data_as_tsdf("init")
        tsdf2 = self.get_data_as_tsdf("missing_col")

        result = tsdf1.unionByName(tsdf2, allowMissingColumns=True)

        self.assertEqual(result.df.count(), 3)
        self.assertIn("price", result.df.columns)
        self.assertIn("volume", result.df.columns)


if __name__ == "__main__":
    unittest.main()
