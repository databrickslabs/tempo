"""
Tests for basic TSDF methods to improve code coverage.
Targets simple, high-value methods that are currently uncovered.
"""

import unittest
from datetime import datetime
from tempo.tsdf import TSDF
from tests.base import SparkTest


class TSDFBasicMethodsTests(SparkTest):
    """Test basic TSDF methods for better coverage."""

    def test_repr(self):
        """Test TSDF.__repr__ method."""
        # Line 145: __repr__ method
        data = [
            ("A", datetime(2024, 1, 1, 10, 0), 100.0),
            ("A", datetime(2024, 1, 1, 10, 1), 101.0),
        ]
        df = self.spark.createDataFrame(data, ["symbol", "timestamp", "price"])
        tsdf = TSDF(df, ts_col="timestamp", series_ids=["symbol"])

        # Test that repr returns a string with expected content
        repr_str = repr(tsdf)
        self.assertIsInstance(repr_str, str)
        self.assertIn("TSDF", repr_str)
        self.assertIn("df=", repr_str)
        self.assertIn("ts_schema=", repr_str)

    def test_eq_same_tsdf(self):
        """Test TSDF.__eq__ with identical TSDFs."""
        # Lines 147-150: __eq__ method
        data = [
            ("A", datetime(2024, 1, 1, 10, 0), 100.0),
            ("A", datetime(2024, 1, 1, 10, 1), 101.0),
        ]
        df = self.spark.createDataFrame(data, ["symbol", "timestamp", "price"])
        tsdf1 = TSDF(df, ts_col="timestamp", series_ids=["symbol"])
        tsdf2 = TSDF(df, ts_col="timestamp", series_ids=["symbol"])

        # Same schema and DataFrame should be equal
        self.assertEqual(tsdf1, tsdf2)

    def test_eq_different_type(self):
        """Test TSDF.__eq__ with non-TSDF object."""
        # Line 148-149: __eq__ with different type
        data = [("A", datetime(2024, 1, 1, 10, 0), 100.0)]
        df = self.spark.createDataFrame(data, ["symbol", "timestamp", "price"])
        tsdf = TSDF(df, ts_col="timestamp", series_ids=["symbol"])

        # Should not equal non-TSDF objects
        self.assertNotEqual(tsdf, "not a tsdf")
        self.assertNotEqual(tsdf, 123)
        self.assertNotEqual(tsdf, None)

    def test_eq_different_schema(self):
        """Test TSDF.__eq__ with different schemas."""
        # Line 150: schema comparison
        data1 = [("A", datetime(2024, 1, 1, 10, 0), 100.0)]
        df1 = self.spark.createDataFrame(data1, ["symbol", "timestamp", "price"])
        tsdf1 = TSDF(df1, ts_col="timestamp", series_ids=["symbol"])

        data2 = [("A", datetime(2024, 1, 1, 10, 0), 100.0)]
        df2 = self.spark.createDataFrame(data2, ["symbol", "timestamp", "price"])
        # Different series_ids
        tsdf2 = TSDF(df2, ts_col="timestamp", series_ids=[])

        self.assertNotEqual(tsdf1, tsdf2)



if __name__ == "__main__":
    unittest.main()
