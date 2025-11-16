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

    def test_repartition_by_series(self):
        """Test repartitionBySeries method"""
        tsdf = self.get_data_as_tsdf("init")

        result = tsdf.repartitionBySeries(numPartitions=2)

        self.assertEqual(result.df.count(), 4)
        self.assertEqual(result.df.rdd.getNumPartitions(), 2)
        self.assertEqual(result.series_ids, tsdf.series_ids)

    def test_repartition_by_series_default_partitions(self):
        """Test repartitionBySeries with default number of partitions"""
        tsdf = self.get_data_as_tsdf("init")

        original_partitions = tsdf.df.rdd.getNumPartitions()
        result = tsdf.repartitionBySeries()

        self.assertEqual(result.df.count(), 2)
        self.assertEqual(result.df.rdd.getNumPartitions(), original_partitions)

    def test_repartition_by_time(self):
        """Test repartitionByTime method"""
        tsdf = self.get_data_as_tsdf("init")

        result = tsdf.repartitionByTime(numPartitions=2)

        self.assertEqual(result.df.count(), 3)
        self.assertEqual(result.df.rdd.getNumPartitions(), 2)

    def test_repartition_by_time_default_partitions(self):
        """Test repartitionByTime with default number of partitions"""
        tsdf = self.get_data_as_tsdf("init")

        original_partitions = tsdf.df.rdd.getNumPartitions()
        result = tsdf.repartitionByTime()

        self.assertEqual(result.df.count(), 2)
        self.assertEqual(result.df.rdd.getNumPartitions(), original_partitions)

    def test_with_column_renamed_series_id(self):
        """Test withColumnRenamed when renaming a series_id column"""
        tsdf = self.get_data_as_tsdf("init")

        result = tsdf.withColumnRenamed("ticker", "symbol")

        self.assertEqual(result.df.count(), 2)
        self.assertIn("symbol", result.df.columns)
        self.assertNotIn("ticker", result.df.columns)
        self.assertEqual(result.series_ids, ["symbol"])

    def test_with_column_renamed_ts_col(self):
        """Test withColumnRenamed when renaming the timestamp column"""
        tsdf = self.get_data_as_tsdf("init")

        result = tsdf.withColumnRenamed("event_time", "timestamp")

        self.assertEqual(result.df.count(), 2)
        self.assertIn("timestamp", result.df.columns)
        self.assertNotIn("event_time", result.df.columns)
        self.assertEqual(result.ts_col, "timestamp")

    def test_with_column_type_changed(self):
        """Test withColumnTypeChanged method"""
        tsdf = self.get_data_as_tsdf("init")

        result = tsdf.withColumnTypeChanged("value", "int")

        self.assertEqual(result.df.count(), 2)
        value_type = dict(result.df.dtypes)["value"]
        self.assertIn("int", value_type.lower())



if __name__ == "__main__":
    unittest.main()
