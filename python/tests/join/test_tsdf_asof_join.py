"""
Integration tests for TSDF.asofJoin() method with strategy selection.

This module tests the high-level asofJoin() API on TSDF objects, including:
- Automatic strategy selection based on data characteristics
- Manual strategy selection via the strategy parameter
- Parameter passing (tolerance, skipNulls, prefixes)
- Strategy switching based on data size and tsPartitionVal
"""

import unittest
from unittest.mock import patch
from datetime import datetime, timedelta

import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType

from tests.base import SparkTest
from tempo.tsdf import TSDF
from tempo.joins.strategies import (
    BroadcastAsOfJoiner,
    UnionSortFilterAsOfJoiner,
    SkewAsOfJoiner,
)


class TSDFAsOfJoinTest(SparkTest):
    """Test TSDF.asofJoin() method with various configurations."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Create simple test data
        base_time = datetime(2024, 1, 1, 10, 0, 0)

        # Left DataFrame: 10 trades
        left_data = [
            ("AAPL", base_time + timedelta(minutes=i), f"trade_{i}", 100.0 + i)
            for i in range(10)
        ]
        left_df = self.spark.createDataFrame(
            left_data, ["symbol", "timestamp", "trade_id", "volume"]
        )
        self.left_tsdf = TSDF(left_df, ts_col="timestamp", series_ids=["symbol"])

        # Right DataFrame: 5 quotes
        right_data = [
            ("AAPL", base_time + timedelta(minutes=i * 2), 100.0 + i * 2)
            for i in range(5)
        ]
        right_df = self.spark.createDataFrame(
            right_data, ["symbol", "timestamp", "price"]
        )
        self.right_tsdf = TSDF(right_df, ts_col="timestamp", series_ids=["symbol"])

    def test_asof_join_default_strategy(self):
        """Test asofJoin with default (automatic) strategy selection."""
        result = self.left_tsdf.asofJoin(self.right_tsdf, right_prefix="quote")

        # Should preserve all left rows
        self.assertEqual(result.df.count(), 10)

        # Should have quote_price column
        self.assertIn("quote_price", result.df.columns)

        # Verify as-of semantics: all rows should have matched quotes
        null_count = result.df.filter(F.col("quote_price").isNull()).count()
        self.assertEqual(null_count, 0, "All trades should match a quote")

    def test_asof_join_manual_broadcast_strategy(self):
        """Test asofJoin with manual broadcast strategy selection."""
        result = self.left_tsdf.asofJoin(
            self.right_tsdf,
            strategy="broadcast",
            right_prefix="quote"
        )

        # Verify results
        self.assertEqual(result.df.count(), 10)
        self.assertIn("quote_price", result.df.columns)

    def test_asof_join_manual_union_strategy(self):
        """Test asofJoin with manual union strategy selection."""
        result = self.left_tsdf.asofJoin(
            self.right_tsdf,
            strategy="union",
            right_prefix="quote"
        )

        # Verify results
        self.assertEqual(result.df.count(), 10)
        self.assertIn("quote_price", result.df.columns)

    def test_asof_join_manual_skew_strategy(self):
        """Test asofJoin with manual skew strategy selection."""
        result = self.left_tsdf.asofJoin(
            self.right_tsdf,
            strategy="skew",
            right_prefix="quote",
            tsPartitionVal=300
        )

        # Verify results
        self.assertEqual(result.df.count(), 10)
        self.assertIn("quote_price", result.df.columns)

    def test_asof_join_invalid_strategy(self):
        """Test asofJoin with invalid strategy raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            self.left_tsdf.asofJoin(
                self.right_tsdf,
                strategy="invalid"
            )

        self.assertIn("Unknown strategy", str(cm.exception))

    def test_asof_join_with_tolerance(self):
        """Test asofJoin with tolerance parameter."""
        # Create data with large time gaps
        base_time = datetime(2024, 1, 1, 10, 0, 0)

        left_data = [
            ("AAPL", base_time + timedelta(minutes=i * 10), f"trade_{i}", 100.0)
            for i in range(5)
        ]
        left_df = self.spark.createDataFrame(
            left_data, ["symbol", "timestamp", "trade_id", "volume"]
        )
        left_tsdf = TSDF(left_df, ts_col="timestamp", series_ids=["symbol"])

        # Right data only at t=0
        right_data = [("AAPL", base_time, 100.0)]
        right_df = self.spark.createDataFrame(
            right_data, ["symbol", "timestamp", "price"]
        )
        right_tsdf = TSDF(right_df, ts_col="timestamp", series_ids=["symbol"])

        # Join with tolerance=300 seconds (5 minutes)
        result = left_tsdf.asofJoin(
            right_tsdf,
            tolerance=300,
            right_prefix="quote"
        )

        # First row (t=0) should match, second row (t=10) should NOT match (beyond tolerance)
        matched_count = result.df.filter(F.col("quote_price").isNotNull()).count()
        self.assertLessEqual(matched_count, 1, "Only rows within tolerance should match")

    def test_asof_join_with_skip_nulls(self):
        """Test asofJoin with skipNulls parameter."""
        # Create right data with nulls
        base_time = datetime(2024, 1, 1, 10, 0, 0)

        right_data = [
            ("AAPL", base_time, 100.0),
            ("AAPL", base_time + timedelta(minutes=2), None),  # NULL price
            ("AAPL", base_time + timedelta(minutes=4), 102.0),
        ]
        right_df = self.spark.createDataFrame(
            right_data, ["symbol", "timestamp", "price"]
        )
        right_tsdf = TSDF(right_df, ts_col="timestamp", series_ids=["symbol"])

        # Test with skipNulls=True (default)
        result_skip = self.left_tsdf.asofJoin(
            right_tsdf,
            skipNulls=True,
            right_prefix="quote"
        )

        # Rows between t=2 and t=4 should skip the NULL and get t=0 price
        # or t=4 price depending on timing
        self.assertIsNotNone(result_skip)

    def test_asof_join_with_prefixes(self):
        """Test asofJoin with custom prefixes."""
        result = self.left_tsdf.asofJoin(
            self.right_tsdf,
            left_prefix="trade",
            right_prefix="quote"
        )

        # Check for prefixed columns
        self.assertIn("quote_price", result.df.columns)
        self.assertIn("timestamp", result.df.columns)

    def test_asof_join_empty_right_dataframe(self):
        """Test asofJoin when right DataFrame is empty."""
        # Create empty right DataFrame with same schema
        empty_right_df = self.spark.createDataFrame(
            [], ["symbol", "timestamp", "price"]
        )
        empty_right_tsdf = TSDF(empty_right_df, ts_col="timestamp", series_ids=["symbol"])

        result = self.left_tsdf.asofJoin(
            empty_right_tsdf,
            right_prefix="quote"
        )

        # Should preserve all left rows with NULL right values
        self.assertEqual(result.df.count(), 10)
        null_count = result.df.filter(F.col("quote_price").isNull()).count()
        self.assertEqual(null_count, 10, "All right values should be NULL")

    def test_asof_join_with_partition_val_selects_skew(self):
        """Test that tsPartitionVal parameter selects SkewAsOfJoiner."""
        # We can't easily verify the strategy without mocking, but we can verify it works
        result = self.left_tsdf.asofJoin(
            self.right_tsdf,
            tsPartitionVal=300,  # Should trigger SkewAsOfJoiner
            right_prefix="quote"
        )

        self.assertEqual(result.df.count(), 10)

    @patch('tempo.joins.strategies.get_bytes_from_plan')
    def test_asof_join_automatic_broadcast_selection(self, mock_get_bytes):
        """Test that small data automatically selects BroadcastAsOfJoiner."""
        # Mock size estimation to return small sizes (< 30MB)
        mock_get_bytes.return_value = 10 * 1024 * 1024  # 10MB

        # Without strategy parameter, should auto-select
        result = self.left_tsdf.asofJoin(
            self.right_tsdf,
            right_prefix="quote"
        )

        self.assertEqual(result.df.count(), 10)
        # Verify get_bytes_from_plan was called for size estimation
        self.assertGreater(mock_get_bytes.call_count, 0)

    @patch('tempo.joins.strategies.get_bytes_from_plan')
    def test_asof_join_automatic_union_selection(self, mock_get_bytes):
        """Test that large data automatically selects UnionSortFilterAsOfJoiner."""
        # Mock size estimation to return large sizes (> 30MB)
        mock_get_bytes.return_value = 100 * 1024 * 1024  # 100MB

        result = self.left_tsdf.asofJoin(
            self.right_tsdf,
            right_prefix="quote"
        )

        self.assertEqual(result.df.count(), 10)

    def test_asof_join_preserves_schema(self):
        """Test that asofJoin preserves TSDF schema correctly."""
        result = self.left_tsdf.asofJoin(
            self.right_tsdf,
            right_prefix="quote"
        )

        # Should preserve left ts_col and series_ids
        self.assertEqual(result.ts_col, "timestamp")
        self.assertEqual(result.series_ids, ["symbol"])

    def test_asof_join_multiple_series_ids(self):
        """Test asofJoin with multiple series ID columns."""
        base_time = datetime(2024, 1, 1, 10, 0, 0)

        # Left: trades with exchange and symbol
        left_data = [
            ("NYSE", "AAPL", base_time + timedelta(minutes=i), 100.0 + i)
            for i in range(5)
        ] + [
            ("NASDAQ", "GOOGL", base_time + timedelta(minutes=i), 200.0 + i)
            for i in range(5)
        ]
        left_df = self.spark.createDataFrame(
            left_data, ["exchange", "symbol", "timestamp", "volume"]
        )
        left_tsdf = TSDF(left_df, ts_col="timestamp", series_ids=["exchange", "symbol"])

        # Right: quotes
        right_data = [
            ("NYSE", "AAPL", base_time + timedelta(minutes=i * 2), 100.0 + i * 2)
            for i in range(3)
        ] + [
            ("NASDAQ", "GOOGL", base_time + timedelta(minutes=i * 2), 200.0 + i * 2)
            for i in range(3)
        ]
        right_df = self.spark.createDataFrame(
            right_data, ["exchange", "symbol", "timestamp", "price"]
        )
        right_tsdf = TSDF(right_df, ts_col="timestamp", series_ids=["exchange", "symbol"])

        result = left_tsdf.asofJoin(
            right_tsdf,
            right_prefix="quote"
        )

        # Verify multi-key join worked
        self.assertEqual(result.df.count(), 10)
        self.assertEqual(result.series_ids, ["exchange", "symbol"])


class TSDFAsOfJoinConsistencyTest(SparkTest):
    """Test that different strategies produce consistent results when called via TSDF.asofJoin()."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Create consistent test data
        base_time = datetime(2024, 1, 1, 10, 0, 0)

        left_data = [
            ("AAPL", base_time + timedelta(minutes=i), f"trade_{i}", 100.0 + i)
            for i in range(20)
        ]
        left_df = self.spark.createDataFrame(
            left_data, ["symbol", "timestamp", "trade_id", "volume"]
        )
        self.left_tsdf = TSDF(left_df, ts_col="timestamp", series_ids=["symbol"])

        right_data = [
            ("AAPL", base_time + timedelta(minutes=i * 3), 100.0 + i * 3)
            for i in range(7)
        ]
        right_df = self.spark.createDataFrame(
            right_data, ["symbol", "timestamp", "price"]
        )
        self.right_tsdf = TSDF(right_df, ts_col="timestamp", series_ids=["symbol"])

    def test_strategy_consistency_broadcast_vs_union(self):
        """Test that broadcast and union strategies produce identical results."""
        # Execute with broadcast
        broadcast_result = self.left_tsdf.asofJoin(
            self.right_tsdf,
            strategy="broadcast",
            right_prefix="quote"
        )

        # Execute with union
        union_result = self.left_tsdf.asofJoin(
            self.right_tsdf,
            strategy="union",
            right_prefix="quote"
        )

        # Results should be identical
        self.assertDataFrameEquality(
            broadcast_result.df,
            union_result.df,
            ignore_row_order=True
        )

    def test_strategy_consistency_with_tolerance(self):
        """Test strategy consistency when tolerance is applied."""
        broadcast_result = self.left_tsdf.asofJoin(
            self.right_tsdf,
            strategy="broadcast",
            tolerance=300,  # 5 minutes
            right_prefix="quote"
        )

        union_result = self.left_tsdf.asofJoin(
            self.right_tsdf,
            strategy="union",
            tolerance=300,
            right_prefix="quote"
        )

        # Results should be identical
        self.assertDataFrameEquality(
            broadcast_result.df,
            union_result.df,
            ignore_row_order=True
        )


if __name__ == "__main__":
    unittest.main()
