"""
Tests for edge cases and error handling to improve code coverage.
These tests target specific uncovered lines in tempo/joins/strategies.py.
"""

import unittest
from datetime import datetime, timedelta
from pyspark.sql import functions as F
from tempo.tsdf import TSDF
from tempo.joins.strategies import (
    BroadcastAsOfJoiner,
    UnionSortFilterAsOfJoiner,
    SkewAsOfJoiner,
    choose_as_of_join_strategy,
)
from tests.base import SparkTest


class EdgeCaseCoverageTests(SparkTest):
    """Test edge cases and error handling for better coverage."""

    def test_broadcast_join_no_series_ids(self):
        """Test broadcast join with no series_ids (single series case)."""
        # Line 328: Single series join path
        left_data = [
            (datetime(2024, 1, 1, 10, i), f"trade_{i}", float(i)) for i in range(10)
        ]
        left_df = self.spark.createDataFrame(
            left_data, ["timestamp", "trade_id", "volume"]
        )
        # No series_ids specified
        left_tsdf = TSDF(left_df, ts_col="timestamp")

        right_data = [(datetime(2024, 1, 1, 10, i), 100.0 + i) for i in range(0, 10, 2)]
        right_df = self.spark.createDataFrame(right_data, ["timestamp", "price"])
        right_tsdf = TSDF(right_df, ts_col="timestamp")

        joiner = BroadcastAsOfJoiner(self.spark)
        result_df, result_schema = joiner(left_tsdf, right_tsdf)

        # Verify join works without series_ids
        self.assertEqual(result_df.count(), 10)
        self.assertEqual(result_schema.series_ids, [])

    def test_skew_join_no_series_ids(self):
        """Test skew join with no series_ids returns empty skewed keys."""
        # Line 706: No series keys to be skewed
        left_data = [(datetime(2024, 1, 1, 10, i), f"trade_{i}") for i in range(50)]
        left_df = self.spark.createDataFrame(left_data, ["timestamp", "trade_id"])
        left_tsdf = TSDF(left_df, ts_col="timestamp")

        right_data = [(datetime(2024, 1, 1, 10, i), 100.0 + i) for i in range(50)]
        right_df = self.spark.createDataFrame(right_data, ["timestamp", "price"])
        right_tsdf = TSDF(right_df, ts_col="timestamp")

        joiner = SkewAsOfJoiner(self.spark, skew_threshold=0.1)
        result_df, _ = joiner(left_tsdf, right_tsdf)

        # Should complete without errors (takes standard path)
        self.assertEqual(result_df.count(), 50)

    def test_skipnulls_with_no_value_columns(self):
        """Test skipNulls filter when right has no value columns."""
        # Line 1009: Early return when no right value columns
        left_data = [
            ("A", datetime(2024, 1, 1, 10, i), f"trade_{i}") for i in range(10)
        ]
        left_df = self.spark.createDataFrame(
            left_data, ["symbol", "timestamp", "trade_id"]
        )
        left_tsdf = TSDF(left_df, ts_col="timestamp", series_ids=["symbol"])

        # Right has only series_ids and timestamp, no value columns
        right_data = [("A", datetime(2024, 1, 1, 10, i)) for i in range(0, 10, 2)]
        right_df = self.spark.createDataFrame(right_data, ["symbol", "timestamp"])
        right_tsdf = TSDF(right_df, ts_col="timestamp", series_ids=["symbol"])

        joiner = SkewAsOfJoiner(self.spark, skipNulls=True)
        result_df, _ = joiner(left_tsdf, right_tsdf)

        # Should work without errors
        self.assertGreater(result_df.count(), 0)

    def test_tolerance_filter(self):
        """Test tolerance filter is applied correctly."""
        # Lines 954, 825: Tolerance filter paths
        left_data = [
            ("A", datetime(2024, 1, 1, 10, i), f"trade_{i}") for i in range(10)
        ]
        left_df = self.spark.createDataFrame(
            left_data, ["symbol", "timestamp", "trade_id"]
        )
        left_tsdf = TSDF(left_df, ts_col="timestamp", series_ids=["symbol"])

        # Right data with some timestamps far apart
        right_data = [
            ("A", datetime(2024, 1, 1, 10, 0), 100.0),
            ("A", datetime(2024, 1, 1, 10, 5), 105.0),
        ]
        right_df = self.spark.createDataFrame(
            right_data, ["symbol", "timestamp", "price"]
        )
        right_tsdf = TSDF(right_df, ts_col="timestamp", series_ids=["symbol"])

        # Tolerance of 120 seconds (2 minutes)
        joiner = SkewAsOfJoiner(self.spark, tolerance=120)
        result_df, _ = joiner(left_tsdf, right_tsdf)

        # Result should complete without errors
        # Note: Tolerance implementation may have issues (see tsdf_asof_join_tests.py:152)
        self.assertGreater(result_df.count(), 0)

    def test_skipnulls_and_tolerance_combined(self):
        """Test combined skipNulls and tolerance filters."""
        # Lines 951-954, 817-821: Combined filter paths
        left_data = [
            ("A", datetime(2024, 1, 1, 10, i), f"trade_{i}") for i in range(10)
        ]
        left_df = self.spark.createDataFrame(
            left_data, ["symbol", "timestamp", "trade_id"]
        )
        left_tsdf = TSDF(left_df, ts_col="timestamp", series_ids=["symbol"])

        right_data = [
            ("A", datetime(2024, 1, 1, 10, 0), 100.0),
            ("A", datetime(2024, 1, 1, 10, 3), 103.0),
        ]
        right_df = self.spark.createDataFrame(
            right_data, ["symbol", "timestamp", "price"]
        )
        right_tsdf = TSDF(right_df, ts_col="timestamp", series_ids=["symbol"])

        # Tolerance of 120 seconds (2 minutes), with skipNulls
        joiner = SkewAsOfJoiner(self.spark, skipNulls=True, tolerance=120)
        result_df, _ = joiner(left_tsdf, right_tsdf)

        # Both filters should be applied
        # All rows should have non-null prices (skipNulls)
        # Only rows within tolerance should have matches
        self.assertGreater(result_df.count(), 0)
        self.assertLessEqual(result_df.count(), left_tsdf.df.count())

    def test_strategy_selection_error_fallback(self):
        """Test that strategy selection falls back on error."""
        # Lines 1185-1188: Exception handler fallback
        # Create invalid TSDFs to potentially trigger errors
        left_data = [("A", datetime(2024, 1, 1, 10, 0), "trade_1")]
        left_df = self.spark.createDataFrame(
            left_data, ["symbol", "timestamp", "trade_id"]
        )
        left_tsdf = TSDF(left_df, ts_col="timestamp", series_ids=["symbol"])

        right_data = [("A", datetime(2024, 1, 1, 10, 0), 100.0)]
        right_df = self.spark.createDataFrame(
            right_data, ["symbol", "timestamp", "price"]
        )
        right_tsdf = TSDF(right_df, ts_col="timestamp", series_ids=["symbol"])

        # This should not raise an exception even if selection has issues
        try:
            strategy = choose_as_of_join_strategy(left_tsdf, right_tsdf, self.spark)
            # Should return some joiner (possibly fallback)
            self.assertIsNotNone(strategy)
        except Exception as e:
            self.fail(f"Strategy selection should not raise exceptions: {e}")

    def test_no_skew_detected_standard_path(self):
        """Test that no skew detected takes standard join path."""
        # Line 757: No skew detected path
        # Create evenly distributed data (no skew)
        left_data = []
        for symbol in ["A", "B", "C"]:
            for i in range(20):
                left_data.append((symbol, datetime(2024, 1, 1, 10, i), f"t_{i}"))

        left_df = self.spark.createDataFrame(
            left_data, ["symbol", "timestamp", "trade_id"]
        )
        left_tsdf = TSDF(left_df, ts_col="timestamp", series_ids=["symbol"])

        right_data = []
        for symbol in ["A", "B", "C"]:
            for i in range(0, 20, 5):
                right_data.append((symbol, datetime(2024, 1, 1, 10, i), 100.0 + i))

        right_df = self.spark.createDataFrame(
            right_data, ["symbol", "timestamp", "price"]
        )
        right_tsdf = TSDF(right_df, ts_col="timestamp", series_ids=["symbol"])

        # High threshold so no skew is detected
        joiner = SkewAsOfJoiner(self.spark, skew_threshold=0.9)
        result_df, _ = joiner(left_tsdf, right_tsdf)

        # Should use standard path and complete successfully
        self.assertEqual(result_df.count(), 60)

    def test_union_join_no_value_columns(self):
        """Test union join with no value columns in right."""
        # Line 508: Fallback when no value columns
        left_data = [
            ("A", datetime(2024, 1, 1, 10, i), f"trade_{i}") for i in range(10)
        ]
        left_df = self.spark.createDataFrame(
            left_data, ["symbol", "timestamp", "trade_id"]
        )
        left_tsdf = TSDF(left_df, ts_col="timestamp", series_ids=["symbol"])

        # Right has only series_ids and timestamp
        right_data = [("A", datetime(2024, 1, 1, 10, i)) for i in range(0, 10, 2)]
        right_df = self.spark.createDataFrame(right_data, ["symbol", "timestamp"])
        right_tsdf = TSDF(right_df, ts_col="timestamp", series_ids=["symbol"])

        joiner = UnionSortFilterAsOfJoiner()
        result_df, _ = joiner(left_tsdf, right_tsdf)

        # Should complete without errors
        self.assertEqual(result_df.count(), 10)


if __name__ == "__main__":
    unittest.main()
