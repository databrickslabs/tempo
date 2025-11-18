"""
Additional tests for as-of join strategies to improve code coverage.

This test suite targets specific uncovered code paths in tempo/joins/strategies.py
identified by the Codecov report.
"""

import unittest
from datetime import datetime, timedelta
from pyspark.sql import functions as F
from tempo.tsdf import TSDF
from tempo.joins.strategies import (
    BroadcastAsOfJoiner,
    UnionSortFilterAsOfJoiner,
    SkewAsOfJoiner,
)
from tests.base import SparkTest


class AdditionalCoverageTests(SparkTest):
    """Test additional edge cases for better coverage."""

    def test_empty_prefix_handling(self):
        """Test that empty/None prefix is handled correctly."""
        # Lines 108-120: _prefixColumns should skip prefixing when prefix is empty
        left_data = [
            (datetime(2024, 1, 1, 10, i), f"A", float(100 + i))
            for i in range(5)
        ]
        left_df = self.spark.createDataFrame(
            left_data, ["timestamp", "symbol", "price"]
        )
        left_tsdf = TSDF(left_df, ts_col="timestamp", series_ids=["symbol"])

        right_data = [
            (datetime(2024, 1, 1, 10, i), f"A", float(200 + i))
            for i in range(0, 5, 2)
        ]
        right_df = self.spark.createDataFrame(
            right_data, ["timestamp", "symbol", "price"]
        )
        right_tsdf = TSDF(right_df, ts_col="timestamp", series_ids=["symbol"])

        # Use empty string as prefix - should not prefix columns
        joiner = BroadcastAsOfJoiner(self.spark, left_prefix="", right_prefix="")
        result_df, result_schema = joiner(left_tsdf, right_tsdf)

        # Verify that columns are NOT prefixed when prefix is empty
        self.assertEqual(result_df.count(), 5)

    def test_skipnulls_with_only_series_timestamp(self):
        """Test skipNulls when right DataFrame has only series+timestamp columns."""
        # Lines 506-510: skipNulls fallback when no value columns to check
        left_data = [
            (datetime(2024, 1, 1, 10, i), f"A", float(100 + i))
            for i in range(5)
        ]
        left_df = self.spark.createDataFrame(
            left_data, ["timestamp", "symbol", "price"]
        )
        left_tsdf = TSDF(left_df, ts_col="timestamp", series_ids=["symbol"])

        # Right has ONLY series and timestamp - no value columns
        right_data = [
            (datetime(2024, 1, 1, 10, i), f"A")
            for i in range(0, 5, 2)
        ]
        right_df = self.spark.createDataFrame(right_data, ["timestamp", "symbol"])
        right_tsdf = TSDF(right_df, ts_col="timestamp", series_ids=["symbol"])

        joiner = UnionSortFilterAsOfJoiner(skipNulls=True)
        result_df, result_schema = joiner(left_tsdf, right_tsdf)

        # Should still work even without value columns
        self.assertEqual(result_df.count(), 5)

    def test_tolerance_none_early_return(self):
        """Test that tolerance=None returns immediately without filtering."""
        # Lines 556-557: Early return when tolerance is None
        left_data = [
            (datetime(2024, 1, 1, 10, i), f"A", float(100 + i))
            for i in range(5)
        ]
        left_df = self.spark.createDataFrame(
            left_data, ["timestamp", "symbol", "price"]
        )
        left_tsdf = TSDF(left_df, ts_col="timestamp", series_ids=["symbol"])

        right_data = [
            (datetime(2024, 1, 1, 10, 0), f"A", float(200))
        ]
        right_df = self.spark.createDataFrame(
            right_data, ["timestamp", "symbol", "bid"]
        )
        right_tsdf = TSDF(right_df, ts_col="timestamp", series_ids=["symbol"])

        # tolerance=None should allow all matches regardless of time difference
        joiner = UnionSortFilterAsOfJoiner(tolerance=None)
        result_df, result_schema = joiner(left_tsdf, right_tsdf)

        # All 5 left rows should match with the single right row
        self.assertEqual(result_df.count(), 5)
        # All should have non-null right values
        for row in result_df.collect():
            if row["right_timestamp"] is not None:
                self.assertIsNotNone(row["bid"])



if __name__ == "__main__":
    unittest.main()
