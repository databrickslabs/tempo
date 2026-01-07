"""
Test suite for error handling in as-of join strategies.

This test suite focuses on covering error handling paths and edge cases
to improve code coverage in tempo/joins/strategies.py.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import math

from tempo.joins.strategies import (
    get_bytes_from_plan,
    choose_as_of_join_strategy,
    _detectSignificantSkew,
    UnionSortFilterAsOfJoiner,
)
from tempo.tsdf import TSDF


class TestGetBytesFromPlanErrorHandling(unittest.TestCase):
    """Test error handling in get_bytes_from_plan function."""

    @patch("tempo.joins.strategies.get_spark_plan")
    def test_get_bytes_from_plan_no_size_in_bytes(self, mock_get_plan):
        """Test when Spark plan doesn't contain sizeInBytes."""
        # Mock plan without sizeInBytes
        mock_get_plan.return_value = "Some plan text without size information"

        mock_df = Mock()
        mock_spark = Mock()

        result = get_bytes_from_plan(mock_df, mock_spark)

        # Should return infinity when sizeInBytes not found
        self.assertEqual(result, float("inf"))

    @patch("tempo.joins.strategies.get_spark_plan")
    def test_get_bytes_from_plan_exception_during_parsing(self, mock_get_plan):
        """Test exception handling during plan parsing."""
        # Mock get_spark_plan to raise an exception
        mock_get_plan.side_effect = Exception("Failed to get plan")

        mock_df = Mock()
        mock_spark = Mock()

        result = get_bytes_from_plan(mock_df, mock_spark)

        # Should return infinity on exception
        self.assertEqual(result, float("inf"))

    @patch("tempo.joins.strategies.get_spark_plan")
    def test_get_bytes_from_plan_kib_units(self, mock_get_plan):
        """Test handling of KiB units in plan."""
        # Mock plan with KiB units
        mock_get_plan.return_value = "Plan text sizeInBytes=100.5 KiB more text"

        mock_df = Mock()
        mock_spark = Mock()

        result = get_bytes_from_plan(mock_df, mock_spark)

        # Should convert KiB to bytes: 100.5 * 1024 = 102912
        self.assertEqual(result, 100.5 * 1024)

    @patch("tempo.joins.strategies.get_spark_plan")
    def test_get_bytes_from_plan_bytes_units(self, mock_get_plan):
        """Test handling of plain bytes (no prefix)."""
        # Mock plan with no unit prefix (just "B")
        mock_get_plan.return_value = "Plan text sizeInBytes=1000 B more text"

        mock_df = Mock()
        mock_spark = Mock()

        result = get_bytes_from_plan(mock_df, mock_spark)

        # Should return size as-is
        self.assertEqual(result, 1000)


class TestChooseStrategyErrorHandling(unittest.TestCase):
    """Test error handling in choose_as_of_join_strategy function."""

    def test_choose_strategy_outer_exception(self):
        """Test outer exception handling that falls back to default strategy."""
        # Create mock TSDFs that will cause an exception
        left_tsdf = Mock(spec=TSDF)
        left_tsdf.df = None  # This should cause issues
        left_tsdf.series_ids = None

        right_tsdf = Mock(spec=TSDF)
        right_tsdf.df = None
        right_tsdf.series_ids = None

        mock_spark = Mock()

        # Should not raise exception, should return UnionSortFilterAsOfJoiner
        result = choose_as_of_join_strategy(left_tsdf, right_tsdf, mock_spark)

        self.assertIsInstance(result, UnionSortFilterAsOfJoiner)
        self.assertEqual(result.left_prefix, "left")
        self.assertEqual(result.right_prefix, "right")

    @patch("tempo.joins.strategies.get_bytes_from_plan")
    def test_choose_strategy_size_estimation_exception(self, mock_get_bytes):
        """Test exception during size estimation falls through to default."""
        # Mock get_bytes_from_plan to raise an exception
        mock_get_bytes.side_effect = Exception("Size estimation failed")

        left_tsdf = Mock(spec=TSDF)
        left_tsdf.df = Mock()
        left_tsdf.series_ids = ["id"]

        right_tsdf = Mock(spec=TSDF)
        right_tsdf.df = Mock()
        right_tsdf.series_ids = ["id"]

        mock_spark = Mock()

        # Should fall through to default strategy
        result = choose_as_of_join_strategy(left_tsdf, right_tsdf, mock_spark)

        self.assertIsInstance(result, UnionSortFilterAsOfJoiner)


class TestDetectSignificantSkewErrorHandling(unittest.TestCase):
    """Test error handling in _detectSignificantSkew function."""

    def test_detect_skew_no_series_ids(self):
        """Test when TSDF has no series IDs."""
        left_tsdf = Mock(spec=TSDF)
        left_tsdf.series_ids = []  # No series IDs

        right_tsdf = Mock(spec=TSDF)
        right_tsdf.series_ids = []

        result = _detectSignificantSkew(left_tsdf, right_tsdf)

        # Should return False when no series IDs
        self.assertFalse(result)

    def test_detect_skew_small_dataset(self):
        """Test with very small dataset (< 100 rows)."""
        mock_df = Mock()
        mock_df.count.return_value = 50  # Small dataset

        left_tsdf = Mock(spec=TSDF)
        left_tsdf.series_ids = ["id"]
        left_tsdf.df = mock_df

        right_tsdf = Mock(spec=TSDF)
        right_tsdf.series_ids = ["id"]

        result = _detectSignificantSkew(left_tsdf, right_tsdf)

        # Should return False for datasets smaller than 100 rows
        self.assertFalse(result)

    def test_detect_skew_exception_during_detection(self):
        """Test exception handling during skew detection."""
        mock_df = Mock()
        mock_df.count.return_value = 10000
        # Make sample() raise an exception
        mock_df.sample.side_effect = Exception("Sampling failed")

        left_tsdf = Mock(spec=TSDF)
        left_tsdf.series_ids = ["id"]
        left_tsdf.df = mock_df

        right_tsdf = Mock(spec=TSDF)
        right_tsdf.series_ids = ["id"]

        result = _detectSignificantSkew(left_tsdf, right_tsdf)

        # Should return False on exception
        self.assertFalse(result)

    def test_detect_skew_null_statistics(self):
        """Test when stddev/avg return None."""
        # Create a mock chain for df.sample().groupBy().count().select().collect()
        mock_collect_result = [{"std": None, "avg": None}]

        mock_select = Mock()
        mock_select.collect.return_value = mock_collect_result

        mock_count = Mock()
        mock_count.select.return_value = mock_select

        mock_group = Mock()
        mock_group.count.return_value = mock_count

        mock_sample = Mock()
        mock_sample.groupBy.return_value = mock_group

        mock_df = Mock()
        mock_df.count.return_value = 10000
        mock_df.sample.return_value = mock_sample

        left_tsdf = Mock(spec=TSDF)
        left_tsdf.series_ids = ["id"]
        left_tsdf.df = mock_df

        right_tsdf = Mock(spec=TSDF)
        right_tsdf.series_ids = ["id"]

        result = _detectSignificantSkew(left_tsdf, right_tsdf)

        # Should return False when stats are None
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
