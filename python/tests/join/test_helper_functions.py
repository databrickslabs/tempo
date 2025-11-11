"""
Unit tests for helper functions in tempo.joins.strategies module.

Tests for:
- get_spark_plan()
- get_bytes_from_plan()
- Other utility functions
"""

import unittest
from unittest.mock import Mock, patch, MagicMock

from pyspark.sql import SparkSession, DataFrame

from tempo.joins.strategies import (
    get_spark_plan,
    get_bytes_from_plan,
)


class TestGetSparkPlan(unittest.TestCase):
    """Test get_spark_plan helper function."""

    @patch('tempo.joins.strategies.SparkSession')
    def test_get_spark_plan_success(self, mock_spark_class):
        """Test successful Spark plan extraction."""
        # Mock SparkSession and DataFrame
        mock_spark = Mock(spec=SparkSession)
        mock_df = Mock(spec=DataFrame)

        # Mock the SQL execution and result
        mock_spark.sql.return_value.collect.return_value = [
            Mock(_1="Statistics(sizeInBytes=1000 B)")
        ]

        # Mock createOrReplaceTempView
        mock_df.createOrReplaceTempView = Mock()

        result = get_spark_plan(mock_df, mock_spark)

        # Verify result contains the plan
        self.assertIsInstance(result, str)
        self.assertIn("Statistics", result)

    @patch('tempo.joins.strategies.SparkSession')
    def test_get_spark_plan_with_temp_view(self, mock_spark_class):
        """Test that get_spark_plan creates temp view with unique name."""
        mock_spark = Mock(spec=SparkSession)
        mock_df = Mock(spec=DataFrame)

        # Mock the SQL execution
        mock_spark.sql.return_value.collect.return_value = [
            Mock(_1="Statistics(sizeInBytes=500 MiB)")
        ]

        get_spark_plan(mock_df, mock_spark)

        # Verify temp view was created
        mock_df.createOrReplaceTempView.assert_called_once()

        # Verify SQL was executed with EXPLAIN
        self.assertTrue(mock_spark.sql.called)
        sql_arg = mock_spark.sql.call_args[0][0]
        self.assertIn("EXPLAIN", sql_arg.upper())


class TestGetBytesFromPlan(unittest.TestCase):
    """Test get_bytes_from_plan helper function."""

    @patch('tempo.joins.strategies.get_spark_plan')
    def test_parse_gib_units(self, mock_get_plan):
        """Test parsing size in GiB units."""
        mock_spark = Mock()
        mock_df = Mock()

        # Mock plan with GiB
        mock_get_plan.return_value = "Statistics(sizeInBytes=2.5 GiB)"

        result = get_bytes_from_plan(mock_df, mock_spark)

        # 2.5 GiB = 2.5 * 1024 * 1024 * 1024 bytes
        expected = 2.5 * 1024 * 1024 * 1024
        self.assertEqual(result, expected)

    @patch('tempo.joins.strategies.get_spark_plan')
    def test_parse_mib_units(self, mock_get_plan):
        """Test parsing size in MiB units."""
        mock_spark = Mock()
        mock_df = Mock()

        mock_get_plan.return_value = "Statistics(sizeInBytes=128.0 MiB)"

        result = get_bytes_from_plan(mock_df, mock_spark)

        # 128 MiB = 128 * 1024 * 1024 bytes
        expected = 128.0 * 1024 * 1024
        self.assertEqual(result, expected)

    @patch('tempo.joins.strategies.get_spark_plan')
    def test_parse_kib_units(self, mock_get_plan):
        """Test parsing size in KiB units."""
        mock_spark = Mock()
        mock_df = Mock()

        mock_get_plan.return_value = "Statistics(sizeInBytes=512.0 KiB)"

        result = get_bytes_from_plan(mock_df, mock_spark)

        # 512 KiB = 512 * 1024 bytes
        expected = 512.0 * 1024
        self.assertEqual(result, expected)

    @patch('tempo.joins.strategies.get_spark_plan')
    def test_parse_bytes_units(self, mock_get_plan):
        """Test parsing size in bytes (no unit suffix)."""
        mock_spark = Mock()
        mock_df = Mock()

        mock_get_plan.return_value = "Statistics(sizeInBytes=1024.0 B)"

        result = get_bytes_from_plan(mock_df, mock_spark)

        # 1024 bytes
        expected = 1024.0
        self.assertEqual(result, expected)

    @patch('tempo.joins.strategies.get_spark_plan')
    def test_parse_no_size_returns_inf(self, mock_get_plan):
        """Test that missing sizeInBytes returns infinity."""
        mock_spark = Mock()
        mock_df = Mock()

        # Plan without sizeInBytes
        mock_get_plan.return_value = "Some other plan information"

        result = get_bytes_from_plan(mock_df, mock_spark)

        # Should return infinity to avoid broadcast
        self.assertEqual(result, float('inf'))

    @patch('tempo.joins.strategies.get_spark_plan')
    def test_parse_error_returns_inf(self, mock_get_plan):
        """Test that parsing errors return infinity."""
        mock_spark = Mock()
        mock_df = Mock()

        # Raise exception during plan extraction
        mock_get_plan.side_effect = Exception("Spark plan error")

        result = get_bytes_from_plan(mock_df, mock_spark)

        # Should return infinity on error
        self.assertEqual(result, float('inf'))

    @patch('tempo.joins.strategies.get_spark_plan')
    def test_parse_decimal_sizes(self, mock_get_plan):
        """Test parsing sizes with decimal points."""
        mock_spark = Mock()
        mock_df = Mock()

        mock_get_plan.return_value = "Statistics(sizeInBytes=3.14159 GiB)"

        result = get_bytes_from_plan(mock_df, mock_spark)

        # 3.14159 GiB
        expected = 3.14159 * 1024 * 1024 * 1024
        self.assertAlmostEqual(result, expected, places=2)

    @patch('tempo.joins.strategies.get_spark_plan')
    def test_parse_size_with_parenthesis(self, mock_get_plan):
        """Test parsing size when plan ends with parenthesis."""
        mock_spark = Mock()
        mock_df = Mock()

        # Some plans end with ) instead of just the unit
        mock_get_plan.return_value = "Statistics(sizeInBytes=64.0 MiB)"

        result = get_bytes_from_plan(mock_df, mock_spark)

        expected = 64.0 * 1024 * 1024
        self.assertEqual(result, expected)

    @patch('tempo.joins.strategies.get_spark_plan')
    def test_parse_zero_size(self, mock_get_plan):
        """Test parsing zero size."""
        mock_spark = Mock()
        mock_df = Mock()

        mock_get_plan.return_value = "Statistics(sizeInBytes=0.0 B)"

        result = get_bytes_from_plan(mock_df, mock_spark)

        self.assertEqual(result, 0.0)

    @patch('tempo.joins.strategies.get_spark_plan')
    def test_parse_very_large_size(self, mock_get_plan):
        """Test parsing very large size (TiB-scale)."""
        mock_spark = Mock()
        mock_df = Mock()

        # 1 TiB (expressed as GiB)
        mock_get_plan.return_value = "Statistics(sizeInBytes=1024.0 GiB)"

        result = get_bytes_from_plan(mock_df, mock_spark)

        # 1 TiB = 1024 GiB
        expected = 1024.0 * 1024 * 1024 * 1024
        self.assertEqual(result, expected)

    @patch('tempo.joins.strategies.get_spark_plan')
    def test_parse_integer_size(self, mock_get_plan):
        """Test parsing size as integer (no decimal point)."""
        mock_spark = Mock()
        mock_df = Mock()

        mock_get_plan.return_value = "Statistics(sizeInBytes=256 MiB)"

        result = get_bytes_from_plan(mock_df, mock_spark)

        expected = 256.0 * 1024 * 1024
        self.assertEqual(result, expected)


class TestHelperFunctionIntegration(unittest.TestCase):
    """Integration tests for helper functions."""

    @patch('tempo.joins.strategies.get_bytes_from_plan')
    def test_size_estimation_in_strategy_selection(self, mock_get_bytes):
        """Test that size estimation is used in strategy selection."""
        from tempo.joins.strategies import choose_as_of_join_strategy
        from tempo.tsdf import TSDF

        # Mock size estimation
        mock_get_bytes.side_effect = [10 * 1024 * 1024, 20 * 1024 * 1024]  # 10MB, 20MB

        # Create mock TSDFs
        mock_spark = Mock()
        left_tsdf = Mock(spec=TSDF)
        left_tsdf.df = Mock()
        right_tsdf = Mock(spec=TSDF)
        right_tsdf.df = Mock()

        # Call strategy selection
        strategy = choose_as_of_join_strategy(
            left_tsdf,
            right_tsdf,
            mock_spark
        )

        # Should select BroadcastAsOfJoiner for small data
        from tempo.joins.strategies import BroadcastAsOfJoiner
        self.assertIsInstance(strategy, BroadcastAsOfJoiner)

        # Verify size estimation was called
        self.assertEqual(mock_get_bytes.call_count, 2)

    @patch('tempo.joins.strategies.get_bytes_from_plan')
    def test_size_estimation_failure_falls_back(self, mock_get_bytes):
        """Test that strategy selection handles size estimation failure."""
        from tempo.joins.strategies import choose_as_of_join_strategy
        from tempo.tsdf import TSDF

        # Mock size estimation to fail
        mock_get_bytes.side_effect = Exception("Cannot estimate size")

        mock_spark = Mock()
        left_tsdf = Mock(spec=TSDF)
        left_tsdf.df = Mock()
        right_tsdf = Mock(spec=TSDF)
        right_tsdf.df = Mock()

        # Should fall back to UnionSortFilterAsOfJoiner
        strategy = choose_as_of_join_strategy(
            left_tsdf,
            right_tsdf,
            mock_spark
        )

        from tempo.joins.strategies import UnionSortFilterAsOfJoiner
        self.assertIsInstance(strategy, UnionSortFilterAsOfJoiner)


if __name__ == "__main__":
    unittest.main()
