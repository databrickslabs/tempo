"""
Test suite for as-of join strategies.

This test suite verifies that the new strategy pattern implementation
works correctly and maintains backward compatibility.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import pyspark.sql.functions as sfn
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType

from tempo.joins.strategies import (
    AsOfJoiner,
    BroadcastAsOfJoiner,
    UnionSortFilterAsOfJoiner,
    SkewAsOfJoiner,
    choose_as_of_join_strategy,
    get_bytes_from_plan,
    _DEFAULT_BROADCAST_BYTES_THRESHOLD,
)
from tempo.tsdf import TSDF
from tempo.tsschema import TSSchema, TSIndex


class TestAsOfJoinerBase(unittest.TestCase):
    """Test the abstract base class and common functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.left_prefix = "left"
        self.right_prefix = "right"

    def test_init_default_prefixes(self):
        """Test default prefix initialization."""
        joiner = UnionSortFilterAsOfJoiner()
        self.assertEqual(joiner.left_prefix, "left")
        self.assertEqual(joiner.right_prefix, "right")

    def test_init_custom_prefixes(self):
        """Test custom prefix initialization."""
        joiner = UnionSortFilterAsOfJoiner(
            left_prefix="l",
            right_prefix="r"
        )
        self.assertEqual(joiner.left_prefix, "l")
        self.assertEqual(joiner.right_prefix, "r")

    def test_common_series_ids(self):
        """Test commonSeriesIDs method."""
        # Create mock TSDFs
        left_tsdf = Mock(spec=TSDF)
        left_tsdf.series_ids = ["id1", "id2", "id3"]

        right_tsdf = Mock(spec=TSDF)
        right_tsdf.series_ids = ["id2", "id3", "id4"]

        joiner = UnionSortFilterAsOfJoiner()
        common = joiner.commonSeriesIDs(left_tsdf, right_tsdf)

        self.assertEqual(common, {"id2", "id3"})

    def test_prefixable_columns(self):
        """Test _prefixableColumns identification."""
        # Create mock TSDFs
        left_tsdf = Mock(spec=TSDF)
        left_tsdf.columns = ["timestamp", "id1", "value1", "shared_col"]
        left_tsdf.series_ids = ["id1"]

        right_tsdf = Mock(spec=TSDF)
        right_tsdf.columns = ["timestamp", "id1", "value2", "shared_col"]
        right_tsdf.series_ids = ["id1"]

        joiner = UnionSortFilterAsOfJoiner()
        prefixable = joiner._prefixableColumns(left_tsdf, right_tsdf)

        # Should include overlapping columns except series IDs
        self.assertEqual(prefixable, {"timestamp", "shared_col"})

    def test_check_are_joinable_valid(self):
        """Test _checkAreJoinable with valid TSDFs."""
        # Create mock TSDFs with matching schemas
        left_tsdf = Mock(spec=TSDF)
        left_tsdf.ts_schema = "schema1"
        left_tsdf.series_ids = ["id1", "id2"]

        right_tsdf = Mock(spec=TSDF)
        right_tsdf.ts_schema = "schema1"
        right_tsdf.series_ids = ["id1", "id2"]

        joiner = UnionSortFilterAsOfJoiner()
        # Should not raise an exception
        joiner._checkAreJoinable(left_tsdf, right_tsdf)

    def test_check_are_joinable_invalid_schema(self):
        """Test _checkAreJoinable with different schemas."""
        # Create mock TSDFs with different schemas
        left_tsdf = Mock(spec=TSDF)
        left_tsdf.ts_schema = "schema1"
        left_tsdf.series_ids = ["id1", "id2"]

        right_tsdf = Mock(spec=TSDF)
        right_tsdf.ts_schema = "schema2"
        right_tsdf.series_ids = ["id1", "id2"]

        joiner = UnionSortFilterAsOfJoiner()
        with self.assertRaises(ValueError) as cm:
            joiner._checkAreJoinable(left_tsdf, right_tsdf)

        self.assertIn("Timestamp schemas must match", str(cm.exception))

    def test_check_are_joinable_invalid_series_ids(self):
        """Test _checkAreJoinable with different series IDs."""
        # Create mock TSDFs with different series IDs
        left_tsdf = Mock(spec=TSDF)
        left_tsdf.ts_schema = "schema1"
        left_tsdf.series_ids = ["id1", "id2"]

        right_tsdf = Mock(spec=TSDF)
        right_tsdf.ts_schema = "schema1"
        right_tsdf.series_ids = ["id3", "id4"]

        joiner = UnionSortFilterAsOfJoiner()
        with self.assertRaises(ValueError) as cm:
            joiner._checkAreJoinable(left_tsdf, right_tsdf)

        self.assertIn("Series IDs must match", str(cm.exception))


class TestBroadcastAsOfJoiner(unittest.TestCase):
    """Test broadcast join strategy."""

    @patch('tempo.joins.strategies.SparkSession')
    def test_init_parameters(self, mock_spark_class):
        """Test initialization with various parameters."""
        mock_spark = Mock()
        joiner = BroadcastAsOfJoiner(
            spark=mock_spark,
            left_prefix="l",
            right_prefix="r",
            range_join_bin_size=120
        )

        self.assertEqual(joiner.spark, mock_spark)
        self.assertEqual(joiner.left_prefix, "l")
        self.assertEqual(joiner.right_prefix, "r")
        self.assertEqual(joiner.range_join_bin_size, 120)

    @patch('tempo.joins.strategies.SparkSession')
    def test_join_sets_config(self, mock_spark_class):
        """Test that join sets Spark configuration."""
        mock_spark = Mock()
        mock_spark.conf = Mock()

        joiner = BroadcastAsOfJoiner(spark=mock_spark)

        # Create mock TSDFs
        left_tsdf = Mock(spec=TSDF)
        right_tsdf = Mock(spec=TSDF)

        # Mock the required attributes and methods
        left_tsdf.ts_index = Mock()
        right_tsdf.ts_index = Mock()
        left_tsdf.series_ids = []
        right_tsdf.series_ids = []
        left_tsdf.df = Mock()
        right_tsdf.df = Mock()
        left_tsdf.ts_schema = Mock()
        right_tsdf.ts_schema = Mock()

        # Mock comparable expression
        right_tsdf.ts_index.comparableExpr = Mock(return_value=sfn.col("timestamp"))
        right_tsdf.ts_index.colname = "timestamp"
        right_tsdf.baseWindow = Mock(return_value=Mock())
        right_tsdf.withColumn = Mock(return_value=right_tsdf)

        # Mock join operation
        mock_df = Mock()
        mock_df.join = Mock(return_value=mock_df)
        mock_df.where = Mock(return_value=mock_df)
        mock_df.drop = Mock(return_value=mock_df)
        left_tsdf.df = mock_df
        right_tsdf.df = Mock()

        # Mock between method
        left_tsdf.ts_index.between = Mock(return_value=True)

        # Execute join
        try:
            joiner._join(left_tsdf, right_tsdf)
        except:
            pass  # We're mainly testing config setting

        # Verify Spark config was set
        mock_spark.conf.set.assert_called_with(
            "spark.databricks.optimizer.rangeJoin.binSize",
            "60"
        )


class TestUnionSortFilterAsOfJoiner(unittest.TestCase):
    """Test union-sort-filter join strategy."""

    def test_init_with_skip_nulls(self):
        """Test initialization with skipNulls parameter."""
        joiner_true = UnionSortFilterAsOfJoiner(skipNulls=True)
        self.assertTrue(joiner_true.skipNulls)

        joiner_false = UnionSortFilterAsOfJoiner(skipNulls=False)
        self.assertFalse(joiner_false.skipNulls)

    def test_init_with_tolerance(self):
        """Test initialization with tolerance parameter."""
        joiner = UnionSortFilterAsOfJoiner(tolerance=3600)
        self.assertEqual(joiner.tolerance, 3600)

        joiner_none = UnionSortFilterAsOfJoiner(tolerance=None)
        self.assertIsNone(joiner_none.tolerance)


class TestSkewAsOfJoiner(unittest.TestCase):
    """Test skew-aware join strategy."""

    def test_init_with_partition_val(self):
        """Test initialization with tsPartitionVal."""
        joiner = SkewAsOfJoiner(
            tsPartitionVal=3600,
            fraction=0.3
        )
        self.assertEqual(joiner.tsPartitionVal, 3600)
        self.assertEqual(joiner.fraction, 0.3)

    def test_init_inherits_from_union_sort_filter(self):
        """Test that SkewAsOfJoiner inherits from UnionSortFilterAsOfJoiner."""
        joiner = SkewAsOfJoiner(
            skipNulls=False,
            tolerance=1800
        )
        self.assertIsInstance(joiner, UnionSortFilterAsOfJoiner)
        self.assertFalse(joiner.skipNulls)
        self.assertEqual(joiner.tolerance, 1800)


class TestStrategySelection(unittest.TestCase):
    """Test choose_as_of_join_strategy function."""

    @patch('tempo.joins.strategies.get_bytes_from_plan')
    @patch('tempo.joins.strategies.SparkSession')
    def test_broadcast_selection_small_data(self, mock_spark_class, mock_get_bytes):
        """Test broadcast selection for small DataFrames."""
        # Mock SparkSession
        mock_spark = Mock()
        mock_spark_class.builder.getOrCreate.return_value = mock_spark

        # Mock small DataFrames (< 30MB)
        mock_get_bytes.side_effect = [10 * 1024 * 1024, 20 * 1024 * 1024]  # 10MB, 20MB

        # Create mock TSDFs
        left_tsdf = Mock(spec=TSDF)
        left_tsdf.df = Mock()

        right_tsdf = Mock(spec=TSDF)
        right_tsdf.df = Mock()

        strategy = choose_as_of_join_strategy(
            left_tsdf,
            right_tsdf,
            sql_join_opt=True
        )

        self.assertIsInstance(strategy, BroadcastAsOfJoiner)

    @patch('tempo.joins.strategies.get_bytes_from_plan')
    @patch('tempo.joins.strategies.SparkSession')
    def test_broadcast_selection_large_data(self, mock_spark_class, mock_get_bytes):
        """Test broadcast not selected for large DataFrames."""
        # Mock SparkSession
        mock_spark = Mock()
        mock_spark_class.builder.getOrCreate.return_value = mock_spark

        # Mock large DataFrames (> 30MB)
        mock_get_bytes.side_effect = [100 * 1024 * 1024, 200 * 1024 * 1024]  # 100MB, 200MB

        # Create mock TSDFs
        left_tsdf = Mock(spec=TSDF)
        left_tsdf.df = Mock()

        right_tsdf = Mock(spec=TSDF)
        right_tsdf.df = Mock()

        strategy = choose_as_of_join_strategy(
            left_tsdf,
            right_tsdf,
            sql_join_opt=True
        )

        self.assertIsInstance(strategy, UnionSortFilterAsOfJoiner)
        self.assertNotIsInstance(strategy, SkewAsOfJoiner)

    def test_skew_selection_with_partition_val(self):
        """Test skew strategy selection."""
        # Create mock TSDFs
        left_tsdf = Mock(spec=TSDF)
        right_tsdf = Mock(spec=TSDF)

        strategy = choose_as_of_join_strategy(
            left_tsdf,
            right_tsdf,
            tsPartitionVal=3600
        )

        self.assertIsInstance(strategy, SkewAsOfJoiner)

    def test_default_selection(self):
        """Test default strategy selection."""
        # Create mock TSDFs
        left_tsdf = Mock(spec=TSDF)
        right_tsdf = Mock(spec=TSDF)

        strategy = choose_as_of_join_strategy(
            left_tsdf,
            right_tsdf
        )

        self.assertIsInstance(strategy, UnionSortFilterAsOfJoiner)
        self.assertNotIsInstance(strategy, SkewAsOfJoiner)


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions."""

    @patch('tempo.joins.strategies.SparkSession')
    def test_get_bytes_from_plan_gib(self, mock_spark_class):
        """Test get_bytes_from_plan with GiB units."""
        mock_spark = Mock()
        mock_df = Mock()

        # Mock the plan extraction
        with patch('tempo.joins.strategies.get_spark_plan') as mock_get_plan:
            mock_get_plan.return_value = "sizeInBytes=1.5 GiB"

            result = get_bytes_from_plan(mock_df, mock_spark)

            expected = 1.5 * 1024 * 1024 * 1024
            self.assertEqual(result, expected)

    @patch('tempo.joins.strategies.SparkSession')
    def test_get_bytes_from_plan_mib(self, mock_spark_class):
        """Test get_bytes_from_plan with MiB units."""
        mock_spark = Mock()
        mock_df = Mock()

        # Mock the plan extraction
        with patch('tempo.joins.strategies.get_spark_plan') as mock_get_plan:
            mock_get_plan.return_value = "sizeInBytes=25.5 MiB"

            result = get_bytes_from_plan(mock_df, mock_spark)

            expected = 25.5 * 1024 * 1024
            self.assertEqual(result, expected)

    @patch('tempo.joins.strategies.SparkSession')
    def test_get_bytes_from_plan_error_handling(self, mock_spark_class):
        """Test get_bytes_from_plan error handling."""
        mock_spark = Mock()
        mock_df = Mock()

        # Mock the plan extraction with no sizeInBytes
        with patch('tempo.joins.strategies.get_spark_plan') as mock_get_plan:
            mock_get_plan.return_value = "no size information"

            result = get_bytes_from_plan(mock_df, mock_spark)

            # Should return inf to avoid broadcast
            self.assertEqual(result, float('inf'))


if __name__ == "__main__":
    unittest.main()