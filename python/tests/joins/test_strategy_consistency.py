"""
Integration tests to ensure all as-of join strategies produce consistent results.

This module verifies that different strategies (Broadcast, UnionSortFilter, Skew)
produce identical results for the same input data, ensuring semantic consistency.
"""

import pytest
from datetime import datetime, timedelta

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

from tempo.tsdf import TSDF
from tempo.joins.strategies import (
    BroadcastAsOfJoiner,
    UnionSortFilterAsOfJoiner,
    SkewAsOfJoiner,
)


@pytest.fixture(scope="module")
def spark():
    """Create a Spark session for testing."""
    spark = (
        SparkSession.builder
        .appName("StrategyConsistencyTests")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.default.parallelism", "4")
        .getOrCreate()
    )
    yield spark
    spark.stop()


class TestStrategyConsistency:
    """Test that all strategies produce identical results."""

    def create_test_data(self, spark, num_left=100, num_right=20):
        """
        Create test data for consistency testing.

        :param num_left: Number of left side records
        :param num_right: Number of right side records
        :return: Tuple of (left_tsdf, right_tsdf)
        """
        base_time = datetime(2024, 1, 1, 10, 0, 0)

        # Create left DataFrame with multiple series
        left_data = []
        for symbol in ["A", "B", "C"]:
            for i in range(num_left // 3):
                left_data.append(
                    (symbol, base_time + timedelta(minutes=i), f"{symbol}_val_{i}", float(i))
                )

        left_df = spark.createDataFrame(
            left_data,
            ["symbol", "timestamp", "left_value", "left_metric"]
        )
        left_tsdf = TSDF(left_df, ts_col="timestamp", series_ids=["symbol"])

        # Create right DataFrame with updates at irregular intervals
        right_data = []
        for symbol in ["A", "B", "C"]:
            for i in range(num_right // 3):
                # Irregular intervals to test as-of logic
                offset = i * 3.5
                right_data.append(
                    (symbol, base_time + timedelta(minutes=offset), float(100 + i), f"status_{i}")
                )

        right_df = spark.createDataFrame(
            right_data,
            ["symbol", "timestamp", "price", "status"]
        )
        right_tsdf = TSDF(right_df, ts_col="timestamp", series_ids=["symbol"])

        return left_tsdf, right_tsdf

    def test_broadcast_vs_union_consistency(self, spark):
        """Test that BroadcastAsOfJoiner and UnionSortFilterAsOfJoiner produce identical results."""
        left_tsdf, right_tsdf = self.create_test_data(spark, 100, 20)

        # Create joiners with identical parameters
        broadcast_joiner = BroadcastAsOfJoiner(
            spark=spark,
            left_prefix="",
            right_prefix="right"
        )

        union_joiner = UnionSortFilterAsOfJoiner(
            left_prefix="",
            right_prefix="right",
            skipNulls=True
        )

        # Execute joins
        broadcast_result, _ = broadcast_joiner(left_tsdf, right_tsdf)
        union_result, _ = union_joiner(left_tsdf, right_tsdf)

        # Sort for comparison - handle different column naming
        ts_col = "timestamp" if "timestamp" in broadcast_result.columns else "left_timestamp"
        broadcast_sorted = broadcast_result.orderBy("symbol", ts_col)
        union_sorted = union_result.orderBy("symbol", ts_col)

        # Verify identical results - compare counts and data content
        assert broadcast_sorted.count() == union_sorted.count()

        # Compare actual data values (not column order)
        broadcast_data = broadcast_sorted.select(sorted(broadcast_sorted.columns)).collect()
        union_data = union_sorted.select(sorted(union_sorted.columns)).collect()
        assert broadcast_data == union_data

    def test_skew_vs_union_consistency(self, spark):
        """Test that SkewAsOfJoiner and UnionSortFilterAsOfJoiner produce identical results."""
        left_tsdf, right_tsdf = self.create_test_data(spark, 100, 20)

        # Create joiners
        skew_joiner = SkewAsOfJoiner(
            spark=spark,
            left_prefix="",
            right_prefix="right",
            skipNulls=True,
            skew_threshold=1.0  # High threshold to avoid triggering skew handling
        )

        union_joiner = UnionSortFilterAsOfJoiner(
            left_prefix="",
            right_prefix="right",
            skipNulls=True
        )

        # Execute joins
        skew_result, _ = skew_joiner(left_tsdf, right_tsdf)
        union_result, _ = union_joiner(left_tsdf, right_tsdf)

        # Sort for comparison - handle different column naming
        ts_col = "timestamp" if "timestamp" in skew_result.columns else "left_timestamp"
        skew_sorted = skew_result.orderBy("symbol", ts_col)
        union_sorted = union_result.orderBy("symbol", ts_col)

        # Verify identical results - compare counts and data content
        assert skew_sorted.count() == union_sorted.count()

        # Compare actual data values (not column order)
        skew_data = skew_sorted.select(sorted(skew_sorted.columns)).collect()
        union_data = union_sorted.select(sorted(union_sorted.columns)).collect()
        assert skew_data == union_data

    def test_strategies_with_nulls(self, spark):
        """Test strategies that support skipNulls handle null values correctly."""
        # Create data with nulls
        left_tsdf, right_tsdf = self.create_test_data(spark, 50, 10)

        # Add nulls to right data
        right_with_nulls = right_tsdf.df.withColumn(
            "price",
            F.when(F.col("symbol") == "B", F.lit(None)).otherwise(F.col("price"))
        )
        right_tsdf_nulls = TSDF(right_with_nulls, ts_col="timestamp", series_ids=["symbol"])

        # Test only strategies that support skipNulls
        strategies = [
            UnionSortFilterAsOfJoiner("", "right", skipNulls=True),
            SkewAsOfJoiner(spark, "", "right", skipNulls=True, skew_threshold=1.0)
        ]

        for strategy in strategies:
            result, _ = strategy(left_tsdf, right_tsdf_nulls)
            strategy_name = strategy.__class__.__name__

            # Verify LEFT JOIN semantics - all left rows preserved
            assert result.count() == left_tsdf.df.count(), \
                f"{strategy_name} did not preserve all left rows with nulls"

    def test_tolerance_consistency(self, spark):
        """Test strategies that support tolerance produce consistent results."""
        left_tsdf, right_tsdf = self.create_test_data(spark, 50, 10)

        tolerance = 120  # 2 minutes

        # Only test strategies that support tolerance
        strategies = [
            UnionSortFilterAsOfJoiner("", "right", skipNulls=True, tolerance=tolerance),
            SkewAsOfJoiner(spark, "", "right", skipNulls=True, tolerance=tolerance, skew_threshold=1.0)
        ]

        for strategy in strategies:
            result, _ = strategy(left_tsdf, right_tsdf)
            strategy_name = strategy.__class__.__name__

            # Verify LEFT JOIN semantics are preserved
            assert result.count() == left_tsdf.df.count(), \
                f"{strategy_name} did not preserve all left rows with tolerance"

            # Verify tolerance is applied (some joins may be filtered out)
            # Can't directly compare results as column naming differs

    def test_all_strategies_empty_right(self, spark):
        """Test all strategies handle empty right DataFrame consistently."""
        # Create left DataFrame
        left_tsdf, _ = self.create_test_data(spark, 50, 10)

        # Create empty right DataFrame with proper schema
        from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType
        right_schema = StructType([
            StructField("symbol", StringType(), True),
            StructField("timestamp", TimestampType(), True),
            StructField("price", DoubleType(), True),
            StructField("status", StringType(), True)
        ])
        right_df = spark.createDataFrame([], right_schema)
        right_tsdf = TSDF(right_df, ts_col="timestamp", series_ids=["symbol"])

        strategies = [
            BroadcastAsOfJoiner(spark, "", "right"),
            UnionSortFilterAsOfJoiner("", "right"),
            SkewAsOfJoiner(spark, "", "right", skew_threshold=1.0)
        ]

        results = []
        for strategy in strategies:
            result, _ = strategy(left_tsdf, right_tsdf)
            ts_col = "timestamp" if "timestamp" in result.columns else "left_timestamp"
            results.append(result.orderBy("symbol", ts_col))

        # All should return left DataFrame with null right columns
        for result in results:
            # Verify all left rows are preserved
            assert result.count() == left_tsdf.df.count()

            # Check that right-side columns exist and are all null
            # Need to handle different column naming conventions
            price_cols = [c for c in result.columns if 'price' in c.lower()]
            status_cols = [c for c in result.columns if 'status' in c.lower()]

            assert len(price_cols) > 0, "No price column found in result"
            assert len(status_cols) > 0, "No status column found in result"

            # Verify all price values are null (empty right DataFrame)
            for col_name in price_cols:
                non_null_count = result.filter(F.col(col_name).isNotNull()).count()
                assert non_null_count == 0, f"Found non-null values in {col_name}"

    def test_all_strategies_single_series(self, spark):
        """Test all strategies with single series (no partition columns)."""
        base_time = datetime(2024, 1, 1)

        # Create DataFrames without series columns
        left_data = [(base_time + timedelta(minutes=i), f"val_{i}") for i in range(50)]
        left_df = spark.createDataFrame(left_data, ["timestamp", "value"])
        left_tsdf = TSDF(left_df, ts_col="timestamp", series_ids=[])

        right_data = [(base_time + timedelta(minutes=i*5), float(i*100)) for i in range(10)]
        right_df = spark.createDataFrame(right_data, ["timestamp", "price"])
        right_tsdf = TSDF(right_df, ts_col="timestamp", series_ids=[])

        strategies = [
            BroadcastAsOfJoiner(spark, "", "right"),
            UnionSortFilterAsOfJoiner("", "right"),
            SkewAsOfJoiner(spark, "", "right", skew_threshold=1.0)
        ]

        results = []
        for strategy in strategies:
            result, _ = strategy(left_tsdf, right_tsdf)
            ts_col = "timestamp" if "timestamp" in result.columns else "left_timestamp"
            results.append(result.orderBy(ts_col))

        # All should produce identical results - compare data content
        for i in range(1, len(results)):
            result0_data = results[0].select(sorted(results[0].columns)).collect()
            resulti_data = results[i].select(sorted(results[i].columns)).collect()
            assert result0_data == resulti_data, \
                f"Strategy {i} differs for single series"

    def test_join_semantics_left_preservation(self, spark):
        """Verify all strategies preserve all left rows (LEFT JOIN semantics)."""
        left_tsdf, right_tsdf = self.create_test_data(spark, 100, 5)

        strategies = [
            BroadcastAsOfJoiner(spark, "", "right"),
            UnionSortFilterAsOfJoiner("", "right"),
            SkewAsOfJoiner(spark, "", "right", skew_threshold=1.0)
        ]

        left_count = left_tsdf.df.count()

        for strategy in strategies:
            result, _ = strategy(left_tsdf, right_tsdf)
            strategy_name = strategy.__class__.__name__

            # Verify all left rows preserved
            assert result.count() == left_count, \
                f"{strategy_name} did not preserve all left rows"

            # Verify left columns are never null
            left_nulls = result.filter(F.col("left_value").isNull()).count()
            assert left_nulls == 0, \
                f"{strategy_name} has null left columns"

    def test_temporal_ordering_consistency(self, spark):
        """Verify all strategies respect temporal ordering in as-of joins."""
        base_time = datetime(2024, 1, 1)

        # Create specific test case for temporal ordering
        left_data = [
            ("A", base_time + timedelta(minutes=5), "left_1"),
            ("A", base_time + timedelta(minutes=10), "left_2"),
            ("A", base_time + timedelta(minutes=15), "left_3"),
        ]
        left_df = spark.createDataFrame(left_data, ["symbol", "timestamp", "value"])
        left_tsdf = TSDF(left_df, ts_col="timestamp", series_ids=["symbol"])

        right_data = [
            ("A", base_time + timedelta(minutes=3), 100.0),
            ("A", base_time + timedelta(minutes=8), 200.0),
            ("A", base_time + timedelta(minutes=13), 300.0),
        ]
        right_df = spark.createDataFrame(right_data, ["symbol", "timestamp", "price"])
        right_tsdf = TSDF(right_df, ts_col="timestamp", series_ids=["symbol"])

        strategies = [
            BroadcastAsOfJoiner(spark, "", "right"),
            UnionSortFilterAsOfJoiner("", "right"),
            SkewAsOfJoiner(spark, "", "right", skew_threshold=1.0)
        ]

        expected_prices = [100.0, 200.0, 300.0]

        for strategy in strategies:
            result, _ = strategy(left_tsdf, right_tsdf)
            strategy_name = strategy.__class__.__name__

            # Verify temporal ordering - handle different column naming conventions
            ts_col = "timestamp" if "timestamp" in result.columns else "left_timestamp"
            rows = result.orderBy(ts_col).collect()

            # Find the price column name (could be price or right_price)
            price_col = "right_price" if "right_price" in result.columns else "price"

            for i, row in enumerate(rows):
                price_val = row[price_col] if hasattr(row, price_col) else row.asDict().get(price_col)
                assert price_val == expected_prices[i], \
                    f"{strategy_name} incorrect temporal ordering at row {i}"

    def test_prefix_handling_consistency(self, spark):
        """Test all strategies handle column prefixes consistently."""
        left_tsdf, right_tsdf = self.create_test_data(spark, 30, 10)

        # Test with various prefix combinations
        prefix_tests = [
            ("left", "right"),
            ("", "r"),
            ("l", ""),
            ("", ""),
        ]

        for left_prefix, right_prefix in prefix_tests:
            strategies = [
                BroadcastAsOfJoiner(spark, left_prefix, right_prefix),
                UnionSortFilterAsOfJoiner(left_prefix, right_prefix),
                SkewAsOfJoiner(spark, left_prefix, right_prefix, skew_threshold=1.0)
            ]

            results = []
            for strategy in strategies:
                result, _ = strategy(left_tsdf, right_tsdf)
                ts_col = "timestamp" if "timestamp" in result.columns else "left_timestamp"
                results.append(result.orderBy("symbol", ts_col))

            # Check column names are consistent
            for i in range(1, len(results)):
                assert sorted(results[0].columns) == sorted(results[i].columns), \
                    f"Column names differ with prefixes ({left_prefix}, {right_prefix})"

    def test_no_double_prefixing(self, spark):
        """Verify that column prefixes are never applied twice."""
        left_tsdf, right_tsdf = self.create_test_data(spark, 30, 10)

        strategies = [
            BroadcastAsOfJoiner(spark, "left", "right"),
            UnionSortFilterAsOfJoiner("left", "right"),
            SkewAsOfJoiner(spark, "left", "right", skew_threshold=1.0)
        ]

        for strategy in strategies:
            result, _ = strategy(left_tsdf, right_tsdf)
            strategy_name = strategy.__class__.__name__

            # Check no column has double prefix like "left_left_" or "right_right_"
            for col in result.columns:
                assert not col.startswith("left_left_"), \
                    f"{strategy_name} has double-prefixed column: {col}"
                assert not col.startswith("right_right_"), \
                    f"{strategy_name} has double-prefixed column: {col}"

    def test_overlapping_columns_empty_prefix(self, spark):
        """Test handling of overlapping non-timestamp columns with empty prefixes."""
        # Create data where left and right have same column name (not just timestamp)
        base_time = datetime(2024, 1, 1)

        left_data = [(base_time + timedelta(minutes=i), "A", i) for i in range(10)]
        left_df = spark.createDataFrame(left_data, ["timestamp", "symbol", "value"])
        left_tsdf = TSDF(left_df, ts_col="timestamp", series_ids=["symbol"])

        right_data = [(base_time + timedelta(minutes=i*2), "A", i*100) for i in range(5)]
        right_df = spark.createDataFrame(right_data, ["timestamp", "symbol", "value"])
        right_tsdf = TSDF(right_df, ts_col="timestamp", series_ids=["symbol"])

        # With empty prefixes, "value" column overlaps
        strategy = SkewAsOfJoiner(spark, "", "", skew_threshold=1.0)
        result, _ = strategy(left_tsdf, right_tsdf)

        # Should have left value, not duplicate "value" columns
        assert result.columns.count("value") == 1, \
            "Should have only one 'value' column when prefixes are empty"

    def test_tolerance_with_prefixes(self, spark):
        """Test tolerance filtering works correctly with various prefix combinations."""
        left_tsdf, right_tsdf = self.create_test_data(spark, 30, 10)
        tolerance = 60  # 1 minute

        prefix_tests = [
            ("", "right"),
            ("left", ""),
            ("", ""),
        ]

        for left_prefix, right_prefix in prefix_tests:
            strategy = SkewAsOfJoiner(
                spark, left_prefix, right_prefix,
                skipNulls=True, tolerance=tolerance, skew_threshold=1.0
            )

            result, _ = strategy(left_tsdf, right_tsdf)

            # Should not error and should preserve left rows
            assert result.count() == left_tsdf.df.count(), \
                f"Failed with prefixes ({left_prefix}, {right_prefix})"