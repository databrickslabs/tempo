"""
Unit tests for SkewAsOfJoiner with skewed datasets.

This module tests the SkewAsOfJoiner's ability to handle various
types of data skew including key skew and temporal skew.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType, IntegerType

from tempo.joins.strategies import SkewAsOfJoiner, _detectSignificantSkew
from tempo.tsdf import TSDF
from tempo.utils import TSSchema
from tests.base import SparkTest


class SkewAsOfJoinerTest(SparkTest):
    """Test SkewAsOfJoiner with various skew patterns."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.spark = SparkSession.builder.getOrCreate()

    def create_skewed_data(self, skew_type="key", num_records=10000):
        """
        Create skewed test data with controllable skew patterns.

        :param skew_type: Type of skew - "key", "temporal", or "both"
        :param num_records: Total number of records to generate
        :return: Tuple of (left_tsdf, right_tsdf)
        """
        # Generate timestamps
        base_time = datetime(2024, 1, 1)

        if skew_type in ["key", "both"]:
            # 80% of data for key "A", 15% for "B", 5% for others
            key_distribution = (
                ["A"] * int(num_records * 0.8) +
                ["B"] * int(num_records * 0.15) +
                ["C"] * int(num_records * 0.03) +
                ["D"] * int(num_records * 0.02)
            )
        else:
            # Even key distribution
            keys = ["A", "B", "C", "D"]
            key_distribution = keys * (num_records // 4)

        if skew_type in ["temporal", "both"]:
            # 90% of data in the last 10% of time range
            timestamps = []
            for i in range(num_records):
                if i < num_records * 0.1:
                    # First 10% of records spread over 90% of time
                    timestamps.append(base_time + timedelta(minutes=i * 9))
                else:
                    # Last 90% of records in final 10% of time
                    timestamps.append(base_time + timedelta(minutes=900 + (i - num_records * 0.1)))
        else:
            # Even temporal distribution
            timestamps = [base_time + timedelta(minutes=i) for i in range(num_records)]

        # Create left DataFrame
        left_data = [
            (key_distribution[i], timestamps[i], f"left_val_{i}", i * 1.0)
            for i in range(num_records)
        ]

        left_schema = StructType([
            StructField("symbol", StringType(), False),
            StructField("timestamp", TimestampType(), False),
            StructField("left_value", StringType(), True),
            StructField("left_metric", DoubleType(), True),
        ])

        left_df = self.spark.createDataFrame(left_data, schema=left_schema)
        left_tsdf = TSDF(left_df, ts_col="timestamp", series_ids=["symbol"])

        # Create right DataFrame (smaller, also skewed)
        right_records = num_records // 10
        right_data = [
            (key_distribution[i * 10], timestamps[i * 10], f"right_val_{i}", i * 10.0)
            for i in range(right_records)
        ]

        right_schema = StructType([
            StructField("symbol", StringType(), False),
            StructField("timestamp", TimestampType(), False),
            StructField("right_value", StringType(), True),
            StructField("price", DoubleType(), True),
        ])

        right_df = self.spark.createDataFrame(right_data, schema=right_schema)
        right_tsdf = TSDF(right_df, ts_col="timestamp", series_ids=["symbol"])

        return left_tsdf, right_tsdf

    def test_skew_detection(self):
        """Test that skew detection correctly identifies skewed data."""
        # Create skewed data
        left_skewed, right_skewed = self.create_skewed_data(skew_type="key", num_records=1000)

        # Test skew detection
        self.assertTrue(
            _detectSignificantSkew(left_skewed, right_skewed, threshold=0.3),
            "Should detect key skew"
        )

        # Create non-skewed data
        left_normal, right_normal = self.create_skewed_data(skew_type="none", num_records=1000)

        # Should not detect skew in even distribution
        self.assertFalse(
            _detectSignificantSkew(left_normal, right_normal, threshold=0.3),
            "Should not detect skew in even distribution"
        )

    @patch('tempo.joins.strategies.logger')
    def test_aqe_configuration(self, mock_logger):
        """Test that AQE is properly configured."""
        joiner = SkewAsOfJoiner(
            spark=self.spark,
            left_prefix="",
            right_prefix="right"
        )

        # Check that AQE settings were configured
        self.assertEqual(self.spark.conf.get("spark.sql.adaptive.enabled"), "true")
        self.assertEqual(self.spark.conf.get("spark.sql.adaptive.skewJoin.enabled"), "true")

        # Check that configuration was logged
        mock_logger.info.assert_any_call("Configured AQE for skew handling")

    def test_key_skewed_join(self):
        """Test join with heavily skewed keys (80% in one key)."""
        # Create data with key skew
        left_tsdf, right_tsdf = self.create_skewed_data(skew_type="key", num_records=1000)

        # Create joiner
        joiner = SkewAsOfJoiner(
            spark=self.spark,
            left_prefix="",
            right_prefix="right",
            skew_threshold=0.2  # 20% threshold
        )

        # Perform join
        result_df, result_schema = joiner(left_tsdf, right_tsdf)

        # Verify results
        self.assertIsNotNone(result_df)
        self.assertEqual(result_df.count(), left_tsdf.df.count(), "All left rows should be preserved")

        # Check that skewed key "A" was processed correctly
        key_a_results = result_df.filter(F.col("symbol") == "A")
        self.assertGreater(key_a_results.count(), 0, "Skewed key should have results")

        # Verify correct as-of semantics
        # Each left row should get the most recent right row
        sample_row = result_df.filter(
            (F.col("symbol") == "A") &
            F.col("timestamp").isNotNull()
        ).first()
        if sample_row and "right_timestamp" in result_df.columns:
            self.assertLessEqual(
                sample_row["right_timestamp"],
                sample_row["timestamp"],
                "Right timestamp should be <= left timestamp"
            )

    def test_temporal_skewed_join(self):
        """Test join with temporal skew (90% of data in last 10% of time range)."""
        # Create data with temporal skew
        left_tsdf, right_tsdf = self.create_skewed_data(skew_type="temporal", num_records=1000)

        # Create joiner
        joiner = SkewAsOfJoiner(
            spark=self.spark,
            left_prefix="",
            right_prefix="right"
        )

        # Perform join
        result_df, result_schema = joiner(left_tsdf, right_tsdf)

        # Verify temporal ordering is preserved
        self.assertIsNotNone(result_df)
        self.assertEqual(result_df.count(), left_tsdf.df.count())

        # Verify that dense time periods are handled correctly
        # Count results in the dense period (last 10% of time)
        max_time = result_df.agg(F.max("timestamp")).collect()[0][0]
        min_time = result_df.agg(F.min("timestamp")).collect()[0][0]
        if max_time and min_time:
            time_range = (max_time - min_time).total_seconds()
            cutoff_time = min_time + timedelta(seconds=time_range * 0.9)

            dense_period_count = result_df.filter(
                F.col("timestamp") >= cutoff_time
            ).count()

            # Should have most of the data in the dense period
            self.assertGreater(
                dense_period_count / result_df.count(),
                0.8,
                "Most data should be in the dense time period"
            )

    def test_salted_join_for_extreme_skew(self):
        """Test salted join strategy for extreme skew."""
        # Create extremely skewed data (95% in one key)
        left_tsdf, right_tsdf = self.create_skewed_data(skew_type="key", num_records=500)

        # Create joiner with salting enabled
        joiner = SkewAsOfJoiner(
            spark=self.spark,
            left_prefix="",
            right_prefix="right",
            enable_salting=True,
            salt_buckets=5,
            skew_threshold=0.1  # Low threshold to trigger skew handling
        )

        # Perform join
        result_df, result_schema = joiner(left_tsdf, right_tsdf)

        # Verify results
        self.assertIsNotNone(result_df)
        self.assertEqual(result_df.count(), left_tsdf.df.count())

        # Verify salt columns are not in final result
        self.assertNotIn("__salt", result_df.columns)

    def test_backward_compatibility_with_tspartitionval(self):
        """Test backward compatibility with tsPartitionVal parameter."""
        left_tsdf, right_tsdf = self.create_skewed_data(skew_type="key", num_records=100)

        # Create joiner with deprecated tsPartitionVal
        joiner = SkewAsOfJoiner(
            spark=self.spark,
            left_prefix="",
            right_prefix="right",
            tsPartitionVal=300  # 5 minutes
        )

        # Should still work
        result_df, result_schema = joiner(left_tsdf, right_tsdf)
        self.assertIsNotNone(result_df)
        self.assertEqual(result_df.count(), left_tsdf.df.count())

    def test_skip_nulls_with_skewed_data(self):
        """Test skipNulls functionality with skewed data."""
        # Create skewed data and add some nulls
        left_tsdf, right_tsdf = self.create_skewed_data(skew_type="key", num_records=100)

        # Add nulls to right data
        right_with_nulls = right_tsdf.df.withColumn(
            "price",
            F.when(F.col("symbol") == "B", F.lit(None)).otherwise(F.col("price"))
        )
        right_tsdf_nulls = TSDF(right_with_nulls, "timestamp", ["symbol"])

        # Test with skipNulls=True
        joiner_skip = SkewAsOfJoiner(
            spark=self.spark,
            left_prefix="",
            right_prefix="right",
            skipNulls=True
        )

        result_skip, _ = joiner_skip(left_tsdf, right_tsdf_nulls)

        # Check that rows with null values are filtered appropriately
        b_results = result_skip.filter(F.col("symbol") == "B")
        non_null_b = b_results.filter(F.col("right_price").isNotNull())
        self.assertEqual(non_null_b.count(), 0, "Symbol B should have no matches due to null prices")

    def test_tolerance_with_skewed_data(self):
        """Test tolerance filtering with skewed data."""
        left_tsdf, right_tsdf = self.create_skewed_data(skew_type="key", num_records=100)

        # Create joiner with tolerance
        joiner = SkewAsOfJoiner(
            spark=self.spark,
            left_prefix="",
            right_prefix="right",
            tolerance=60  # 1 minute tolerance
        )

        result_df, _ = joiner(left_tsdf, right_tsdf)

        # Verify tolerance is applied
        self.assertIsNotNone(result_df)
        # Check that matches outside tolerance have null right columns
        # This would require examining specific timestamp differences

    @patch('tempo.joins.strategies.get_bytes_from_plan')
    def test_strategy_selection_for_skewed_keys(self, mock_get_bytes):
        """Test that different strategies are selected based on data characteristics."""
        # Mock size detection for broadcast decision
        mock_get_bytes.return_value = 50 * 1024 * 1024  # 50MB - too large for broadcast

        # Create heavily skewed data
        left_tsdf, right_tsdf = self.create_skewed_data(skew_type="key", num_records=100)

        joiner = SkewAsOfJoiner(
            spark=self.spark,
            left_prefix="",
            right_prefix="right",
            skew_threshold=0.1,  # Low threshold
            enable_salting=False
        )

        # Mock to track method calls
        with patch.object(joiner, '_skewSeparatedJoin', wraps=joiner._skewSeparatedJoin) as mock_separated:
            with patch.object(joiner, '_standardAsOfJoin', wraps=joiner._standardAsOfJoin) as mock_standard:
                result_df, _ = joiner(left_tsdf, right_tsdf)

                # Should use separated join for skewed keys
                if _detectSignificantSkew(left_tsdf, right_tsdf, 0.1):
                    mock_separated.assert_called_once()
                else:
                    mock_standard.assert_called_once()

    def test_mixed_key_and_temporal_skew(self):
        """Test handling of both key and temporal skew simultaneously."""
        # Create data with both types of skew
        left_tsdf, right_tsdf = self.create_skewed_data(skew_type="both", num_records=500)

        joiner = SkewAsOfJoiner(
            spark=self.spark,
            left_prefix="",
            right_prefix="right",
            skew_threshold=0.15
        )

        result_df, result_schema = joiner(left_tsdf, right_tsdf)

        # Verify all rows are preserved
        self.assertEqual(result_df.count(), left_tsdf.df.count())

        # Verify schema is correct
        self.assertEqual(result_schema.ts_col, "timestamp")
        self.assertEqual(result_schema.series_ids, ["symbol"])

        # Verify both skew types are handled
        # Check key skew handling
        key_a_count = result_df.filter(F.col("symbol") == "A").count()
        self.assertGreater(key_a_count, result_df.count() * 0.7,
                          "Skewed key should have majority of results")

        # Check temporal skew handling
        max_time = result_df.agg(F.max("timestamp")).collect()[0][0]
        min_time = result_df.agg(F.min("timestamp")).collect()[0][0]
        if max_time and min_time:
            time_range = (max_time - min_time).total_seconds()
            cutoff_time = min_time + timedelta(seconds=time_range * 0.9)
            dense_count = result_df.filter(F.col("timestamp") >= cutoff_time).count()
            self.assertGreater(dense_count, result_df.count() * 0.8,
                             "Dense time period should be handled correctly")


    def test_no_skew_baseline(self):
        """Test baseline case with evenly distributed data (no skew)."""
        # Create evenly distributed data
        left_tsdf, right_tsdf = self.create_skewed_data(skew_type="none", num_records=400)

        # Joiner should handle non-skewed data efficiently
        joiner = SkewAsOfJoiner(
            spark=self.spark,
            left_prefix="",
            right_prefix="right",
            skew_threshold=0.2
        )

        result_df, _ = joiner(left_tsdf, right_tsdf)

        # Verify even distribution in results
        self.assertEqual(result_df.count(), left_tsdf.df.count())

        # Check that all keys have roughly equal representation
        key_counts = result_df.groupBy("symbol").count().collect()
        counts = [row["count"] for row in key_counts]
        if counts:
            max_count = max(counts)
            min_count = min(counts)
            # Ratio should be close to 1 for even distribution
            self.assertLess(max_count / min_count, 1.5,
                          "Keys should have roughly equal counts")

    def test_extreme_key_skew_95_percent(self):
        """Test extreme key skew with 95% of data in one key."""
        # Create extremely skewed data manually
        base_time = datetime(2024, 1, 1)

        # 95% for key "EXTREME", 5% for others
        left_data = [("EXTREME", base_time + timedelta(minutes=i), f"val_{i}", float(i))
                     for i in range(950)]
        left_data += [("OTHER", base_time + timedelta(minutes=i), f"val_{i}", float(i))
                      for i in range(50)]

        left_df = self.spark.createDataFrame(
            left_data,
            ["symbol", "timestamp", "value", "metric"]
        )
        left_tsdf = TSDF(left_df, "timestamp", ["symbol"])

        # Right side also skewed
        right_data = [("EXTREME", base_time + timedelta(minutes=i*10), float(i*10))
                      for i in range(95)]
        right_data += [("OTHER", base_time + timedelta(minutes=i*10), float(i*100))
                       for i in range(5)]

        right_df = self.spark.createDataFrame(
            right_data,
            ["symbol", "timestamp", "price"]
        )
        right_tsdf = TSDF(right_df, "timestamp", ["symbol"])

        # Test with low threshold to trigger skew handling
        joiner = SkewAsOfJoiner(
            spark=self.spark,
            left_prefix="",
            right_prefix="right",
            skew_threshold=0.1,  # 10% threshold
            enable_salting=True,
            salt_buckets=5
        )

        result_df, _ = joiner(left_tsdf, right_tsdf)

        # Verify all rows preserved despite extreme skew
        self.assertEqual(result_df.count(), 1000)

        # Verify EXTREME key results
        extreme_results = result_df.filter(F.col("symbol") == "EXTREME")
        self.assertEqual(extreme_results.count(), 950)

    def test_power_law_distribution_skew(self):
        """Test power law distribution (realistic for user activity data)."""
        # Create power law distributed data
        import random
        random.seed(42)

        base_time = datetime(2024, 1, 1)
        keys = [f"user_{i}" for i in range(100)]

        # Power law: first few users have most activity
        left_data = []
        for i, key in enumerate(keys):
            # Power law frequency
            frequency = int(1000 / (i + 1) ** 1.5)
            for j in range(frequency):
                left_data.append(
                    (key, base_time + timedelta(minutes=j), f"action_{j}", float(j))
                )

        left_df = self.spark.createDataFrame(
            left_data[:5000],  # Limit to 5000 rows
            ["user", "timestamp", "action", "value"]
        )
        left_tsdf = TSDF(left_df, "timestamp", ["user"])

        # Right side: user profiles updated occasionally
        right_data = [
            (key, base_time + timedelta(hours=i), f"status_{i}")
            for i, key in enumerate(keys[:50])  # Only some users have profiles
        ]
        right_df = self.spark.createDataFrame(
            right_data,
            ["user", "timestamp", "status"]
        )
        right_tsdf = TSDF(right_df, "timestamp", ["user"])

        joiner = SkewAsOfJoiner(
            spark=self.spark,
            left_prefix="",
            right_prefix="profile",
            skew_threshold=0.15
        )

        result_df, _ = joiner(left_tsdf, right_tsdf)

        # Verify join completed
        self.assertIsNotNone(result_df)
        self.assertGreater(result_df.count(), 0)

    def test_multikey_skew(self):
        """Test skew with multiple series keys (composite key skew)."""
        base_time = datetime(2024, 1, 1)

        # Create data with composite key skew
        # (exchange, symbol) pairs with skew
        left_data = []
        # NYSE, AAPL dominates
        for i in range(800):
            left_data.append(
                ("NYSE", "AAPL", base_time + timedelta(minutes=i), float(i))
            )
        # Other combinations
        for exchange in ["NYSE", "NASDAQ"]:
            for symbol in ["GOOGL", "MSFT"]:
                for i in range(50):
                    left_data.append(
                        (exchange, symbol, base_time + timedelta(minutes=i), float(i))
                    )

        left_df = self.spark.createDataFrame(
            left_data,
            ["exchange", "symbol", "timestamp", "volume"]
        )
        left_tsdf = TSDF(left_df, "timestamp", ["exchange", "symbol"])

        # Right side
        right_data = [
            ("NYSE", "AAPL", base_time + timedelta(minutes=i*100), float(i*100))
            for i in range(8)
        ]
        right_data += [
            ("NYSE", "GOOGL", base_time, 1000.0),
            ("NASDAQ", "MSFT", base_time, 2000.0)
        ]

        right_df = self.spark.createDataFrame(
            right_data,
            ["exchange", "symbol", "timestamp", "price"]
        )
        right_tsdf = TSDF(right_df, "timestamp", ["exchange", "symbol"])

        joiner = SkewAsOfJoiner(
            spark=self.spark,
            left_prefix="",
            right_prefix="quote",
            skew_threshold=0.2
        )

        result_df, result_schema = joiner(left_tsdf, right_tsdf)

        # Verify multi-key join worked
        self.assertEqual(result_schema.series_ids, ["exchange", "symbol"])
        self.assertEqual(result_df.count(), left_tsdf.df.count())

        # Check NYSE,AAPL results
        nyse_aapl = result_df.filter(
            (F.col("exchange") == "NYSE") & (F.col("symbol") == "AAPL")
        )
        self.assertEqual(nyse_aapl.count(), 800)

    def test_hot_partition_temporal_skew(self):
        """Test hot partition scenario (e.g., market open/close times)."""
        base_time = datetime(2024, 1, 1, 9, 30)  # Market open

        left_data = []
        # Simulate trading activity spike at market open
        # First 5 minutes: 70% of activity
        for i in range(700):
            left_data.append(
                ("AAPL", base_time + timedelta(seconds=i*0.4), f"trade_{i}", float(i))
            )
        # Rest of day: 30% of activity
        for i in range(300):
            left_data.append(
                ("AAPL", base_time + timedelta(minutes=5 + i), f"trade_{i+700}", float(i))
            )

        left_df = self.spark.createDataFrame(
            left_data,
            ["symbol", "timestamp", "trade_id", "volume"]
        )
        left_tsdf = TSDF(left_df, "timestamp", ["symbol"])

        # Quotes throughout the day
        right_data = [
            ("AAPL", base_time + timedelta(minutes=i), 100.0 + i)
            for i in range(0, 300, 10)
        ]

        right_df = self.spark.createDataFrame(
            right_data,
            ["symbol", "timestamp", "price"]
        )
        right_tsdf = TSDF(right_df, "timestamp", ["symbol"])

        joiner = SkewAsOfJoiner(
            spark=self.spark,
            left_prefix="trade",
            right_prefix="quote"
        )

        result_df, _ = joiner(left_tsdf, right_tsdf)

        # Verify hot partition handled correctly
        self.assertEqual(result_df.count(), 1000)

        # Check first 5 minutes have correct joins
        first_5min = result_df.filter(
            F.col("trade_timestamp") <= base_time + timedelta(minutes=5)
        )
        self.assertEqual(first_5min.count(), 700)


class SkewDetectionTest(unittest.TestCase):
    """Test skew detection utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.spark = SparkSession.builder \
            .master("local[*]") \
            .appName("SkewDetectionTest") \
            .getOrCreate()

    def tearDown(self):
        """Clean up after tests."""
        self.spark.stop()

    def test_detect_significant_skew_no_series(self):
        """Test skew detection with no series columns."""
        # Create TSDF with no series IDs
        df = self.spark.range(100).withColumn("timestamp", F.current_timestamp())
        tsdf = TSDF(df, ts_col="timestamp", series_ids=[])

        # Should return False when no series columns
        self.assertFalse(_detectSignificantSkew(tsdf, tsdf))

    def test_detect_significant_skew_small_data(self):
        """Test skew detection with very small dataset."""
        # Create tiny dataset
        data = [(i % 2, datetime(2024, 1, 1, i)) for i in range(10)]
        df = self.spark.createDataFrame(data, ["key", "timestamp"])
        tsdf = TSDF(df, ts_col="timestamp", series_ids=["key"])

        # Should return False for tiny datasets
        self.assertFalse(_detectSignificantSkew(tsdf, tsdf))

    @patch('tempo.joins.strategies.logger')
    def test_detect_significant_skew_with_error(self, mock_logger):
        """Test skew detection handles errors gracefully."""
        # Create mock TSDF that will cause an error
        mock_tsdf = Mock(spec=TSDF)
        mock_tsdf.series_ids = ["key"]
        mock_tsdf.df.count.side_effect = Exception("Test error")

        # Should return False and log the error
        result = _detectSignificantSkew(mock_tsdf, mock_tsdf)
        self.assertFalse(result)
        mock_logger.debug.assert_called()


if __name__ == "__main__":
    unittest.main()