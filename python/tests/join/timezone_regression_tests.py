"""
Timezone regression tests for composite indexes in as-of joins.

This module tests timezone handling in as-of joins, especially for
composite timestamp indexes with nanosecond precision.
"""

import unittest
from datetime import datetime, timezone
import pytz

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType

from tests.base import SparkTest
from tempo import TSDF
from tempo.joins.strategies import BroadcastAsOfJoiner, UnionSortFilterAsOfJoiner


class TimezoneRegressionTest(SparkTest):
    """Test timezone handling in as-of joins with composite indexes."""

    def setUp(self):
        """Set up test data with various timezone scenarios."""
        super().setUp()

        # Set explicit timezone for testing
        self.spark.conf.set("spark.sql.session.timeZone", "UTC")

    def test_simple_timestamp_timezone_consistency(self):
        """Test that simple timestamp columns maintain timezone consistency."""
        # Create test data with explicit UTC timestamps
        left_data = [
            ("S1", datetime(2022, 1, 1, 10, 0, 0, tzinfo=timezone.utc), 100.0),
            ("S1", datetime(2022, 1, 1, 11, 0, 0, tzinfo=timezone.utc), 101.0),
        ]
        left_df = self.spark.createDataFrame(
            left_data, ["symbol", "timestamp", "value"]
        )

        right_data = [
            ("S1", datetime(2022, 1, 1, 9, 30, 0, tzinfo=timezone.utc), 99.5),
            ("S1", datetime(2022, 1, 1, 10, 30, 0, tzinfo=timezone.utc), 100.5),
        ]
        right_df = self.spark.createDataFrame(
            right_data, ["symbol", "timestamp", "price"]
        )

        # Create TSDFs
        left_tsdf = TSDF(left_df, ts_col="timestamp", series_ids=["symbol"])
        right_tsdf = TSDF(right_df, ts_col="timestamp", series_ids=["symbol"])

        # Test both join strategies
        broadcast_joiner = BroadcastAsOfJoiner(self.spark)
        broadcast_result = broadcast_joiner(left_tsdf, right_tsdf)

        union_joiner = UnionSortFilterAsOfJoiner()
        union_result = union_joiner(left_tsdf, right_tsdf)

        # Both strategies should produce the same result
        self.assertEqual(broadcast_result.df.count(), union_result.df.count())

        # Check that timestamps are preserved correctly
        broadcast_rows = broadcast_result.df.collect()
        union_rows = union_result.df.collect()

        for b_row, u_row in zip(broadcast_rows, union_rows):
            # Compare timestamps
            self.assertEqual(b_row["left_timestamp"], u_row["left_timestamp"])
            self.assertEqual(b_row["right_timestamp"], u_row["right_timestamp"])

    def test_nanosecond_timestamp_timezone_handling(self):
        """Test timezone handling with nanosecond precision timestamps."""
        # Create test data with nanosecond precision
        # Using string timestamps that will be parsed
        left_data = [
            ("S1", "2022-01-01 10:00:00.123456789", 100.0),
            ("S1", "2022-01-01 11:00:00.987654321", 101.0),
        ]
        left_df = self.spark.createDataFrame(
            left_data, ["symbol", "timestamp_str", "value"]
        )

        right_data = [
            ("S1", "2022-01-01 09:30:00.111111111", 99.5),
            ("S1", "2022-01-01 10:30:00.222222222", 100.5),
        ]
        right_df = self.spark.createDataFrame(
            right_data, ["symbol", "timestamp_str", "price"]
        )

        # Parse timestamps with nanosecond precision
        # This simulates the fromStringTimestamp behavior
        left_df = left_df.withColumn(
            "timestamp",
            F.to_timestamp("timestamp_str", "yyyy-MM-dd HH:mm:ss.SSSSSSSSS")
        )

        right_df = right_df.withColumn(
            "timestamp",
            F.to_timestamp("timestamp_str", "yyyy-MM-dd HH:mm:ss.SSSSSSSSS")
        )

        # Create TSDFs
        left_tsdf = TSDF(left_df, ts_col="timestamp", series_ids=["symbol"])
        right_tsdf = TSDF(right_df, ts_col="timestamp", series_ids=["symbol"])

        # Test both join strategies
        broadcast_joiner = BroadcastAsOfJoiner(self.spark)
        broadcast_result = broadcast_joiner(left_tsdf, right_tsdf)

        union_joiner = UnionSortFilterAsOfJoiner()
        union_result = union_joiner(left_tsdf, right_tsdf)

        # Both should have the same number of rows
        self.assertEqual(broadcast_result.df.count(), 2)
        self.assertEqual(union_result.df.count(), 2)

    def test_composite_index_timezone_consistency(self):
        """Test that composite indexes maintain timezone consistency."""
        # Create test data that will result in composite timestamp indexes
        left_data = [
            ("S1", "2022-01-01T10:00:00.123456789Z", 100.0),
            ("S1", "2022-01-01T11:00:00.123456789Z", 101.0),
        ]

        schema = StructType([
            StructField("symbol", StringType(), True),
            StructField("timestamp_str", StringType(), True),
            StructField("value", DoubleType(), True)
        ])

        left_df = self.spark.createDataFrame(left_data, schema)

        # Parse the ISO timestamp string
        left_df = left_df.withColumn(
            "timestamp",
            F.to_timestamp("timestamp_str")
        )

        right_data = [
            ("S1", "2022-01-01T09:30:00.123456789Z", 99.5),
            ("S1", "2022-01-01T10:30:00.123456789Z", 100.5),
        ]

        right_df = self.spark.createDataFrame(right_data, schema)
        right_df = right_df.withColumn(
            "timestamp",
            F.to_timestamp("timestamp_str")
        )

        # Create TSDFs with nanosecond precision
        # This would create composite indexes internally
        left_tsdf = TSDF(left_df, ts_col="timestamp", series_ids=["symbol"])
        right_tsdf = TSDF(right_df, ts_col="timestamp", series_ids=["symbol"])

        # Test broadcast join
        broadcast_joiner = BroadcastAsOfJoiner(self.spark)
        result = broadcast_joiner(left_tsdf, right_tsdf)

        # Verify result has expected number of rows
        self.assertEqual(result.df.count(), 2)

        # Check that the timestamps are correctly aligned
        rows = result.df.orderBy("left_timestamp").collect()

        # First row should match first left with first right
        self.assertIsNotNone(rows[0]["right_timestamp"])
        # Second row should match second left with second right
        self.assertIsNotNone(rows[1]["right_timestamp"])

    def test_different_timezone_conversion(self):
        """Test joining data from different timezones."""
        # Create left data in US/Eastern timezone
        eastern = pytz.timezone('US/Eastern')
        left_data = [
            ("S1", eastern.localize(datetime(2022, 1, 1, 10, 0, 0)), 100.0),
            ("S1", eastern.localize(datetime(2022, 1, 1, 11, 0, 0)), 101.0),
        ]
        left_df = self.spark.createDataFrame(
            left_data, ["symbol", "timestamp", "value"]
        )

        # Create right data in US/Pacific timezone
        pacific = pytz.timezone('US/Pacific')
        right_data = [
            # These times are actually simultaneous with left times when converted to UTC
            ("S1", pacific.localize(datetime(2022, 1, 1, 7, 0, 0)), 99.5),
            ("S1", pacific.localize(datetime(2022, 1, 1, 8, 0, 0)), 100.5),
        ]
        right_df = self.spark.createDataFrame(
            right_data, ["symbol", "timestamp", "price"]
        )

        # Spark should normalize to session timezone (UTC)
        left_tsdf = TSDF(left_df, ts_col="timestamp", series_ids=["symbol"])
        right_tsdf = TSDF(right_df, ts_col="timestamp", series_ids=["symbol"])

        # Perform join
        joiner = UnionSortFilterAsOfJoiner()
        result = joiner(left_tsdf, right_tsdf)

        # Should match correctly despite different source timezones
        self.assertEqual(result.df.count(), 2)

        # Verify the join matched correctly based on actual time
        rows = result.df.orderBy("left_timestamp").collect()
        for row in rows:
            self.assertIsNotNone(row["right_timestamp"])

    def test_null_timezone_handling(self):
        """Test that null timestamps are handled correctly."""
        # Create test data with some null timestamps
        left_data = [
            ("S1", datetime(2022, 1, 1, 10, 0, 0), 100.0),
            ("S1", None, 101.0),  # Null timestamp
            ("S1", datetime(2022, 1, 1, 12, 0, 0), 102.0),
        ]
        left_df = self.spark.createDataFrame(
            left_data, ["symbol", "timestamp", "value"]
        )

        right_data = [
            ("S1", datetime(2022, 1, 1, 9, 30, 0), 99.5),
            ("S1", datetime(2022, 1, 1, 10, 30, 0), 100.5),
            ("S1", None, 101.5),  # Null timestamp
        ]
        right_df = self.spark.createDataFrame(
            right_data, ["symbol", "timestamp", "price"]
        )

        # Create TSDFs - should handle nulls gracefully
        left_tsdf = TSDF(left_df.filter(F.col("timestamp").isNotNull()),
                        ts_col="timestamp", series_ids=["symbol"])
        right_tsdf = TSDF(right_df.filter(F.col("timestamp").isNotNull()),
                         ts_col="timestamp", series_ids=["symbol"])

        # Perform join
        joiner = BroadcastAsOfJoiner(self.spark)
        result = joiner(left_tsdf, right_tsdf)

        # Should only join non-null timestamps
        self.assertEqual(result.df.count(), 2)  # Only rows with valid timestamps

    def test_dst_transition_handling(self):
        """Test handling of daylight saving time transitions."""
        # Create data around DST transition (Spring forward in US/Eastern)
        # March 13, 2022 at 2:00 AM -> 3:00 AM
        eastern = pytz.timezone('US/Eastern')

        left_data = [
            # Before DST
            ("S1", eastern.localize(datetime(2022, 3, 13, 1, 30, 0)), 100.0),
            # After DST (3:30 AM EDT)
            ("S1", eastern.localize(datetime(2022, 3, 13, 3, 30, 0)), 101.0),
        ]
        left_df = self.spark.createDataFrame(
            left_data, ["symbol", "timestamp", "value"]
        )

        right_data = [
            # Before DST
            ("S1", eastern.localize(datetime(2022, 3, 13, 1, 0, 0)), 99.5),
            # After DST
            ("S1", eastern.localize(datetime(2022, 3, 13, 3, 0, 0)), 100.5),
        ]
        right_df = self.spark.createDataFrame(
            right_data, ["symbol", "timestamp", "price"]
        )

        # Create TSDFs
        left_tsdf = TSDF(left_df, ts_col="timestamp", series_ids=["symbol"])
        right_tsdf = TSDF(right_df, ts_col="timestamp", series_ids=["symbol"])

        # Test both strategies handle DST correctly
        broadcast_joiner = BroadcastAsOfJoiner(self.spark)
        broadcast_result = broadcast_joiner(left_tsdf, right_tsdf)

        union_joiner = UnionSortFilterAsOfJoiner()
        union_result = union_joiner(left_tsdf, right_tsdf)

        # Both should produce consistent results
        self.assertEqual(broadcast_result.df.count(), union_result.df.count())

        # Verify correct matching across DST boundary
        broadcast_rows = broadcast_result.df.orderBy("left_timestamp").collect()
        union_rows = union_result.df.orderBy("left_timestamp").collect()

        # Compare that both strategies produce same matches
        for b_row, u_row in zip(broadcast_rows, union_rows):
            if b_row["right_timestamp"] and u_row["right_timestamp"]:
                # Timestamps should match between strategies
                self.assertEqual(
                    b_row["right_timestamp"],
                    u_row["right_timestamp"],
                    f"Mismatch in DST handling between strategies"
                )