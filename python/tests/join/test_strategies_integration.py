"""
Integration tests for as-of join strategies using real Spark DataFrames.

This module tests the join strategies with actual Spark operations using
test data loaded from JSON files, following the established pattern.
"""

import os
import unittest
from parameterized import parameterized

from tests.base import TestDataFrame

# Import strategy implementations (handle circular import)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tempo.joins.strategies import (
    BroadcastAsOfJoiner,
    UnionSortFilterAsOfJoiner,
    SkewAsOfJoiner,
    choose_as_of_join_strategy,
)


class StrategiesIntegrationTest(TestDataFrame):
    """Integration tests for join strategies with real Spark DataFrames."""

    @classmethod
    def get_test_data_dir(cls):
        """Get the test data directory for join tests."""
        return "join"

    @classmethod
    def get_test_data_file(cls):
        """Get the test data file for integration tests."""
        return "strategies_integration.json"

    @parameterized.expand(["basic_join"])
    def test_broadcast_join_basic(self, test_name):
        """Test BroadcastAsOfJoiner with basic test data."""
        # Load test data
        left_tsdf = self.get_test_tsdf(test_name, "left")
        right_tsdf = self.get_test_tsdf(test_name, "right")
        expected_tsdf = self.get_test_tsdf(test_name, "expected_broadcast")

        # Create and execute broadcast join
        joiner = BroadcastAsOfJoiner(
            self.spark,
            left_prefix="",
            right_prefix="right"
        )

        # Execute join
        result = joiner(left_tsdf, right_tsdf)

        # Compare with expected results
        self.assertDataFrameEquality(
            result.df,
            expected_tsdf.df,
            from_tsdf=True,
            order_by_cols=["symbol", "timestamp"]
        )

    @parameterized.expand(["basic_join"])
    def test_union_sort_filter_join_basic(self, test_name):
        """Test UnionSortFilterAsOfJoiner with basic test data."""
        # Load test data
        left_tsdf = self.get_test_tsdf(test_name, "left")
        right_tsdf = self.get_test_tsdf(test_name, "right")
        expected_tsdf = self.get_test_tsdf(test_name, "expected_broadcast")

        # Create and execute union-sort-filter join
        joiner = UnionSortFilterAsOfJoiner(
            left_prefix="",
            right_prefix="right",
            skipNulls=True
        )

        # Execute join
        result = joiner(left_tsdf, right_tsdf)

        # Compare with expected results
        self.assertDataFrameEquality(
            result.df,
            expected_tsdf.df,
            from_tsdf=True,
            order_by_cols=["symbol", "timestamp"]
        )

    @parameterized.expand(["tolerance_test"])
    def test_tolerance_filtering(self, test_name):
        """Test tolerance filtering functionality."""
        # Load test data
        left_tsdf = self.get_test_tsdf(test_name, "left")
        right_tsdf = self.get_test_tsdf(test_name, "right")
        expected_tsdf = self.get_test_tsdf(test_name, "expected_with_tolerance_120")

        # Create joiner with 120 second tolerance
        joiner = UnionSortFilterAsOfJoiner(
            left_prefix="",
            right_prefix="right",
            skipNulls=True,
            tolerance=120  # 2 minutes
        )

        # Execute join
        result = joiner(left_tsdf, right_tsdf)

        # Compare with expected results
        self.assertDataFrameEquality(
            result.df,
            expected_tsdf.df,
            from_tsdf=True,
            order_by_cols=["id", "timestamp"]
        )

    @parameterized.expand(["null_handling"])
    def test_skip_nulls_behavior(self, test_name):
        """Test skipNulls parameter behavior."""
        # Load test data
        left_tsdf = self.get_test_tsdf(test_name, "left")
        right_tsdf = self.get_test_tsdf(test_name, "right")
        expected_tsdf = self.get_test_tsdf(test_name, "expected_skip_nulls")

        # Test with skipNulls=True
        joiner_skip = UnionSortFilterAsOfJoiner(
            left_prefix="",
            right_prefix="right",
            skipNulls=True
        )

        result_skip = joiner_skip(left_tsdf, right_tsdf)

        # Compare with expected results
        self.assertDataFrameEquality(
            result_skip.df,
            expected_tsdf.df,
            from_tsdf=True,
            order_by_cols=["symbol", "timestamp"]
        )

    @parameterized.expand(["empty_dataframes"])
    def test_empty_dataframe_handling(self, test_name):
        """Test handling of empty DataFrames."""
        # Load test data
        empty_tsdf = self.get_test_tsdf(test_name, "empty")
        non_empty_tsdf = self.get_test_tsdf(test_name, "non_empty")
        expected_tsdf = self.get_test_tsdf(test_name, "expected_empty_right")

        # Test empty right DataFrame
        joiner = UnionSortFilterAsOfJoiner(
            left_prefix="",
            right_prefix="right",
            skipNulls=True
        )

        # Join non-empty left with empty right
        result = joiner(non_empty_tsdf, empty_tsdf)

        # Should return left DataFrame with null right columns
        self.assertDataFrameEquality(
            result.df,
            expected_tsdf.df,
            from_tsdf=True,
            order_by_cols=["symbol", "timestamp"]
        )

        # Test empty left DataFrame
        result_empty_left = joiner(empty_tsdf, non_empty_tsdf)
        self.assertEqual(result_empty_left.df.count(), 0)

    def test_strategy_consistency(self):
        """Test that different strategies produce consistent results."""
        # Load basic join test data
        test_name = "basic_join"
        left_tsdf = self.get_test_tsdf(test_name, "left")
        right_tsdf = self.get_test_tsdf(test_name, "right")

        # Execute with broadcast strategy
        broadcast_joiner = BroadcastAsOfJoiner(
            self.spark,
            left_prefix="",
            right_prefix="right"
        )
        broadcast_result = broadcast_joiner(left_tsdf, right_tsdf)

        # Execute with union-sort-filter strategy
        union_joiner = UnionSortFilterAsOfJoiner(
            left_prefix="",
            right_prefix="right",
            skipNulls=True
        )
        union_result = union_joiner(left_tsdf, right_tsdf)

        # Both strategies should produce the same result
        broadcast_data = broadcast_result.df.orderBy("symbol", "timestamp").collect()
        union_data = union_result.df.orderBy("symbol", "timestamp").collect()

        # Verify same number of rows
        self.assertEqual(len(broadcast_data), len(union_data))

        # Verify same data (allowing for minor differences)
        for b_row, u_row in zip(broadcast_data, union_data):
            self.assertEqual(b_row["symbol"], u_row["symbol"])
            self.assertEqual(b_row["timestamp"], u_row["timestamp"])
            self.assertEqual(b_row["price"], u_row["price"])
            self.assertEqual(b_row["volume"], u_row["volume"])

    def test_automatic_strategy_selection(self):
        """Test automatic strategy selection."""
        # Load test data
        test_name = "basic_join"
        left_tsdf = self.get_test_tsdf(test_name, "left")
        right_tsdf = self.get_test_tsdf(test_name, "right")

        # Test automatic selection without optimization
        strategy_default = choose_as_of_join_strategy(
            left_tsdf,
            right_tsdf,
            sql_join_opt=False
        )
        self.assertIsInstance(strategy_default, UnionSortFilterAsOfJoiner)

        # Test with skew handling
        strategy_skew = choose_as_of_join_strategy(
            left_tsdf,
            right_tsdf,
            tsPartitionVal=3600
        )
        self.assertIsInstance(strategy_skew, SkewAsOfJoiner)


if __name__ == "__main__":
    unittest.main()