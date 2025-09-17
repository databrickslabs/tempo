"""
Integration tests for as-of join strategies using real Spark DataFrames.

This module tests the join strategies with actual Spark operations using
test data loaded from JSON files, following the established pattern.
"""

import unittest
from parameterized import parameterized

from tests.base import SparkTest

from tempo.joins.strategies import (
    BroadcastAsOfJoiner,
    UnionSortFilterAsOfJoiner,
    SkewAsOfJoiner,
    choose_as_of_join_strategy,
)


class StrategiesIntegrationTest(SparkTest):
    """Integration tests for join strategies with real Spark DataFrames."""

    @parameterized.expand([("basic_join",)])
    def test_broadcast_join_basic(self, test_name):
        """Test BroadcastAsOfJoiner with basic test data."""
        # Load test data
        left_tsdf = self.get_test_df_builder(test_name, "left").as_tsdf()
        right_tsdf = self.get_test_df_builder(test_name, "right").as_tsdf()
        expected_tsdf = self.get_test_df_builder(test_name, "expected_broadcast").as_tsdf()

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
            expected_tsdf.df
        )

    @parameterized.expand([("basic_join",)])
    def test_union_sort_filter_join_basic(self, test_name):
        """Test UnionSortFilterAsOfJoiner with basic test data."""
        # Load test data
        left_tsdf = self.get_test_df_builder(test_name, "left").as_tsdf()
        right_tsdf = self.get_test_df_builder(test_name, "right").as_tsdf()
        expected_tsdf = self.get_test_df_builder(test_name, "expected_broadcast").as_tsdf()

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
            expected_tsdf.df
        )

    @parameterized.expand([("tolerance_test",)])
    def test_tolerance_filtering(self, test_name):
        """Test tolerance filtering functionality."""
        # Load test data
        left_tsdf = self.get_test_df_builder(test_name, "left").as_tsdf()
        right_tsdf = self.get_test_df_builder(test_name, "right").as_tsdf()
        expected_tsdf = self.get_test_df_builder(test_name, "expected_with_tolerance_120").as_tsdf()

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
        )

    @parameterized.expand([("null_handling",)])
    def test_skip_nulls_behavior(self, test_name):
        """Test skipNulls parameter behavior."""
        # Load test data
        left_tsdf = self.get_test_df_builder(test_name, "left").as_tsdf()
        right_tsdf = self.get_test_df_builder(test_name, "right").as_tsdf()
        expected_tsdf = self.get_test_df_builder(test_name, "expected_skip_nulls").as_tsdf()

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
            expected_tsdf.df
        )

    @parameterized.expand([("empty_dataframes",)])
    def test_empty_dataframe_handling(self, test_name):
        """Test handling of empty DataFrames."""
        # Load test data
        empty_tsdf = self.get_test_df_builder(test_name, "empty").as_tsdf()
        non_empty_tsdf = self.get_test_df_builder(test_name, "non_empty").as_tsdf()
        expected_tsdf = self.get_test_df_builder(test_name, "expected_empty_right").as_tsdf()

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
            expected_tsdf.df
        )

        # Test empty left DataFrame
        result_empty_left = joiner(empty_tsdf, non_empty_tsdf)
        self.assertEqual(result_empty_left.df.count(), 0)

    def test_strategy_consistency(self):
        """Test that different strategies produce consistent results."""
        # Load basic join test data
        test_name = "basic_join"
        left_tsdf = self.get_test_df_builder(test_name, "left").as_tsdf()
        right_tsdf = self.get_test_df_builder(test_name, "right").as_tsdf()

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

    @parameterized.expand([("null_lead_regression",)])
    def test_null_lead_regression(self, test_name):
        """
        Regression test for NULL lead bug in BroadcastAsOfJoiner.

        This test ensures that when the last row in a right partition has no
        "next" row (lead is NULL), it still matches with future left timestamps.
        This was a bug where the between condition would fail when comparing with NULL.

        Fixed in PR #XXX (TODO: Update PR number once created)
        """
        # Load test data
        left_tsdf = self.get_test_df_builder(test_name, "left").as_tsdf()
        right_tsdf = self.get_test_df_builder(test_name, "right").as_tsdf()
        expected_tsdf = self.get_test_df_builder(test_name, "expected").as_tsdf()

        # Test with BroadcastAsOfJoiner
        broadcast_joiner = BroadcastAsOfJoiner(
            self.spark,
            left_prefix="",
            right_prefix="right"
        )
        broadcast_result = broadcast_joiner(left_tsdf, right_tsdf)

        # Verify results
        self.assertDataFrameEquality(
            broadcast_result.df,
            expected_tsdf.df
        )

        # Also test with UnionSortFilterAsOfJoiner to ensure consistency
        union_joiner = UnionSortFilterAsOfJoiner(
            left_prefix="",
            right_prefix="right",
            skipNulls=True
        )
        union_result = union_joiner(left_tsdf, right_tsdf)

        # Both strategies should produce the same result
        self.assertDataFrameEquality(
            union_result.df,
            expected_tsdf.df
        )

    def test_automatic_strategy_selection(self):
        """Test automatic strategy selection."""
        # Load test data
        test_name = "basic_join"
        left_tsdf = self.get_test_df_builder(test_name, "left").as_tsdf()
        right_tsdf = self.get_test_df_builder(test_name, "right").as_tsdf()

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