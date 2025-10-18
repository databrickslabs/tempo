"""
Integration tests for as-of join strategies using real Spark DataFrames.

This module tests the join strategies with actual Spark operations using
test data loaded from JSON files, following the established pattern.
"""

import unittest
from tests.base import SparkTest

from tempo.joins.strategies import (
    BroadcastAsOfJoiner,
    UnionSortFilterAsOfJoiner,
    SkewAsOfJoiner,
    choose_as_of_join_strategy,
)


class StrategiesIntegrationTest(SparkTest):
    """Integration tests for join strategies with real Spark DataFrames."""

    def test_broadcast_join_basic(self):
        """Test BroadcastAsOfJoiner with basic test data."""
        # Load test data using function-based pattern
        left_tsdf = self.get_test_function_df_builder("left").as_tsdf()
        right_tsdf = self.get_test_function_df_builder("right").as_tsdf()
        expected_tsdf = self.get_test_function_df_builder("expected_broadcast").as_tsdf()

        # Create and execute broadcast join
        joiner = BroadcastAsOfJoiner(
            self.spark,
            left_prefix="",
            right_prefix="right"
        )

        # Execute join - returns (DataFrame, TSSchema) tuple
        result_df, result_schema = joiner(left_tsdf, right_tsdf)

        # Compare with expected results
        self.assertDataFrameEquality(
            result_df,
            expected_tsdf.df,
            ignore_row_order=True
        )

    def test_union_sort_filter_join_basic(self):
        """Test UnionSortFilterAsOfJoiner with basic test data."""
        # Load test data using function-based pattern
        left_tsdf = self.get_test_function_df_builder("left").as_tsdf()
        right_tsdf = self.get_test_function_df_builder("right").as_tsdf()
        expected_tsdf = self.get_test_function_df_builder("expected_union").as_tsdf()

        # Create and execute union-sort-filter join
        joiner = UnionSortFilterAsOfJoiner(
            left_prefix="",
            right_prefix="right",
            skipNulls=True
        )

        # Execute join - returns (DataFrame, TSSchema) tuple
        result_df, result_schema = joiner(left_tsdf, right_tsdf)

        # Compare with expected results
        self.assertDataFrameEquality(
            result_df,
            expected_tsdf.df,
            ignore_row_order=True
        )

    def test_tolerance_filtering(self):
        """Test tolerance parameter filtering."""
        # Load test data using function-based pattern
        left_tsdf = self.get_test_function_df_builder("left").as_tsdf()
        right_tsdf = self.get_test_function_df_builder("right").as_tsdf()
        expected_tsdf = self.get_test_function_df_builder("expected_tolerance_120").as_tsdf()

        # Create joiner with tolerance
        joiner = UnionSortFilterAsOfJoiner(
            left_prefix="",
            right_prefix="right",
            skipNulls=True,
            tolerance=120  # 2 minutes tolerance
        )

        # Execute join - returns (DataFrame, TSSchema) tuple
        result_df, result_schema = joiner(left_tsdf, right_tsdf)

        # Compare with expected results
        self.assertDataFrameEquality(
            result_df,
            expected_tsdf.df,
            ignore_row_order=True
        )

    def test_skip_nulls_behavior(self):
        """Test skipNulls parameter behavior."""
        # Load test data using function-based pattern
        left_tsdf = self.get_test_function_df_builder("left").as_tsdf()
        right_tsdf = self.get_test_function_df_builder("right").as_tsdf()

        # Test with skipNulls=True
        expected_tsdf = self.get_test_function_df_builder("expected_skip_nulls_true").as_tsdf()
        joiner = UnionSortFilterAsOfJoiner(
            left_prefix="",
            right_prefix="right",
            skipNulls=True
        )
        result_df, result_schema = joiner(left_tsdf, right_tsdf)
        self.assertDataFrameEquality(
            result_df,
            expected_tsdf.df,
            ignore_row_order=True
        )

        # Test with skipNulls=False
        expected_tsdf = self.get_test_function_df_builder("expected_skip_nulls_false").as_tsdf()
        joiner = UnionSortFilterAsOfJoiner(
            left_prefix="",
            right_prefix="right",
            skipNulls=False
        )
        result_df, result_schema = joiner(left_tsdf, right_tsdf)
        self.assertDataFrameEquality(
            result_df,
            expected_tsdf.df,
            ignore_row_order=True
        )

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        # Test empty left DataFrame
        left_tsdf = self.get_test_function_df_builder("left").as_tsdf()
        right_tsdf = self.get_test_function_df_builder("right").as_tsdf()
        expected_tsdf = self.get_test_function_df_builder("expected").as_tsdf()

        joiner = UnionSortFilterAsOfJoiner(
            left_prefix="",
            right_prefix="right",
            skipNulls=True
        )

        result_df, result_schema = joiner(left_tsdf, right_tsdf)

        # Empty left should result in empty output
        self.assertEqual(result_df.count(), 0)
        self.assertEqual(result_df.count(), expected_tsdf.df.count())

    def test_null_lead_regression(self):
        """
        Regression test for NULL lead bug in BroadcastAsOfJoiner.
        Tests the fix for when the last row in a partition has NULL lead value.
        See PR #XXX for details.
        """
        left_tsdf = self.get_test_function_df_builder("left").as_tsdf()
        right_tsdf = self.get_test_function_df_builder("right").as_tsdf()
        expected_tsdf = self.get_test_function_df_builder("expected").as_tsdf()

        # Create and execute broadcast join
        joiner = BroadcastAsOfJoiner(
            self.spark,
            left_prefix="",
            right_prefix="right"
        )

        # Execute join - should not fail with NULL lead
        result_df, result_schema = joiner(left_tsdf, right_tsdf)

        # All left rows should be preserved
        self.assertEqual(result_df.count(), left_tsdf.df.count())

        # Compare with expected results
        self.assertDataFrameEquality(
            result_df,
            expected_tsdf.df,
            ignore_row_order=True
        )

    def test_strategy_consistency(self):
        """Test that different strategies produce consistent results for the same data."""
        left_tsdf = self.get_test_function_df_builder("left").as_tsdf()
        right_tsdf = self.get_test_function_df_builder("right").as_tsdf()

        # Test broadcast join
        broadcast_joiner = BroadcastAsOfJoiner(
            self.spark,
            left_prefix="",
            right_prefix="right"
        )
        broadcast_result_df, broadcast_result_schema = broadcast_joiner(left_tsdf, right_tsdf)

        # Test union-sort-filter join
        union_joiner = UnionSortFilterAsOfJoiner(
            left_prefix="",
            right_prefix="right",
            skipNulls=True
        )
        union_result_df, union_result_schema = union_joiner(left_tsdf, right_tsdf)

        # Results should be identical
        self.assertDataFrameEquality(
            broadcast_result_df,
            union_result_df,
            ignore_row_order=True
        )

    def test_automatic_strategy_selection(self):
        """Test automatic strategy selection."""
        # Load test data
        left_tsdf = self.get_test_function_df_builder("left").as_tsdf()
        right_tsdf = self.get_test_function_df_builder("right").as_tsdf()

        # Test automatic strategy selection (should potentially select broadcast for small data)
        strategy = choose_as_of_join_strategy(
            left_tsdf,
            right_tsdf
        )

        # Execute join
        result_df, result_schema = strategy(left_tsdf, right_tsdf)

        # Verify result is valid
        self.assertIsNotNone(result_df)
        self.assertEqual(result_df.count(), left_tsdf.df.count())

        # Test with tsPartitionVal (should select SkewAsOfJoiner)
        strategy = choose_as_of_join_strategy(
            left_tsdf,
            right_tsdf,
            tsPartitionVal=300  # 5 minutes
        )

        self.assertIsInstance(strategy, SkewAsOfJoiner)