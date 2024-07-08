import unittest
from unittest.mock import patch

from tests.base import SparkTest


class AsOfJoinTest(SparkTest):
    def test_asof_join(self):
        """AS-OF Join without a time-partition test"""

        # Construct dataframes
        tsdf_left = self.get_test_df_builder("left").as_tsdf()
        tsdf_right = self.get_test_df_builder("right").as_tsdf()
        df_expected = self.get_test_df_builder("expected").as_sdf()
        no_right_prefixdf_expected = self.get_test_df_builder("expected_no_right_prefix").as_sdf()

        # perform the join
        joined_df = tsdf_left.asofJoin(
            tsdf_right, left_prefix="left", right_prefix="right"
        ).df
        non_prefix_joined_df = tsdf_left.asofJoin(
            tsdf_right, left_prefix="left", right_prefix=""
        ).df

        # joined dataframe should equal the expected dataframe
        self.assertDataFrameEquality(joined_df, df_expected)
        self.assertDataFrameEquality(non_prefix_joined_df, no_right_prefixdf_expected)

        spark_sql_joined_df = tsdf_left.asofJoin(
            tsdf_right, left_prefix="left", right_prefix="right"
        ).df
        self.assertDataFrameEquality(spark_sql_joined_df, df_expected)

    def test_asof_join_skip_nulls_disabled(self):
        """AS-OF Join with skip nulls disabled"""

        # fetch test data
        tsdf_left = self.get_test_df_builder("left").as_tsdf()
        tsdf_right = self.get_test_df_builder("right").as_tsdf()
        df_expected_skip_nulls = self.get_test_df_builder("expected_skip_nulls").as_sdf()
        df_expected_skip_nulls_disabled = self.get_test_df_builder(
            "expected_skip_nulls_disabled"
        ).as_sdf()

        # perform the join with skip nulls enabled (default)
        joined_df = tsdf_left.asofJoin(
            tsdf_right, left_prefix="left", right_prefix="right"
        ).df

        # joined dataframe should equal the expected dataframe with nulls skipped
        self.assertDataFrameEquality(joined_df, df_expected_skip_nulls)

        # perform the join with skip nulls disabled
        joined_df = tsdf_left.asofJoin(
            tsdf_right, left_prefix="left", right_prefix="right", skipNulls=False
        ).df

        # joined dataframe should equal the expected dataframe without nulls skipped
        self.assertDataFrameEquality(joined_df, df_expected_skip_nulls_disabled)

    def test_sequence_number_sort(self):
        """Skew AS-OF Join with Partition Window Test"""

        # fetch test data
        tsdf_left = self.get_test_df_builder("left").as_tsdf()
        tsdf_right = self.get_test_df_builder("right").as_tsdf()
        df_expected = self.get_test_df_builder("expected").as_sdf()

        # perform the join
        joined_df = tsdf_left.asofJoin(tsdf_right, right_prefix="right").df

        # joined dataframe should equal the expected dataframe
        self.assertDataFrameEquality(joined_df, df_expected)

    def test_partitioned_asof_join(self):
        """AS-OF Join with a time-partition"""
        with self.assertLogs(level="WARNING") as warning_captured:
            # fetch test data
            tsdf_left = self.get_test_df_builder("left").as_tsdf()
            tsdf_right = self.get_test_df_builder("right").as_tsdf()
            df_expected = self.get_test_df_builder("expected").as_sdf()

            joined_df = tsdf_left.asofJoin(
                tsdf_right,
                left_prefix="left",
                right_prefix="right",
                tsPartitionVal=10,
                fraction=0.1,
            ).df

            self.assertDataFrameEquality(joined_df, df_expected)
            self.assertEqual(
                warning_captured.output,
                [
                    "WARNING:tempo.tsdf:You are using the skew version of the AS OF join. This "
                    "may result in null values if there are any values outside of the maximum "
                    "lookback. For maximum efficiency, choose smaller values of maximum lookback, "
                    "trading off performance and potential blank AS OF values for sparse keys"
                ],
            )

    def test_asof_join_nanos(self):
        """As of join with nanosecond timestamps"""

        # fetch test data
        tsdf_left = self.get_test_df_builder("left").as_tsdf()
        tsdf_right = self.get_test_df_builder("right").as_tsdf()
        dfExpected = self.get_test_df_builder("expected").as_sdf()

        # perform join
        joined_df = tsdf_left.asofJoin(
            tsdf_right, left_prefix="left", right_prefix="right"
        ).df

        joined_df.show()

        # compare
        self.assertDataFrameEquality(joined_df, dfExpected)

    def test_asof_join_tolerance(self):
        """As of join with tolerance band"""

        # fetch test data
        tsdf_left = self.get_test_df_builder("left").as_tsdf()
        tsdf_right = self.get_test_df_builder("right").as_tsdf()

        tolerance_test_values = [None, 0, 5.5, 7, 10]
        for tolerance in tolerance_test_values:
            # perform join
            joined_df = tsdf_left.asofJoin(
                tsdf_right,
                left_prefix="left",
                right_prefix="right",
                tolerance=tolerance,
            ).df

            # compare
            expected_tolerance = self.get_test_df_builder(f"expected_tolerance_{tolerance}").as_sdf()
            self.assertDataFrameEquality(joined_df, expected_tolerance)

    def test_asof_join_sql_join_opt_and_bytes_threshold(self):
        """AS-OF Join with out a time-partition test"""
        with patch("tempo.tsdf.TSDF._TSDF__getBytesFromPlan", return_value=1000):
            # Construct dataframes
            tsdf_left = self.get_test_df_builder("left").as_tsdf()
            tsdf_right = self.get_test_df_builder("right").as_tsdf()
            df_expected = self.get_test_df_builder("expected").as_sdf()
            no_right_prefixdf_expected = self.get_test_df_builder("expected_no_right_prefix").as_sdf()

            # perform the join
            joined_df = tsdf_left.asofJoin(
                tsdf_right, left_prefix="left", right_prefix="right", sql_join_opt=True
            ).df
            non_prefix_joined_df = tsdf_left.asofJoin(
                tsdf_right, left_prefix="left", right_prefix="", sql_join_opt=True
            ).df

            # joined dataframe should equal the expected dataframe
            self.assertDataFrameEquality(joined_df, df_expected)
            self.assertDataFrameEquality(non_prefix_joined_df, no_right_prefixdf_expected)

            spark_sql_joined_df = tsdf_left.asofJoin(
                tsdf_right, left_prefix="left", right_prefix="right"
            ).df
            self.assertDataFrameEquality(spark_sql_joined_df, df_expected)


# MAIN
if __name__ == "__main__":
    unittest.main()
