import unittest

from tests.base import SparkTest


class AsOfJoinTest(SparkTest):
    def test_asof_join(self):
        """AS-OF Join with out a time-partition test"""

        # Construct dataframes
        tsdf_left = self.get_data_as_tsdf("left")
        tsdf_right = self.get_data_as_tsdf("right")
        dfExpected = self.get_data_as_sdf("expected")
        noRightPrefixdfExpected = self.get_data_as_sdf("expected_no_right_prefix")

        # perform the join
        joined_df = tsdf_left.asofJoin(
            tsdf_right, left_prefix="left", right_prefix="right"
        ).df
        non_prefix_joined_df = tsdf_left.asofJoin(
            tsdf_right, left_prefix="left", right_prefix=""
        ).df

        # joined dataframe should equal the expected dataframe
        self.assertDataFrameEquality(joined_df, dfExpected)
        self.assertDataFrameEquality(non_prefix_joined_df, noRightPrefixdfExpected)

        spark_sql_joined_df = tsdf_left.asofJoin(
            tsdf_right, left_prefix="left", right_prefix="right"
        ).df
        self.assertDataFrameEquality(spark_sql_joined_df, dfExpected)

    def test_asof_join_skip_nulls_disabled(self):
        """AS-OF Join with skip nulls disabled"""

        # fetch test data
        tsdf_left = self.get_data_as_tsdf("left")
        tsdf_right = self.get_data_as_tsdf("right")
        dfExpectedSkipNulls = self.get_data_as_sdf("expected_skip_nulls")
        dfExpectedSkipNullsDisabled = self.get_data_as_sdf(
            "expected_skip_nulls_disabled"
        )

        # perform the join with skip nulls enabled (default)
        joined_df = tsdf_left.asofJoin(
            tsdf_right, left_prefix="left", right_prefix="right"
        ).df

        # joined dataframe should equal the expected dataframe with nulls skipped
        self.assertDataFrameEquality(joined_df, dfExpectedSkipNulls)

        # perform the join with skip nulls disabled
        joined_df = tsdf_left.asofJoin(
            tsdf_right, left_prefix="left", right_prefix="right", skipNulls=False
        ).df

        # joined dataframe should equal the expected dataframe without nulls skipped
        self.assertDataFrameEquality(joined_df, dfExpectedSkipNullsDisabled)

    def test_sequence_number_sort(self):
        """Skew AS-OF Join with Partition Window Test"""

        # fetch test data
        tsdf_left = self.get_data_as_tsdf("left")
        tsdf_right = self.get_data_as_tsdf("right")
        dfExpected = self.get_data_as_sdf("expected")

        # perform the join
        joined_df = tsdf_left.asofJoin(tsdf_right, right_prefix="right").df

        # joined dataframe should equal the expected dataframe
        self.assertDataFrameEquality(joined_df, dfExpected)

    def test_partitioned_asof_join(self):
        """AS-OF Join with a time-partition"""

        # fetch test data
        tsdf_left = self.get_data_as_tsdf("left")
        tsdf_right = self.get_data_as_tsdf("right")
        dfExpected = self.get_data_as_sdf("expected")

        joined_df = tsdf_left.asofJoin(
            tsdf_right,
            left_prefix="left",
            right_prefix="right",
            tsPartitionVal=10,
            fraction=0.1,
        ).df

        self.assertDataFrameEquality(joined_df, dfExpected)

    def test_asof_join_nanos(self):
        """As of join with nanosecond timestamps"""

        # fetch test data
        tsdf_left = self.get_data_as_tsdf("left")
        tsdf_right = self.get_data_as_tsdf("right")
        dfExpected = self.get_data_as_sdf("expected")

        # perform join
        joined_df = tsdf_left.asofJoin(
            tsdf_right, left_prefix="left", right_prefix="right"
        ).df

        # compare
        self.assertDataFrameEquality(joined_df, dfExpected)

    def test_asof_join_tolerance(self):
        """As of join with tolerance band"""

        # fetch test data
        tsdf_left = self.get_data_as_tsdf("left")
        tsdf_right = self.get_data_as_tsdf("right")
        expected_tolerance_0 = self.get_data_as_sdf("expected_tolerance_0")
        expected_tolerance_10 = self.get_data_as_sdf("expected_tolerance_10")

        # perform join
        joined_df_0 = tsdf_left.asofJoin(
            tsdf_right, left_prefix="left", right_prefix="right", tolerance=0
        ).df
        joined_df_10 = tsdf_left.asofJoin(
            tsdf_right, left_prefix="left", right_prefix="right", tolerance=10
        ).df

        # compare
        self.assertDataFrameEquality(joined_df_0, expected_tolerance_0)
        self.assertDataFrameEquality(joined_df_10, expected_tolerance_10)


# MAIN
if __name__ == "__main__":
    unittest.main()
