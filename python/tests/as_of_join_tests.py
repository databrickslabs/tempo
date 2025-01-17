from parameterized import parameterized_class

from tempo.as_of_join import BroadcastAsOfJoiner, UnionSortFilterAsOfJoiner

from tests.base import SparkTest, TestDataFrameBuilder


@parameterized_class(
    ("test_name",),
    [
        ("simple_ts",),
        ("nanos",)
    ]
)
class AsOfJoinTest(SparkTest):

    def test_broadcast_join(self):
        # set up dataframes
        left_tsdf = self.get_test_df_builder(self.test_name, "left").as_tsdf()
        right_tsdf = self.get_test_df_builder(self.test_name, "right").as_tsdf()
        expected_tsdf = self.get_test_df_builder(self.test_name, "expected").as_tsdf()

        # perform join
        joiner = BroadcastAsOfJoiner(self.spark)
        joined_tsdf = joiner(left_tsdf, right_tsdf)
        joined_tsdf.show()

        # check that it matches expectations
        self.assertDataFrameEquality(joined_tsdf.df, expected_tsdf.df)

    def test_union_sort_filter_join(self):
        # set up dataframes
        left_tsdf = self.get_test_df_builder(self.test_name, "left").as_tsdf()
        right_tsdf = self.get_test_df_builder(self.test_name, "right").as_tsdf()
        expected_tsdf = self.get_test_df_builder(self.test_name, "expected").as_tsdf()

        # perform join
        joiner = UnionSortFilterAsOfJoiner()
        joined_tsdf = joiner(left_tsdf, right_tsdf)
        joined_tsdf.show()

        # check that it matches expectations
        self.assertDataFrameEquality(joined_tsdf.df, expected_tsdf.df)
