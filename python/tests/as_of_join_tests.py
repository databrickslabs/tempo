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

    def setUp(self) -> None:
        super().setUp()
        cur_test_case = self.test_data[self.test_name]
        self.left_tsdf = TestDataFrameBuilder(self.spark,
                                              cur_test_case["left"]).as_tsdf()
        self.right_tsdf = TestDataFrameBuilder(self.spark,
                                               cur_test_case["right"]).as_tsdf()
        self.expected_tsdf = TestDataFrameBuilder(self.spark,
                                                  cur_test_case["expected"]).as_tsdf()

    def test_broadcast_join(self):
        joiner = BroadcastAsOfJoiner(self.spark)
        joined_tsdf = joiner(self.left_tsdf, self.right_tsdf)
        joined_tsdf.show()
        self.assertDataFrameEquality(joined_tsdf.df, self.expected_tsdf.df)

    def test_union_sort_filter_join(self):
        joiner = UnionSortFilterAsOfJoiner()
        joined_tsdf = joiner(self.left_tsdf, self.right_tsdf)
        joined_tsdf.show()
        self.assertDataFrameEquality(joined_tsdf.df, self.expected_tsdf.df)
