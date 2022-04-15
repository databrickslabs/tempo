import unittest
import logging

from pyspark.sql.types import *

from tempo.tsdf import TSDF

from tests.base import SparkTest


class DeltaWriteTest(SparkTest):

    def test_write_to_delta(self):
        """Test table write to delta format"""
        schema = StructType([StructField("symbol", StringType()),
                             StructField("date", StringType()),
                             StructField("event_ts", StringType()),
                             StructField("trade_pr", FloatType()),
                             StructField("trade_pr_2", FloatType())])

        data = [["S1", "SAME_DT", "2020-08-01 00:00:10", 349.21, 10.0],
                ["S1", "SAME_DT", "2020-08-01 00:00:11", 340.21, 9.0],
                ["S1", "SAME_DT", "2020-08-01 00:01:12", 353.32, 8.0],
                ["S1", "SAME_DT", "2020-08-01 00:01:13", 351.32, 7.0],
                ["S1", "SAME_DT", "2020-08-01 00:01:14", 350.32, 6.0],
                ["S1", "SAME_DT", "2020-09-01 00:01:12", 361.1, 5.0],
                ["S1", "SAME_DT", "2020-09-01 00:19:12", 362.1, 4.0]]

        # construct dataframe
        df = self.buildTestDF(schema, data)

        # convert to TSDF
        tsdf_left = TSDF(df, partition_cols=["symbol"], ts_col="event_ts")

        #test write to delta
        tsdf_left.write(self.spark, "my_table")
        logging.info('delta table count ' + str(self.spark.table("my_table").count()))

        # should be equal to the expected dataframe
        assert self.spark.table("my_table").count() == 7


# MAIN
if __name__ == '__main__':
    unittest.main()
