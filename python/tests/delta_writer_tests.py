import unittest
import logging

from pyspark.sql.types import *

from tempo.tsdf import TSDF

from tests.base import SparkTest


class DeltaWriteTest(SparkTest):
    def test_write_to_delta(self):
        """Test table write to delta format"""

        # load test data
        tsdf_init = self.get_data_as_tsdf('init')

        # test write to delta
        tsdf_init.write(self.spark, "my_table")
        logging.info("delta table count " + str(self.spark.table("my_table").count()))

        # should be equal to the expected dataframe
        assert self.spark.table("my_table").count() == 7


# MAIN
if __name__ == "__main__":
    unittest.main()
