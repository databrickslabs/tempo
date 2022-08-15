import logging
import unittest

from tests.base import SparkTest


class DeltaWriteTest(SparkTest):

    def test_write_to_delta_without_optimization_cols(self):
        """Test table write to delta format"""

        table_name = "my_table_no_optimization_col"

        # load test data
        input_tsdf = self.get_data_as_tsdf("input_data")

        # test write to delta
        input_tsdf.write(self.spark, table_name)
        logging.info("delta table count ", str(self.spark.table(table_name).count()))

        # should be equal to the expected dataframe
        assert self.spark.table(table_name).count() == 7

    def test_write_to_delta_with_optimization_cols(self):
        """Test table write to delta format"""

        table_name = "my_table_optimization_col"

        # load test data
        input_tsdf = self.get_data_as_tsdf("input_data")

        # test write to delta
        input_tsdf.write(self.spark, table_name, ["date"])
        logging.info("delta table count ", str(self.spark.table(table_name).count()))

        # should be equal to the expected dataframe
        self.assertEqual(self.spark.table(table_name).count(), 7)


# MAIN
if __name__ == "__main__":
    unittest.main()
