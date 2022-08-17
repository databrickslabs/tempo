import logging
import os
import unittest
from unittest import mock
from pyspark.sql.utils import ParseException
from tests.base import SparkTest
from tempo.io import write


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
        """Test table write to delta format with optimizations columns"""

        table_name = "my_table_optimization_col"

        # load test data
        input_tsdf = self.get_data_as_tsdf("input_data")

        # test write to delta
        input_tsdf.write(self.spark, table_name, ["date"])
        logging.info("delta table count ", str(self.spark.table(table_name).count()))

        # should be equal to the expected dataframe
        self.assertEqual(self.spark.table(table_name).count(), 7)

    def test_write_to_delta_non_dbr_environment_logging(self):
        """Test logging when writing"""

        table_name = "my_table_optimization_col"

        # load test data
        input_tsdf = self.get_data_as_tsdf("input_data")

        with self.assertLogs(level="WARNING") as warning_captured:
            # test write to delta
            input_tsdf.write(self.spark, table_name, ["date"])

        self.assertEqual(len(warning_captured.records), 1)
        self.assertEqual(
            warning_captured.output,
            [
                "WARNING:tempo.io:"
                "Delta optimizations attempted on a non-Databricks platform. "
                "Switch to use Databricks Runtime to get optimization advantages."
            ],
        )

    @mock.patch.dict(os.environ, {"DATABRICKS_RUNTIME_VERSION": "10.4"})
    def test_write_to_delta_bad_dbr_environment_logging(self):
        """Test useDeltaOpt Exception"""

        table_name = "my_table_optimization_col_fails"

        # load test data
        input_tsdf = self.get_data_as_tsdf("input_data")

        with self.assertLogs(level="ERROR") as error_captured:
            # test write to delta
            input_tsdf.write(self.spark, table_name, ["date"])

        self.assertEqual(len(error_captured.records), 1)
        print(error_captured.output)
        self.assertEqual(
            error_captured.output,
            [
                "ERROR:tempo.io:"
                "Delta optimizations attempted, but was not successful.\nError: \nmismatched input "
                "'optimize' expecting {'(', 'ADD', 'ALTER', 'ANALYZE', 'CACHE', 'CLEAR', 'COMMENT', 'COMMIT', "
                "'CREATE', 'DELETE', 'DESC', 'DESCRIBE', 'DFS', 'DROP', 'EXPLAIN', 'EXPORT', 'FROM', 'GRANT', "
                "'IMPORT', 'INSERT', 'LIST', 'LOAD', 'LOCK', 'MAP', 'MERGE', 'MSCK', 'REDUCE', 'REFRESH', 'REPLACE', "
                "'RESET', 'REVOKE', 'ROLLBACK', 'SELECT', 'SET', 'SHOW', 'START', 'TABLE', 'TRUNCATE', 'UNCACHE', "
                "'UNLOCK', 'UPDATE', 'USE', 'VALUES', 'WITH'}(line 1, pos 0)\n\n== SQL ==\noptimize "
                "my_table_optimization_col_fails zorder by (symbol,date,event_time)\n^^^\n"
            ],
        )


# MAIN
if __name__ == "__main__":
    unittest.main()
