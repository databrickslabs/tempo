import logging
import unittest
from importlib.metadata import version

from packaging import version as pkg_version

from tests.base import SparkTest

DELTA_VERSION = version("delta-spark")


class DeltaWriteTest(SparkTest):
    def test_write_to_delta_without_optimization_cols(self):
        """Test table write to delta format"""

        table_name = "my_table_no_optimization_col"

        # load test data
        input_tsdf = self.get_test_df_builder("init").as_tsdf()

        # test write to delta
        input_tsdf.write(self.spark, table_name)
        logging.info("delta table count ", str(self.spark.table(table_name).count()))

        # should be equal to the expected dataframe
        assert self.spark.table(table_name).count() == 7

    def test_write_to_delta_with_optimization_cols(self):
        """Test table write to delta format with optimizations columns"""

        table_name = "my_table_optimization_col"

        # load test data
        input_tsdf = self.get_test_df_builder("init").as_tsdf()

        # test write to delta
        input_tsdf.write(self.spark, table_name, ["date"])
        logging.info("delta table count ", str(self.spark.table(table_name).count()))

        # should be equal to the expected dataframe
        self.assertEqual(self.spark.table(table_name).count(), 7)

    def test_write_to_delta_bad_dbr_environment_logging(self):
        """Test useDeltaOpt Exception"""

        table_name = "my_table_optimization_col_fails"

        # load test data
        input_tsdf = self.get_test_df_builder("init").as_tsdf()

        if pkg_version.parse(DELTA_VERSION) < pkg_version.parse("2.0.0"):

            with self.assertLogs(level="ERROR") as error_captured:
                # should fail to run optimize
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
        else:
            pass


# MAIN
if __name__ == "__main__":
    unittest.main()
