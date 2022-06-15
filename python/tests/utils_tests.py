from pyspark.sql.types import *
from tests.tsdf_tests import SparkTest
from tempo.utils import calculate_time_horizon
from chispa.dataframe_comparer import *
from tempo.tsdf import TSDF
from tempo.utils import *
import unittest


class UtilsTest(SparkTest):
    def buildTestingDataFrame(self):

        # construct dataframes
        self.simple_input_df = self.get_data_as_sdf("simple_data")

        self.simple_input_tsdf = TSDF(
            self.simple_input_df,
            partition_cols=["partition_a", "partition_b"],
            ts_col="event_ts",
        )


class UtilsTest(UtilsTest):
    def test_display(self):
        """Test of the display utility"""
        if IS_DATABRICKS:
            self.assertEqual(id(display), id(display_improvised))
        elif ENV_CAN_RENDER_HTML:
            self.assertEqual(id(display), id(display_html_improvised))
        else:
            self.assertEqual(id(display), id(display_unavailable))

    def test_calculate_time_horizon(self):
        """Test calculate time horizon warning and number of expected output rows"""
        self.buildTestingDataFrame()
        with warnings.catch_warnings(record=True) as w:
            calculate_time_horizon(
                self.simple_input_tsdf.df,
                self.simple_input_tsdf.ts_col,
                "30 seconds",
                ["partition_a", "partition_b"],
            )
            warning_message = """
            Resample Metrics Warning:
                Earliest Timestamp: 2020-01-01 00:00:10
                Latest Timestamp: 2020-01-01 00:05:31
                No. of Unique Partitions: 3
                Resampled Min No. Values in Single a Partition: 7.0
                Resampled Max No. Values in Single a Partition: 12.0
                Resampled P25 No. Values in Single a Partition: 7.0
                Resampled P50 No. Values in Single a Partition: 12.0
                Resampled P75 No. Values in Single a Partition: 12.0
                Resampled Total No. Values Across All Partitions: 31.0
            """
            assert warning_message.strip() == str(w[-1].message).strip()


# MAIN
if __name__ == "__main__":
    unittest.main()
