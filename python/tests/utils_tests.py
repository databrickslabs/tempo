from pyspark.sql.types import *
from tests.tsdf_tests import SparkTest
from tempo.utils import calculate_time_horizon
from chispa.dataframe_comparer import *
from tempo.tsdf import TSDF
from tempo.interpol import Interpolation
from tempo.utils import *
import unittest


class UtilsTest(SparkTest):
    def buildTestingDataFrame(self):
        schema = StructType(
            [
                StructField("partition_a", StringType()),
                StructField("partition_b", StringType()),
                StructField("event_ts", StringType()),
                StructField("value_a", FloatType()),
                StructField("value_b", FloatType()),
            ]
        )

        simple_data = [
            ["A", "A-1", "2020-01-01 00:00:10", 0.0, None],
            ["A", "A-1", "2020-01-01 00:01:10", 2.0, 2.0],
            ["A", "A-1", "2020-01-01 00:01:32", None, None],
            ["A", "A-1", "2020-01-01 00:02:03", None, None],
            ["A", "A-1", "2020-01-01 00:03:32", None, 7.0],
            ["A", "A-1", "2020-01-01 00:04:12", 8.0, 8.0],
            ["A", "A-1", "2020-01-01 00:05:31", 11.0, None],
            ["A", "A-2", "2020-01-01 00:00:10", 0.0, None],
            ["A", "A-2", "2020-01-01 00:01:10", 2.0, 2.0],
            ["A", "A-2", "2020-01-01 00:01:32", None, None],
            ["A", "A-2", "2020-01-01 00:02:03", None, None],
            ["A", "A-2", "2020-01-01 00:04:12", 8.0, 8.0],
            ["A", "A-2", "2020-01-01 00:05:31", 11.0, None],
            ["B", "A-2", "2020-01-01 00:01:10", 2.0, 2.0],
            ["B", "A-2", "2020-01-01 00:01:32", None, None],
            ["B", "A-2", "2020-01-01 00:02:03", None, None],
            ["B", "A-2", "2020-01-01 00:03:32", None, 7.0],
            ["B", "A-2", "2020-01-01 00:04:12", 8.0, 8.0],
        ]

        # construct dataframes
        self.simple_input_df = self.buildTestDF(schema, simple_data)

        self.simple_input_tsdf = TSDF(
            self.simple_input_df,
            partition_cols=["partition_a", "partition_b"],
            ts_col="event_ts",
        )


class UtilsTest(UtilsTest):
    def test_display(self):
        """Test of the display utility"""
        if PLATFORM == "DATABRICKS":
            self.assertEqual(id(display), id(display_improvised))
        elif ENV_BOOLEAN:
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
            print(w[-1].message)

            assert (
                f"""
                    Upsample Metrics Warning: 
                        Earliest Timestamp: 2020-01-01 00:00:10
                        Latest Timestamp: 2020-01-01 00:05:31
                        No. of Unique Partitions: 3
                        Min No. Values in Single a Partition: 7.0
                        Max No. Values in Single a Partition: 12.0
                        P25 No. Values in Single a Partition: 7.0
                        P50 No. Values in Single a Partition: 12.0
                        P75 No. Values in Single a Partition: 12.0
                        Total No. Values Across All Partitions: 31.0
                """
                == str(w[-1].message)
            )


## MAIN
if __name__ == "__main__":
    unittest.main()
