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

        self.expected_schema = StructType(
            [
                StructField("partition_a", StringType()),
                StructField("partition_b", StringType()),
                StructField("event_ts", StringType(), False),
                StructField("value_a", DoubleType()),
                StructField("value_b", DoubleType()),
                StructField("is_ts_interpolated", BooleanType(), False),
                StructField("is_interpolated_value_a", BooleanType(), False),
                StructField("is_interpolated_value_b", BooleanType(), False),
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
                self.simple_input_tsdf.df, self.simple_input_tsdf.ts_col, "30 seconds"
            )
            assert (
                "Time Horizon Warning: The resulting dataframe will contain 12.0 values."
                == str(w[-1].message)
            )


## MAIN
if __name__ == "__main__":
    unittest.main()
