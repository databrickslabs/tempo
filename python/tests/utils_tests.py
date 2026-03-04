import sys
import unittest
import warnings
from io import StringIO
from unittest.mock import patch

from tempo.resample import calculate_time_horizon
from tempo.utils import *  # noqa: F403
from tests.base import SparkTest


class UtilsTest(SparkTest):
    def test_display(self):
        # TODO: we shouldn't have conditions in test cases
        #  the first and second branch are never reached when testing
        #  split out & mock appropriately to test each
        """Test of the display utility"""
        if IS_DATABRICKS:
            self.assertEqual(id(display), id(display_improvised))
        elif ENV_CAN_RENDER_HTML:
            self.assertEqual(id(display), id(display_html_improvised))
        else:
            self.assertEqual(id(display), id(display_unavailable))

    @patch.dict(os.environ, {"TZ": "UTC"})
    def test_calculate_time_horizon(self):
        """Test calculate time horizon warning and number of expected output rows"""

        # fetch test data
        tsdf = self.get_test_function_df_builder("init").as_tsdf()

        with warnings.catch_warnings(record=True) as w:
            calculate_time_horizon(tsdf, "30 seconds")
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

    def test_display_html_TSDF(self):
        tsdf = self.get_test_function_df_builder("init").as_tsdf()

        with self.assertLogs(level="ERROR") as error_captured:
            display_html(tsdf)

        self.assertEqual(len(error_captured.records), 1)
        self.assertEqual(
            error_captured.output,
            ["ERROR:tempo.utils:" "'display' method not available for this object"],
        )

    def test_display_html_dataframe(self):
        sdf = self.get_test_function_df_builder("init").as_sdf()

        captured_output = StringIO()
        sys.stdout = captured_output
        display_html(sdf)
        self.assertEqual(
            captured_output.getvalue(),
            (
                "<IPython.core.display.HTML object>\n"
                "+------+-------------------+--------+\n"
                "|symbol|event_ts           |trade_pr|\n"
                "+------+-------------------+--------+\n"
                "|S1    |2020-08-01 00:00:10|349.21  |\n"
                "|S1    |2020-08-01 00:01:12|351.32  |\n"
                "|S1    |2020-09-01 00:02:10|361.1   |\n"
                "|S1    |2020-09-01 00:19:12|362.1   |\n"
                "|S2    |2020-08-01 00:01:10|743.01  |\n"
                "|S2    |2020-08-01 00:01:24|751.92  |\n"
                "|S2    |2020-09-01 00:02:10|761.1   |\n"
                "|S2    |2020-09-01 00:20:42|762.33  |\n"
                "+------+-------------------+--------+\n"
                "\n"
            ),
        )

    def test_display_html_pandas_dataframe(self):
        sdf = self.get_test_function_df_builder("init").as_sdf()
        pandas_dataframe = sdf.toPandas()

        captured_output = StringIO()
        sys.stdout = captured_output
        display_html(pandas_dataframe)
        self.assertEqual(
            captured_output.getvalue(),
            (
                "<IPython.core.display.HTML object>\n"
                "  symbol            event_ts    trade_pr\n"
                "0     S1 2020-08-01 00:00:10  349.209991\n"
                "1     S1 2020-08-01 00:01:12  351.320007\n"
                "2     S1 2020-09-01 00:02:10  361.100006\n"
                "3     S1 2020-09-01 00:19:12  362.100006\n"
                "4     S2 2020-08-01 00:01:10  743.010010\n"
            ),
        )

    def test_display_unavailable(self):
        with self.assertLogs(level="ERROR") as error_captured:
            display_unavailable()

        self.assertEqual(len(error_captured.records), 1)
        self.assertEqual(
            error_captured.output,
            [
                "ERROR:tempo.utils:"
                "'display' method not available in this environment. Use 'show' method instead."
            ],
        )

    def test_get_display_df(self):
        init = self.get_test_function_df_builder("init").as_tsdf()
        expected_df = self.get_test_function_df_builder("expected").as_sdf()

        actual_df = get_display_df(init, 2)

        self.assertDataFrameEquality(actual_df, expected_df)

    def test_get_display_df_sequence_col(self):
        init = self.get_test_function_df_builder("init").as_tsdf()
        expected_df = self.get_test_function_df_builder("expected").as_sdf()

        actual_df = get_display_df(init, 2)

        self.assertDataFrameEquality(actual_df, expected_df)

    def test_time_range_with_step_size(self):
        """Test time_range with start_time, end_time, and step_size."""
        # Lines 63-97: time_range function with step_size provided
        from datetime import datetime, timedelta
        from tempo.utils import time_range

        start = datetime(2024, 1, 1, 10, 0, 0)
        end = datetime(2024, 1, 1, 10, 10, 0)
        step = timedelta(minutes=2)

        result = time_range(self.spark, start, end, step)

        # Should have 5 intervals (0, 2, 4, 6, 8 minutes)
        self.assertEqual(result.count(), 5)
        self.assertIn("ts", result.columns)

    def test_time_range_with_num_intervals(self):
        """Test time_range with start_time, end_time, and num_intervals."""
        # Lines 63-69: time_range computing step_size from num_intervals
        from datetime import datetime
        from tempo.utils import time_range

        start = datetime(2024, 1, 1, 10, 0, 0)
        end = datetime(2024, 1, 1, 10, 10, 0)
        num_intervals = 5

        result = time_range(self.spark, start, end, num_intervals=num_intervals)

        # Should have exactly 5 intervals
        self.assertEqual(result.count(), 5)

    def test_time_range_compute_num_intervals(self):
        """Test time_range computing num_intervals from end_time and step_size."""
        # Lines 72-78: time_range computing num_intervals
        from datetime import datetime, timedelta
        from tempo.utils import time_range

        start = datetime(2024, 1, 1, 10, 0, 0)
        end = datetime(2024, 1, 1, 10, 11, 0)
        step = timedelta(minutes=2)

        result = time_range(self.spark, start, end, step_size=step)

        # 11 minutes / 2 minute step = 6 intervals (ceiling)
        self.assertEqual(result.count(), 6)

    def test_time_range_with_custom_column_name(self):
        """Test time_range with custom timestamp column name."""
        # Lines 87-90: custom ts_colname parameter
        from datetime import datetime, timedelta
        from tempo.utils import time_range

        start = datetime(2024, 1, 1, 10, 0, 0)
        step = timedelta(minutes=5)
        num_intervals = 3

        result = time_range(
            self.spark,
            start,
            step_size=step,
            num_intervals=num_intervals,
            ts_colname="custom_ts",
        )

        self.assertEqual(result.count(), 3)
        self.assertIn("custom_ts", result.columns)
        self.assertNotIn("ts", result.columns)
        self.assertNotIn("id", result.columns)  # id column should be dropped

    def test_time_range_with_interval_ends(self):
        """Test time_range with include_interval_ends=True."""
        # Lines 91-96: include_interval_ends parameter
        from datetime import datetime, timedelta
        from tempo.utils import time_range

        start = datetime(2024, 1, 1, 10, 0, 0)
        step = timedelta(minutes=5)
        num_intervals = 3

        result = time_range(
            self.spark,
            start,
            step_size=step,
            num_intervals=num_intervals,
            include_interval_ends=True,
        )

        self.assertEqual(result.count(), 3)
        self.assertIn("ts", result.columns)
        self.assertIn("ts_interval_end", result.columns)


# MAIN
if __name__ == "__main__":
    unittest.main()
