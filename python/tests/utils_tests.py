from io import StringIO
import sys
import unittest

from tempo.utils import *  # noqa: F403

from tests.tsdf_tests import SparkTest
from unittest import mock


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

    @mock.patch.dict(os.environ, {"TZ": "UTC"})
    def test_calculate_time_horizon(self):
        """Test calculate time horizon warning and number of expected output rows"""

        # fetch test data
        simple_input_tsdf = self.get_data_as_tsdf("simple_input")

        with warnings.catch_warnings(record=True) as w:
            calculate_time_horizon(
                simple_input_tsdf.df,
                simple_input_tsdf.ts_col,
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

    def test_display_html_TSDF(self):
        init_tsdf = self.get_data_as_tsdf("init")

        with self.assertLogs(level="ERROR") as error_captured:
            display_html(init_tsdf)

        self.assertEqual(len(error_captured.records), 1)
        self.assertEqual(
            error_captured.output,
            ["ERROR:tempo.utils:" "'display' method not available for this object"],
        )

    def test_display_html_dataframe(self):
        init_tsdf = self.get_data_as_tsdf("init")

        captured_output = StringIO()
        sys.stdout = captured_output
        display_html(init_tsdf.df)
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
        init_tsdf = self.get_data_as_tsdf("init")
        pandas_dataframe = init_tsdf.df.toPandas()

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
        init_tsdf = self.get_data_as_tsdf("init")

        with self.assertLogs(level="ERROR") as error_captured:
            display_unavailable(init_tsdf)

        self.assertEqual(len(error_captured.records), 1)
        self.assertEqual(
            error_captured.output,
            [
                "ERROR:tempo.utils:"
                "'display' method not available in this environment. Use 'show' method instead."
            ],
        )

    def test_get_display_df(self):
        init_tsdf = self.get_data_as_tsdf("init")
        expected_df = self.get_data_as_sdf("expected")

        actual_df = get_display_df(init_tsdf, 2)

        self.assertDataFrameEquality(actual_df, expected_df)

    def test_get_display_df_sequence_col(self):
        init_tsdf = self.get_data_as_tsdf("init")
        expected_df = self.get_data_as_sdf("expected")

        actual_df = get_display_df(init_tsdf, 2)

        self.assertDataFrameEquality(actual_df, expected_df)


# MAIN
if __name__ == "__main__":
    unittest.main()
