import unittest

import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame

from tests.base import SparkTest


class BasicTests(SparkTest):
    def test_describe(self):
        """AS-OF Join with out a time-partition test"""

        # Construct dataframes
        tsdf_init = self.get_data_as_tsdf("init")

        # generate description dataframe
        res = tsdf_init.describe()

        # joined dataframe should equal the expected dataframe
        # self.assertDataFramesEqual(res, dfExpected)
        assert res.count() == 7
        assert (
            res.filter(F.col("unique_time_series_count") != " ")
            .select(F.max(F.col("unique_time_series_count")))
            .collect()[0][0]
            == "1"
        )
        assert (
            res.filter(F.col("min_ts") != " ")
            .select(F.col("min_ts").cast("string"))
            .collect()[0][0]
            == "2020-08-01 00:00:10"
        )
        assert (
            res.filter(F.col("max_ts") != " ")
            .select(F.col("max_ts").cast("string"))
            .collect()[0][0]
            == "2020-09-01 00:19:12"
        )


class FourierTransformTest(SparkTest):
    def test_fourier_transform(self):
        """Test of fourier transform functionality in TSDF objects"""

        # construct dataframes
        tsdf_init = self.get_data_as_tsdf("init")
        dfExpected = self.get_data_as_sdf("expected")

        # convert to TSDF
        result_tsdf = tsdf_init.fourier_transform(1, "val")

        # should be equal to the expected dataframe
        self.assertDataFramesEqual(result_tsdf.df, dfExpected)


class RangeStatsTest(SparkTest):
    def test_range_stats(self):
        """Test of range stats for 20 minute rolling window"""

        # construct dataframes
        tsdf_init = self.get_data_as_tsdf("init")
        dfExpected = self.get_data_as_sdf("expected")

        # convert to TSDF

        # using lookback of 20 minutes
        featured_df = tsdf_init.withRangeStats(rangeBackWindowSecs=1200).df

        # cast to decimal with precision in cents for simplicity
        featured_df = featured_df.select(
            F.col("symbol"),
            F.col("event_ts"),
            F.col("mean_trade_pr").cast("decimal(5, 2)"),
            F.col("count_trade_pr"),
            F.col("min_trade_pr").cast("decimal(5,2)"),
            F.col("max_trade_pr").cast("decimal(5,2)"),
            F.col("sum_trade_pr").cast("decimal(5,2)"),
            F.col("stddev_trade_pr").cast("decimal(5,2)"),
            F.col("zscore_trade_pr").cast("decimal(5,2)"),
        )

        # cast to decimal with precision in cents for simplicity
        dfExpected = dfExpected.select(
            F.col("symbol"),
            F.col("event_ts"),
            F.col("mean_trade_pr").cast("decimal(5, 2)"),
            F.col("count_trade_pr"),
            F.col("min_trade_pr").cast("decimal(5,2)"),
            F.col("max_trade_pr").cast("decimal(5,2)"),
            F.col("sum_trade_pr").cast("decimal(5,2)"),
            F.col("stddev_trade_pr").cast("decimal(5,2)"),
            F.col("zscore_trade_pr").cast("decimal(5,2)"),
        )

        # should be equal to the expected dataframe
        self.assertDataFramesEqual(featured_df, dfExpected)

    def test_group_stats(self):
        """Test of range stats for 20 minute rolling window"""

        # construct dataframes
        tsdf_init = self.get_data_as_tsdf("init")
        dfExpected = self.get_data_as_sdf("expected")

        # using lookback of 20 minutes
        featured_df = tsdf_init.withGroupedStats(freq="1 min").df

        # cast to decimal with precision in cents for simplicity
        featured_df = featured_df.select(
            F.col("symbol"),
            F.col("event_ts"),
            F.col("mean_trade_pr").cast("decimal(5, 2)"),
            F.col("count_trade_pr"),
            F.col("min_trade_pr").cast("decimal(5,2)"),
            F.col("max_trade_pr").cast("decimal(5,2)"),
            F.col("sum_trade_pr").cast("decimal(5,2)"),
            F.col("stddev_trade_pr").cast("decimal(5,2)"),
        )

        # cast to decimal with precision in cents for simplicity
        dfExpected = dfExpected.select(
            F.col("symbol"),
            F.col("event_ts"),
            F.col("mean_trade_pr").cast("decimal(5, 2)"),
            F.col("count_trade_pr"),
            F.col("min_trade_pr").cast("decimal(5,2)"),
            F.col("max_trade_pr").cast("decimal(5,2)"),
            F.col("sum_trade_pr").cast("decimal(5,2)"),
            F.col("stddev_trade_pr").cast("decimal(5,2)"),
        )

        # should be equal to the expected dataframe
        self.assertDataFramesEqual(featured_df, dfExpected)


class ResampleTest(SparkTest):
    def test_resample(self):
        """Test of range stats for 20 minute rolling window"""

        # construct dataframes
        tsdf_input = self.get_data_as_tsdf("input")
        dfExpected = self.get_data_as_sdf("expected")
        expected_30s_df = self.get_data_as_sdf("expected30m")
        barsExpected = self.get_data_as_sdf("expectedbars")

        # 1 minute aggregation
        featured_df = tsdf_input.resample(freq="min", func="floor", prefix="floor").df
        # 30 minute aggregation
        resample_30m = tsdf_input.resample(freq="5 minutes", func="mean").df.withColumn(
            "trade_pr", F.round(F.col("trade_pr"), 2)
        )

        bars = tsdf_input.calc_bars(
            freq="min", metricCols=["trade_pr", "trade_pr_2"]
        ).df

        # should be equal to the expected dataframe
        self.assertDataFramesEqual(featured_df, dfExpected)
        self.assertDataFramesEqual(resample_30m, expected_30s_df)

        # test bars summary
        self.assertDataFramesEqual(bars, barsExpected)

    def test_resample_millis(self):
        """Test of resampling for millisecond windows"""

        # construct dataframes
        tsdf_init = self.get_data_as_tsdf("init")
        dfExpected = self.get_data_as_sdf("expectedms")

        # 30 minute aggregation
        resample_ms = tsdf_init.resample(freq="ms", func="mean").df.withColumn(
            "trade_pr", F.round(F.col("trade_pr"), 2)
        )

        self.assertDataFramesEqual(resample_ms, dfExpected)

    def test_upsample(self):
        """Test of range stats for 20 minute rolling window"""

        # construct dataframes
        tsdf_input = self.get_data_as_tsdf("input")
        expected_30s_df = self.get_data_as_sdf("expected30m")
        barsExpected = self.get_data_as_sdf("expectedbars")

        resample_30m = tsdf_input.resample(
            freq="5 minutes", func="mean", fill=True
        ).df.withColumn("trade_pr", F.round(F.col("trade_pr"), 2))

        bars = tsdf_input.calc_bars(
            freq="min", metricCols=["trade_pr", "trade_pr_2"]
        ).df

        upsampled = resample_30m.filter(
            F.col("event_ts").isin(
                "2020-08-01 00:00:00",
                "2020-08-01 00:05:00",
                "2020-09-01 00:00:00",
                "2020-09-01 00:15:00",
            )
        )

        # test upsample summary
        self.assertDataFramesEqual(upsampled, expected_30s_df)

        # test bars summary
        self.assertDataFramesEqual(bars, barsExpected)


class extractStateIntervalsTest(SparkTest):
    """Test of finding time ranges for metrics with constant state."""

    def create_expected_test_df(
        self,
        df,
    ) -> DataFrame:
        return (
            # StringType not converting to TimeStamp type inside of struct so forcing
            df.withColumn(
                "event_ts",
                F.struct(
                    F.to_timestamp("event_ts.start").alias("start"),
                    F.to_timestamp("event_ts.end").alias("end"),
                ),
            )
        )

    def test_eq_extractStateIntervals(self):

        # construct dataframes
        input_tsdf = self.get_data_as_tsdf("input")
        expected_df = self.get_data_as_sdf("expected")
        expected_df = self.create_expected_test_df(expected_df)

        # call extractStateIntervals method
        extractStateIntervals_eq_1_df = input_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3"
        ).df
        extractStateIntervals_eq_2_df = input_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition="<=>"
        ).df

        # test extractStateIntervals_tsdf summary
        self.assertDataFramesEqual(extractStateIntervals_eq_1_df, expected_df)
        self.assertDataFramesEqual(extractStateIntervals_eq_2_df, expected_df)

    def test_ne_extractStateIntervals(self):

        # construct dataframes
        input_tsdf = self.get_data_as_tsdf("input")
        expected_df = self.get_data_as_sdf("expected")
        expected_df = self.create_expected_test_df(expected_df)

        # call extractStateIntervals method
        extractStateIntervals_ne_1_df = input_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition="!="
        ).df
        extractStateIntervals_ne_2_df = input_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition="<>"
        ).df

        # test extractStateIntervals_tsdf summary
        self.assertDataFramesEqual(extractStateIntervals_ne_1_df, expected_df)
        self.assertDataFramesEqual(extractStateIntervals_ne_2_df, expected_df)

    def test_gt_extractStateIntervals(self):

        # construct dataframes
        input_tsdf = self.get_data_as_tsdf("input")
        expected_df = self.get_data_as_sdf("expected")
        expected_df = self.create_expected_test_df(expected_df)

        # call extractStateIntervals method
        extractStateIntervals_gt_df = input_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition=">"
        ).df

        self.assertDataFramesEqual(extractStateIntervals_gt_df, expected_df)

    def test_lt_extractStateIntervals(self):
        # construct dataframes
        input_tsdf = self.get_data_as_tsdf("input")
        expected_df = self.get_data_as_sdf("expected")
        expected_df = self.create_expected_test_df(expected_df)

        # call extractStateIntervals method
        extractStateIntervals_lt_df = input_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition="<"
        ).df

        # test extractStateIntervals_tsdf summary
        self.assertDataFramesEqual(extractStateIntervals_lt_df, expected_df)

    def test_gte_extractStateIntervals(self):
        # construct dataframes
        input_tsdf = self.get_data_as_tsdf("input")
        expected_df = self.get_data_as_sdf("expected")
        expected_df = self.create_expected_test_df(expected_df)

        # call extractStateIntervals method
        extractStateIntervals_gt_df = input_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition=">="
        ).df

        self.assertDataFramesEqual(extractStateIntervals_gt_df, expected_df)

    def test_lte_extractStateIntervals(self):

        # construct dataframes
        input_tsdf = self.get_data_as_tsdf("input")
        expected_df = self.get_data_as_sdf("expected")
        expected_df = self.create_expected_test_df(expected_df)

        # call extractStateIntervals method
        extractStateIntervals_lte_df = input_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition="<="
        ).df

        # test extractStateIntervals_tsdf summary
        self.assertDataFramesEqual(extractStateIntervals_lte_df, expected_df)

    def test_bool_col_extractStateIntervals(self):

        # construct dataframes
        input_tsdf = self.get_data_as_tsdf("input")
        expected_df = self.get_data_as_sdf("expected")
        expected_df = self.create_expected_test_df(expected_df)

        # call extractStateIntervals method
        extractStateIntervals_bool_col_df = input_tsdf.extractStateIntervals(
            "metric_1",
            "metric_2",
            "metric_3",
            state_definition=F.abs(
                F.col("metric_1") - F.col("metric_2") - F.col("metric_3")
            )
            < F.lit(10),
        ).df

        # test extractStateIntervals_tsdf summary
        self.assertDataFramesEqual(extractStateIntervals_bool_col_df, expected_df)


# MAIN
if __name__ == "__main__":
    unittest.main()
