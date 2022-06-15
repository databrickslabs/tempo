import unittest

import pyspark.sql.functions as F

from tempo.tsdf import TSDF

from tests.base import SparkTest


class BasicTests(SparkTest):
    def test_describe(self):
        """AS-OF Join with out a time-partition test"""

        dfLeft = self.get_data_as_sdf("left")

        # perform the join
        tsdf_left = TSDF(dfLeft, ts_col="event_ts", partition_cols=["symbol"])
        res = tsdf_left.describe()

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

    def test_at(self):
        pass  # TODO

    def test_before(self):
        pass  # TODO

    def test_atOrBefore(self):
        pass  # TODO

    def test_after(self):
        pass  # TODO

    def test_atOrAfter(self):
        pass  # TODO

    def test_between(self):
        pass  # TODO

    def test_between_inclusive(self):
        pass  # TODO

    def test_previous(self):
        pass  # TODO

    def test_next(self):
        pass  # TODO

    def test_asOf(self):
        pass  # TODO


class FourierTransformTest(SparkTest):
    def test_fourier_transform(self):
        """Test of fourier transform functionality in TSDF objects"""

        # construct dataframes
        df = self.get_data_as_sdf("data")
        dfExpected = self.get_data_as_sdf("expected")

        # convert to TSDF
        tsdf_left = TSDF(df, ts_col="time", partition_cols=["group"])
        result_tsdf = tsdf_left.fourier_transform(1, "val")

        # should be equal to the expected dataframe
        self.assertDataFramesEqual(result_tsdf.df, dfExpected)


class RangeStatsTest(SparkTest):
    def test_range_stats(self):
        """Test of range stats for 20 minute rolling window"""

        # construct dataframes
        df = self.get_data_as_sdf("data")
        dfExpected = self.get_data_as_sdf("expected")

        # convert to TSDF
        tsdf_left = TSDF(df, partition_cols=["symbol"])

        # using lookback of 20 minutes
        featured_df = tsdf_left.withRangeStats(rangeBackWindowSecs=1200).df

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
        df = self.get_data_as_sdf("data")
        dfExpected = self.get_data_as_sdf("expected")

        # convert to TSDF
        tsdf_left = TSDF(df, partition_cols=["symbol"])

        # using lookback of 20 minutes
        featured_df = tsdf_left.withGroupedStats(freq="1 min").df

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
        df = self.get_data_as_sdf("input_data")
        dfExpected = self.get_data_as_sdf("expected")
        expected_30s_df = self.get_data_as_sdf("expected30m")
        barsExpected = self.get_data_as_sdf("expectedbars")

        # convert to TSDF
        tsdf_left = TSDF(df, partition_cols=["symbol"])

        # 1 minute aggregation
        featured_df = tsdf_left.resample(freq="min", func="floor", prefix="floor").df
        # 30 minute aggregation
        resample_30m = tsdf_left.resample(freq="5 minutes", func="mean").df.withColumn(
            "trade_pr", F.round(F.col("trade_pr"), 2)
        )

        bars = tsdf_left.calc_bars(freq="min", metricCols=["trade_pr", "trade_pr_2"]).df

        # should be equal to the expected dataframe
        self.assertDataFramesEqual(featured_df, dfExpected)
        self.assertDataFramesEqual(resample_30m, expected_30s_df)

        # test bars summary
        self.assertDataFramesEqual(bars, barsExpected)

    def test_resample_millis(self):
        """Test of resampling for millisecond windows"""

        # construct dataframes
        df = self.get_data_as_sdf("data")
        dfExpected = self.get_data_as_sdf("expectedms")

        # convert to TSDF
        tsdf_left = TSDF(df, partition_cols=["symbol"])

        # 30 minute aggregation
        resample_ms = tsdf_left.resample(freq="ms", func="mean").df.withColumn(
            "trade_pr", F.round(F.col("trade_pr"), 2)
        )

        self.assertDataFramesEqual(resample_ms, dfExpected)

    def test_upsample(self):
        """Test of range stats for 20 minute rolling window"""

        # construct dataframes
        df = self.get_data_as_sdf("input_data")
        expected_30s_df = self.get_data_as_sdf("expected30m")
        barsExpected = self.get_data_as_sdf("expectedbars")

        # convert to TSDF
        tsdf_left = TSDF(df, partition_cols=["symbol"])

        resample_30m = tsdf_left.resample(
            freq="5 minutes", func="mean", fill=True
        ).df.withColumn("trade_pr", F.round(F.col("trade_pr"), 2))

        bars = tsdf_left.calc_bars(freq="min", metricCols=["trade_pr", "trade_pr_2"]).df

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


# MAIN
if __name__ == "__main__":
    unittest.main()
