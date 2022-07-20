import unittest

from dateutil import parser as dt_parser

from pyspark.sql import DataFrame
from pyspark.sql import Column
import pyspark.sql.functions as F

from tempo.tsdf import TSDF
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

    def __timestamp_to_double(self, ts: str) -> float:
        return dt_parser.isoparse(ts).timestamp()

    def __tsdf_with_double_tscol(self, tsdf: TSDF) -> TSDF:
        with_double_tscol_df = tsdf.df.withColumn(
            tsdf.ts_col, F.col(tsdf.ts_col).cast("double")
        )
        return TSDF(with_double_tscol_df, tsdf.ts_col, tsdf.partitionCols)

    def test_at(self):
        """
        Test of time-slicing at(..) function
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        target_ts = "2020-09-01 00:02:10"
        at_tsdf = init_tsdf.at(target_ts)

        self.assertTSDFsEqual(at_tsdf, expected_tsdf)

        # test with numeric ts_col
        init_dbl_tsdf = self.__tsdf_with_double_tscol(init_tsdf)
        expected_dbl_tsdf = self.__tsdf_with_double_tscol(expected_tsdf)

        target_dbl = self.__timestamp_to_double(target_ts)
        at_dbl_tsdf = init_dbl_tsdf.at(target_dbl)

        self.assertTSDFsEqual(at_dbl_tsdf, expected_dbl_tsdf)

    def test_before(self):
        """
        Test of time-slicing before(..) function
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        target_ts = "2020-09-01 00:02:10"
        before_tsdf = init_tsdf.before(target_ts)

        self.assertTSDFsEqual(before_tsdf, expected_tsdf)

        # test with numeric ts_col
        init_dbl_tsdf = self.__tsdf_with_double_tscol(init_tsdf)
        expected_dbl_tsdf = self.__tsdf_with_double_tscol(expected_tsdf)

        target_dbl = self.__timestamp_to_double(target_ts)
        before_dbl_tsdf = init_dbl_tsdf.before(target_dbl)

        self.assertTSDFsEqual(before_dbl_tsdf, expected_dbl_tsdf)

    def test_atOrBefore(self):
        """
        Test of time-slicing atOrBefore(..) function
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        target_ts = "2020-09-01 00:02:10"
        before_tsdf = init_tsdf.atOrBefore(target_ts)

        self.assertTSDFsEqual(before_tsdf, expected_tsdf)

        # test with numeric ts_col
        init_dbl_tsdf = self.__tsdf_with_double_tscol(init_tsdf)
        expected_dbl_tsdf = self.__tsdf_with_double_tscol(expected_tsdf)

        target_dbl = self.__timestamp_to_double(target_ts)
        before_dbl_tsdf = init_dbl_tsdf.atOrBefore(target_dbl)

        self.assertTSDFsEqual(before_dbl_tsdf, expected_dbl_tsdf)

    def test_after(self):
        """
        Test of time-slicing after(..) function
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        target_ts = "2020-09-01 00:02:10"
        after_tsdf = init_tsdf.after(target_ts)

        self.assertTSDFsEqual(after_tsdf, expected_tsdf)

        # test with numeric ts_col
        init_dbl_tsdf = self.__tsdf_with_double_tscol(init_tsdf)
        expected_dbl_tsdf = self.__tsdf_with_double_tscol(expected_tsdf)

        target_dbl = self.__timestamp_to_double(target_ts)
        after_dbl_tsdf = init_dbl_tsdf.after(target_dbl)

        self.assertTSDFsEqual(after_dbl_tsdf, expected_dbl_tsdf)

    def test_atOrAfter(self):
        """
        Test of time-slicing atOrAfter(..) function
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        target_ts = "2020-09-01 00:02:10"
        after_tsdf = init_tsdf.atOrAfter(target_ts)

        self.assertTSDFsEqual(after_tsdf, expected_tsdf)

        # test with numeric ts_col
        init_dbl_tsdf = self.__tsdf_with_double_tscol(init_tsdf)
        expected_dbl_tsdf = self.__tsdf_with_double_tscol(expected_tsdf)

        target_dbl = self.__timestamp_to_double(target_ts)
        after_dbl_tsdf = init_dbl_tsdf.atOrAfter(target_dbl)

        self.assertTSDFsEqual(after_dbl_tsdf, expected_dbl_tsdf)

    def test_between(self):
        """
        Test of time-slicing between(..) function
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        ts1 = "2020-08-01 00:01:10"
        ts2 = "2020-09-01 00:18:00"
        between_tsdf = init_tsdf.between(ts1, ts2)

        self.assertTSDFsEqual(between_tsdf, expected_tsdf)

        # test with numeric ts_col
        init_dbl_tsdf = self.__tsdf_with_double_tscol(init_tsdf)
        expected_dbl_tsdf = self.__tsdf_with_double_tscol(expected_tsdf)

        ts1_dbl = self.__timestamp_to_double(ts1)
        ts2_dbl = self.__timestamp_to_double(ts2)
        between_dbl_tsdf = init_dbl_tsdf.between(ts1_dbl, ts2_dbl)

        self.assertTSDFsEqual(between_dbl_tsdf, expected_dbl_tsdf)

    def test_between_exclusive(self):
        """
        Test of time-slicing between(..) function
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        ts1 = "2020-08-01 00:01:10"
        ts2 = "2020-09-01 00:18:00"
        between_tsdf = init_tsdf.between(ts1, ts2, inclusive=False)

        self.assertTSDFsEqual(between_tsdf, expected_tsdf)

        # test with numeric ts_col
        init_dbl_tsdf = self.__tsdf_with_double_tscol(init_tsdf)
        expected_dbl_tsdf = self.__tsdf_with_double_tscol(expected_tsdf)

        ts1_dbl = self.__timestamp_to_double(ts1)
        ts2_dbl = self.__timestamp_to_double(ts2)
        between_dbl_tsdf = init_dbl_tsdf.between(ts1_dbl, ts2_dbl, inclusive=False)

        self.assertTSDFsEqual(between_dbl_tsdf, expected_dbl_tsdf)

    def test_earliest(self):
        """
        Test of time-slicing earliest(..) function
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        earliest_tsdf = init_tsdf.earliest(n=3)

        self.assertTSDFsEqual(earliest_tsdf, expected_tsdf)

        # test with numeric ts_col
        init_dbl_tsdf = self.__tsdf_with_double_tscol(init_tsdf)
        expected_dbl_tsdf = self.__tsdf_with_double_tscol(expected_tsdf)

        earliest_dbl_tsdf = init_dbl_tsdf.earliest(n=3)

        self.assertTSDFsEqual(earliest_dbl_tsdf, expected_dbl_tsdf)

    def test_latest(self):
        """
        Test of time-slicing latest(..) function
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        latest_tsdf = init_tsdf.latest(n=3)

        self.assertTSDFsEqual(latest_tsdf, expected_tsdf)

        # test with numeric ts_col
        init_dbl_tsdf = self.__tsdf_with_double_tscol(init_tsdf)
        expected_dbl_tsdf = self.__tsdf_with_double_tscol(expected_tsdf)

        latest_dbl_tsdf = init_dbl_tsdf.latest(n=3)

        self.assertTSDFsEqual(latest_dbl_tsdf, expected_dbl_tsdf)

    def test_priorTo(self):
        """
        Test of time-slicing priorTo(..) function
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        target_ts = "2020-09-01 00:02:00"
        prior_tsdf = init_tsdf.priorTo(target_ts)

        self.assertTSDFsEqual(prior_tsdf, expected_tsdf)

        # test with numeric ts_col
        init_dbl_tsdf = self.__tsdf_with_double_tscol(init_tsdf)
        expected_dbl_tsdf = self.__tsdf_with_double_tscol(expected_tsdf)

        target_dbl = self.__timestamp_to_double(target_ts)
        prior_dbl_tsdf = init_dbl_tsdf.priorTo(target_dbl)

        self.assertTSDFsEqual(prior_dbl_tsdf, expected_dbl_tsdf)

    def test_subsequentTo(self):
        """
        Test of time-slicing subsequentTo(..) function
        """
        init_tsdf = self.get_data_as_tsdf("init")
        expected_tsdf = self.get_data_as_tsdf("expected")

        target_ts = "2020-09-01 00:02:00"
        subsequent_tsdf = init_tsdf.subsequentTo(target_ts)

        self.assertTSDFsEqual(subsequent_tsdf, expected_tsdf)

        # test with numeric ts_col
        init_dbl_tsdf = self.__tsdf_with_double_tscol(init_tsdf)
        expected_dbl_tsdf = self.__tsdf_with_double_tscol(expected_tsdf)

        target_dbl = self.__timestamp_to_double(target_ts)
        subsequent_dbl_tsdf = init_dbl_tsdf.subsequentTo(target_dbl)

        self.assertTSDFsEqual(subsequent_dbl_tsdf, expected_dbl_tsdf)


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


class ExtractStateIntervalsTest(SparkTest):
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
        #expected_df = self.create_expected_test_df(expected_df)

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
        input_1_tsdf = self.get_data_as_tsdf("input_1")
        expected_1_df = self.get_data_as_sdf("expected_1")
        expected_1_df = self.create_expected_test_df(expected_1_df)
        input_2_tsdf = self.get_data_as_tsdf("input_2")
        expected_2_df = self.get_data_as_sdf("expected_2")
        expected_2_df = self.create_expected_test_df(expected_2_df)

        # call extractStateIntervals method
        ExtractStateIntervals_ne_1_1_df = input_1_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition="!="
        ).df
        ExtractStateIntervals_ne_1_2_df = input_1_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition="<>"
        ).df
        ExtractStateIntervals_ne_2_1_df = input_2_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition="!="
        ).df
        ExtractStateIntervals_ne_2_2_df = input_2_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition="<>"
        ).df

        # test extractStateIntervals_tsdf summary
        self.assertDataFramesEqual(ExtractStateIntervals_ne_1_1_df, expected_1_df)
        self.assertDataFramesEqual(ExtractStateIntervals_ne_1_2_df, expected_1_df)
        self.assertDataFramesEqual(ExtractStateIntervals_ne_2_1_df, expected_2_df)
        self.assertDataFramesEqual(ExtractStateIntervals_ne_2_2_df, expected_2_df)

    def test_gt_extractStateIntervals(self):
        # construct dataframes
        input_1_tsdf = self.get_data_as_tsdf("input_1")
        expected_1_df = self.get_data_as_sdf("expected_1")
        expected_1_df = self.create_expected_test_df(expected_1_df)
        input_2_tsdf = self.get_data_as_tsdf("input_2")
        expected_2_df = self.get_data_as_sdf("expected_2")
        expected_2_df = self.create_expected_test_df(expected_2_df)

        # call extractStateIntervals method
        extractStateIntervals_gt_1_1_df = input_1_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition=">"
        ).df
        extractStateIntervals_gt_2_1_df = input_2_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition=">"
        ).df

        self.assertDataFramesEqual(extractStateIntervals_gt_1_1_df, expected_1_df)
        self.assertDataFramesEqual(extractStateIntervals_gt_2_1_df, expected_2_df)

    def test_lt_extractStateIntervals(self):
        # construct dataframes
        input_1_tsdf = self.get_data_as_tsdf("input_1")
        expected_1_df = self.get_data_as_sdf("expected_1")
        expected_1_df = self.create_expected_test_df(expected_1_df)
        input_2_tsdf = self.get_data_as_tsdf("input_2")
        expected_2_df = self.get_data_as_sdf("expected_2")
        expected_2_df = self.create_expected_test_df(expected_2_df)

        # call extractStateIntervals method
        extractStateIntervals_lt_1_1_df = input_1_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition="<"
        ).df
        extractStateIntervals_lt_2_1_df = input_2_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition="<"
        ).df

        # test extractStateIntervals_tsdf summary
        self.assertDataFramesEqual(extractStateIntervals_lt_1_1_df, expected_1_df)
        self.assertDataFramesEqual(extractStateIntervals_lt_2_1_df, expected_2_df)

    def test_gte_extractStateIntervals(self):
        # construct dataframes
        input_1_tsdf = self.get_data_as_tsdf("input_1")
        expected_1_df = self.get_data_as_sdf("expected_1")
        expected_1_df = self.create_expected_test_df(expected_1_df)
        input_2_tsdf = self.get_data_as_tsdf("input_2")
        expected_2_df = self.get_data_as_sdf("expected_2")
        expected_2_df = self.create_expected_test_df(expected_2_df)

        # call extractStateIntervals method
        extractStateIntervals_gt_1_1_df = input_1_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition=">="
        ).df
        extractStateIntervals_gt_2_1_df = input_2_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition=">="
        ).df

        self.assertDataFramesEqual(extractStateIntervals_gt_1_1_df, expected_2_df)
        self.assertDataFramesEqual(extractStateIntervals_gt_2_1_df, expected_2_df)

    def test_lte_extractStateIntervals(self):
        # construct dataframes
        input_1_tsdf = self.get_data_as_tsdf("input_1")
        expected_1_df = self.get_data_as_sdf("expected_1")
        expected_1_df = self.create_expected_test_df(expected_1_df)
        input_2_tsdf = self.get_data_as_tsdf("input_2")
        expected_2_df = self.get_data_as_sdf("expected_2")
        expected_2_df = self.create_expected_test_df(expected_2_df)

        # call extractStateIntervals method
        extractStateIntervals_lte_1_1_df = input_1_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition="<="
        ).df
        extractStateIntervals_lte_2_1_df = input_2_tsdf.extractStateIntervals(
            "metric_1", "metric_2", "metric_3", state_definition="<="
        ).df

        # test extractStateIntervals_tsdf summary
        self.assertDataFramesEqual(extractStateIntervals_lte_1_1_df, expected_1_df)
        self.assertDataFramesEqual(extractStateIntervals_lte_2_1_df, expected_1_df)

    def test_threshold_fn_extractStateIntervals(self):
        # construct dataframes
        input_tsdf = self.get_data_as_tsdf("input")
        expected_df = self.get_data_as_sdf("expected")
        #expected_df = self.create_expected_test_df(expected_df)

        # threshold state function
        def threshold_fn( a: Column, b: Column ) -> Column:
            return F.abs(a - b) < F.lit(0.5)

        # call extractStateIntervals method
        extracted_intervals_df = input_tsdf.extractStateIntervals(
            "metric_1",
            "metric_2",
            "metric_3",
            state_definition=threshold_fn
        )

        #print(f"extracted state intervals has schema: {extracted_intervals_df.schema}")
        #print(f"expected state intervals has schema: {expected_df.schema}")

        # test extractStateIntervals_tsdf summary
        self.assertDataFramesEqual(extracted_intervals_df, expected_df)


# MAIN
if __name__ == "__main__":
    unittest.main()
