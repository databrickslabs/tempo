from tempo.stats import *
from tests.base import SparkTest


class FourierTransformTest(SparkTest):
    def test_fourier_transform(self):
        """Test of fourier transform functionality in TSDF objects"""

        # construct dataframes
        tsdf_init = self.get_data_as_tsdf("init")
        dfExpected = self.get_data_as_sdf("expected")

        # convert to TSDF
        result_tsdf = fourier_transform(tsdf_init, 1, "val")

        # should be equal to the expected dataframe
        self.assertDataFrameEquality(result_tsdf.df, dfExpected)

    def test_fourier_transform_valid_sequence_col_empty_partition_cols(self):
        """Test of fourier transform functionality in TSDF objects"""

        # construct dataframes
        tsdf_init = self.get_data_as_tsdf("init")
        dfExpected = self.get_data_as_sdf("expected")

        # convert to TSDF
        result_tsdf = fourier_transform(tsdf_init, 1, "val")

        # should be equal to the expected dataframe
        self.assertDataFrameEquality(result_tsdf.df, dfExpected)

    def test_fourier_transform_valid_sequence_col_valid_partition_cols(self):
        """Test of fourier transform functionality in TSDF objects"""

        # construct dataframes
        tsdf_init = self.get_data_as_tsdf("init")
        dfExpected = self.get_data_as_sdf("expected")

        # convert to TSDF
        result_tsdf = fourier_transform(tsdf_init, 1, "val")

        # should be equal to the expected dataframe
        self.assertDataFrameEquality(result_tsdf.df, dfExpected)

    def test_fourier_transform_no_sequence_col_empty_partition_cols(self):
        """Test of fourier transform functionality in TSDF objects"""

        # construct dataframes
        tsdf_init = self.get_data_as_tsdf("init")
        dfExpected = self.get_data_as_sdf("expected")

        # convert to TSDF
        result_tsdf = fourier_transform(tsdf_init, 1, "val")

        # should be equal to the expected dataframe
        self.assertDataFrameEquality(result_tsdf.df, dfExpected)


class RangeStatsTest(SparkTest):
    def test_range_stats(self):
        """Test of range stats for 20 minute rolling window"""

        # construct dataframes
        tsdf_init = self.get_data_as_tsdf("init")
        dfExpected = self.get_data_as_sdf("expected")

        # convert to TSDF

        # using lookback of 20 minutes
        featured_df = withRangeStats(tsdf_init, range_back_window_secs=1200).df

        # cast to decimal with precision in cents for simplicity
        featured_df = featured_df.select(
            sfn.col("symbol"),
            sfn.col("event_ts"),
            sfn.col("mean_trade_pr").cast("decimal(5, 2)"),
            sfn.col("count_trade_pr"),
            sfn.col("min_trade_pr").cast("decimal(5,2)"),
            sfn.col("max_trade_pr").cast("decimal(5,2)"),
            sfn.col("sum_trade_pr").cast("decimal(5,2)"),
            sfn.col("stddev_trade_pr").cast("decimal(5,2)"),
            sfn.col("zscore_trade_pr").cast("decimal(5,2)"),
        )

        # cast to decimal with precision in cents for simplicity
        dfExpected = dfExpected.select(
            sfn.col("symbol"),
            sfn.col("event_ts"),
            sfn.col("mean_trade_pr").cast("decimal(5, 2)"),
            sfn.col("count_trade_pr"),
            sfn.col("min_trade_pr").cast("decimal(5,2)"),
            sfn.col("max_trade_pr").cast("decimal(5,2)"),
            sfn.col("sum_trade_pr").cast("decimal(5,2)"),
            sfn.col("stddev_trade_pr").cast("decimal(5,2)"),
            sfn.col("zscore_trade_pr").cast("decimal(5,2)"),
        )

        # should be equal to the expected dataframe
        self.assertDataFrameEquality(featured_df, dfExpected)

    def test_group_stats(self):
        """Test of range stats for 20 minute rolling window"""

        # construct dataframes
        tsdf_init = self.get_data_as_tsdf("init")
        dfExpected = self.get_data_as_sdf("expected")

        # using lookback of 20 minutes
        featured_df = withGroupedStats(tsdf_init, freq="1 min").df

        # cast to decimal with precision in cents for simplicity
        featured_df = featured_df.select(
            sfn.col("symbol"),
            sfn.col("event_ts"),
            sfn.col("mean_trade_pr").cast("decimal(5, 2)"),
            sfn.col("count_trade_pr"),
            sfn.col("min_trade_pr").cast("decimal(5,2)"),
            sfn.col("max_trade_pr").cast("decimal(5,2)"),
            sfn.col("sum_trade_pr").cast("decimal(5,2)"),
            sfn.col("stddev_trade_pr").cast("decimal(5,2)"),
        )

        # cast to decimal with precision in cents for simplicity
        dfExpected = dfExpected.select(
            sfn.col("symbol"),
            sfn.col("event_ts"),
            sfn.col("mean_trade_pr").cast("decimal(5, 2)"),
            sfn.col("count_trade_pr"),
            sfn.col("min_trade_pr").cast("decimal(5,2)"),
            sfn.col("max_trade_pr").cast("decimal(5,2)"),
            sfn.col("sum_trade_pr").cast("decimal(5,2)"),
            sfn.col("stddev_trade_pr").cast("decimal(5,2)"),
        )

        # should be equal to the expected dataframe
        self.assertDataFrameEquality(featured_df, dfExpected)
