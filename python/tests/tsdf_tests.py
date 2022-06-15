import unittest

import pyspark.sql.functions as F
from pyspark.sql.types import *

from tempo.tsdf import TSDF

from tests.base import SparkTest


class BasicTests(SparkTest):
    def test_describe(self):
        """AS-OF Join with out a time-partition test"""
        leftSchema = StructType(
            [
                StructField("symbol", StringType()),
                StructField("event_ts", StringType()),
                StructField("trade_pr", FloatType()),
            ]
        )

        rightSchema = StructType(
            [
                StructField("symbol", StringType()),
                StructField("event_ts", StringType()),
                StructField("bid_pr", FloatType()),
                StructField("ask_pr", FloatType()),
            ]
        )

        expectedSchema = StructType(
            [
                StructField("symbol", StringType()),
                StructField("left_event_ts", StringType()),
                StructField("left_trade_pr", FloatType()),
                StructField("right_event_ts", StringType()),
                StructField("right_bid_pr", FloatType()),
                StructField("right_ask_pr", FloatType()),
            ]
        )

        left_data = [
            ["S1", "2020-08-01 00:00:10", 349.21],
            ["S1", "2020-08-01 00:01:12", 351.32],
            ["S1", "2020-09-01 00:02:10", 361.1],
            ["S1", "2020-09-01 00:19:12", 362.1],
        ]

        right_data = [
            ["S1", "2020-08-01 00:00:01", 345.11, 351.12],
            ["S1", "2020-08-01 00:01:05", 348.10, 353.13],
            ["S1", "2020-09-01 00:02:01", 358.93, 365.12],
            ["S1", "2020-09-01 00:15:01", 359.21, 365.31],
        ]

        expected_data = [
            [
                "S1",
                "2020-08-01 00:00:10",
                349.21,
                "2020-08-01 00:00:01",
                345.11,
                351.12,
            ],
            [
                "S1",
                "2020-08-01 00:01:12",
                351.32,
                "2020-08-01 00:01:05",
                348.10,
                353.13,
            ],
            ["S1", "2020-09-01 00:02:10", 361.1, "2020-09-01 00:02:01", 358.93, 365.12],
            ["S1", "2020-09-01 00:19:12", 362.1, "2020-09-01 00:15:01", 359.21, 365.31],
        ]

        # Construct dataframes
        dfLeft = self.buildTestDF(leftSchema, left_data)
        dfRight = self.buildTestDF(rightSchema, right_data)
        dfExpected = self.buildTestDF(
            expectedSchema, expected_data, ["left_event_ts", "right_event_ts"]
        )

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


class FourierTransformTest(SparkTest):
    def test_fourier_transform(self):
        """Test of fourier transform functionality in TSDF objects"""
        schema = StructType(
            [
                StructField("group", StringType()),
                StructField("time", LongType()),
                StructField("val", DoubleType()),
            ]
        )

        expectedSchema = StructType(
            [
                StructField("group", StringType()),
                StructField("time", LongType()),
                StructField("val", DoubleType()),
                StructField("freq", DoubleType()),
                StructField("ft_real", DoubleType()),
                StructField("ft_imag", DoubleType()),
            ]
        )

        data = [
            ["Emissions", 1949, 2206.690829],
            ["Emissions", 1950, 2382.046176],
            ["Emissions", 1951, 2526.687327],
            ["Emissions", 1952, 2473.373964],
            ["WindGen", 1980, 0.0],
            ["WindGen", 1981, 0.0],
            ["WindGen", 1982, 0.0],
            ["WindGen", 1983, 0.029667962],
        ]

        expected_data = [
            ["Emissions", 1949, 2206.690829, 0.0, 9588.798296, -0.0],
            ["Emissions", 1950, 2382.046176, 0.25, -319.996498, 91.32778800000006],
            ["Emissions", 1951, 2526.687327, -0.5, -122.0419839999995, -0.0],
            ["Emissions", 1952, 2473.373964, -0.25, -319.996498, -91.32778800000006],
            ["WindGen", 1980, 0.0, 0.0, 0.029667962, -0.0],
            ["WindGen", 1981, 0.0, 0.25, 0.0, 0.029667962],
            ["WindGen", 1982, 0.0, -0.5, -0.029667962, -0.0],
            ["WindGen", 1983, 0.029667962, -0.25, 0.0, -0.029667962],
        ]

        # construct dataframes
        df = self.buildTestDF(schema, data, ts_cols=["time"])
        dfExpected = self.buildTestDF(expectedSchema, expected_data, ts_cols=["time"])

        # convert to TSDF
        tsdf_left = TSDF(df, ts_col="time", partition_cols=["group"])
        result_tsdf = tsdf_left.fourier_transform(1, "val")

        # should be equal to the expected dataframe
        self.assertDataFramesEqual(result_tsdf.df, dfExpected)


class RangeStatsTest(SparkTest):
    def test_range_stats(self):
        """Test of range stats for 20 minute rolling window"""
        schema = StructType(
            [
                StructField("symbol", StringType()),
                StructField("event_ts", StringType()),
                StructField("trade_pr", FloatType()),
            ]
        )

        expectedSchema = StructType(
            [
                StructField("symbol", StringType()),
                StructField("event_ts", StringType()),
                StructField("mean_trade_pr", FloatType()),
                StructField("count_trade_pr", LongType(), nullable=False),
                StructField("min_trade_pr", FloatType()),
                StructField("max_trade_pr", FloatType()),
                StructField("sum_trade_pr", FloatType()),
                StructField("stddev_trade_pr", FloatType()),
                StructField("zscore_trade_pr", FloatType()),
            ]
        )

        data = [
            ["S1", "2020-08-01 00:00:10", 349.21],
            ["S1", "2020-08-01 00:01:12", 351.32],
            ["S1", "2020-09-01 00:02:10", 361.1],
            ["S1", "2020-09-01 00:19:12", 362.1],
        ]

        expected_data = [
            [
                "S1",
                "2020-08-01 00:00:10",
                349.21,
                1,
                349.21,
                349.21,
                349.21,
                None,
                None,
            ],
            [
                "S1",
                "2020-08-01 00:01:12",
                350.26,
                2,
                349.21,
                351.32,
                700.53,
                1.49,
                0.71,
            ],
            ["S1", "2020-09-01 00:02:10", 361.1, 1, 361.1, 361.1, 361.1, None, None],
            ["S1", "2020-09-01 00:19:12", 361.6, 2, 361.1, 362.1, 723.2, 0.71, 0.71],
        ]

        # construct dataframes
        df = self.buildTestDF(schema, data)
        dfExpected = self.buildTestDF(expectedSchema, expected_data)

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
        schema = StructType(
            [
                StructField("symbol", StringType()),
                StructField("event_ts", StringType()),
                StructField("trade_pr", FloatType()),
                StructField("index", IntegerType()),
            ]
        )

        expectedSchema = StructType(
            [
                StructField("symbol", StringType()),
                StructField("event_ts", StringType()),
                StructField("mean_trade_pr", FloatType()),
                StructField("count_trade_pr", LongType(), nullable=False),
                StructField("min_trade_pr", FloatType()),
                StructField("max_trade_pr", FloatType()),
                StructField("sum_trade_pr", FloatType()),
                StructField("stddev_trade_pr", FloatType()),
                StructField("mean_index", IntegerType()),
                StructField("count_index", IntegerType(), nullable=False),
                StructField("min_index", IntegerType()),
                StructField("max_index", IntegerType()),
                StructField("sum_index", IntegerType()),
                StructField("stddev_index", IntegerType()),
            ]
        )

        data = [
            ["S1", "2020-08-01 00:00:10", 349.21, 1],
            ["S1", "2020-08-01 00:00:33", 351.32, 1],
            ["S1", "2020-09-01 00:02:10", 361.1, 1],
            ["S1", "2020-09-01 00:02:49", 362.1, 1],
        ]

        expected_data = [
            [
                "S1",
                "2020-08-01 00:00:00",
                350.26,
                2,
                349.21,
                351.32,
                700.53,
                1.49,
                1,
                2,
                1,
                1,
                2,
                0,
            ],
            [
                "S1",
                "2020-09-01 00:02:00",
                361.6,
                2,
                361.1,
                362.1,
                723.2,
                0.71,
                1,
                2,
                1,
                1,
                2,
                0,
            ],
        ]

        # construct dataframes
        df = self.buildTestDF(schema, data)
        dfExpected = self.buildTestDF(expectedSchema, expected_data)

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
        schema = StructType(
            [
                StructField("symbol", StringType()),
                StructField("date", StringType()),
                StructField("event_ts", StringType()),
                StructField("trade_pr", FloatType()),
                StructField("trade_pr_2", FloatType()),
            ]
        )

        expectedSchema = StructType(
            [
                StructField("symbol", StringType()),
                StructField("event_ts", StringType()),
                StructField("floor_trade_pr", FloatType()),
                StructField("floor_date", StringType()),
                StructField("floor_trade_pr_2", FloatType()),
            ]
        )

        expected_30m_Schema = StructType(
            [
                StructField("symbol", StringType()),
                StructField("event_ts", StringType(), True),
                StructField("date", DoubleType()),
                StructField("trade_pr", DoubleType()),
                StructField("trade_pr_2", DoubleType()),
            ]
        )

        expectedBarsSchema = StructType(
            [
                StructField("symbol", StringType()),
                StructField("event_ts", StringType()),
                StructField("close_trade_pr", FloatType()),
                StructField("close_trade_pr_2", FloatType()),
                StructField("high_trade_pr", FloatType()),
                StructField("high_trade_pr_2", FloatType()),
                StructField("low_trade_pr", FloatType()),
                StructField("low_trade_pr_2", FloatType()),
                StructField("open_trade_pr", FloatType()),
                StructField("open_trade_pr_2", FloatType()),
            ]
        )

        data = [
            ["S1", "SAME_DT", "2020-08-01 00:00:10", 349.21, 10.0],
            ["S1", "SAME_DT", "2020-08-01 00:00:11", 340.21, 9.0],
            ["S1", "SAME_DT", "2020-08-01 00:01:12", 353.32, 8.0],
            ["S1", "SAME_DT", "2020-08-01 00:01:13", 351.32, 7.0],
            ["S1", "SAME_DT", "2020-08-01 00:01:14", 350.32, 6.0],
            ["S1", "SAME_DT", "2020-09-01 00:01:12", 361.1, 5.0],
            ["S1", "SAME_DT", "2020-09-01 00:19:12", 362.1, 4.0],
        ]

        expected_data = [
            ["S1", "2020-08-01 00:00:00", 349.21, "SAME_DT", 10.0],
            ["S1", "2020-08-01 00:01:00", 353.32, "SAME_DT", 8.0],
            ["S1", "2020-09-01 00:01:00", 361.1, "SAME_DT", 5.0],
            ["S1", "2020-09-01 00:19:00", 362.1, "SAME_DT", 4.0],
        ]

        expected_data_30m = [
            ["S1", "2020-08-01 00:00:00", None, 348.88, 8.0],
            ["S1", "2020-09-01 00:00:00", None, 361.1, 5.0],
            ["S1", "2020-09-01 00:15:00", None, 362.1, 4.0],
        ]

        expected_bars = [
            [
                "S1",
                "2020-08-01 00:00:00",
                340.21,
                9.0,
                349.21,
                10.0,
                340.21,
                9.0,
                349.21,
                10.0,
            ],
            [
                "S1",
                "2020-08-01 00:01:00",
                350.32,
                6.0,
                353.32,
                8.0,
                350.32,
                6.0,
                353.32,
                8.0,
            ],
            [
                "S1",
                "2020-09-01 00:01:00",
                361.1,
                5.0,
                361.1,
                5.0,
                361.1,
                5.0,
                361.1,
                5.0,
            ],
            [
                "S1",
                "2020-09-01 00:19:00",
                362.1,
                4.0,
                362.1,
                4.0,
                362.1,
                4.0,
                362.1,
                4.0,
            ],
        ]

        # construct dataframes
        df = self.buildTestDF(schema, data)
        dfExpected = self.buildTestDF(expectedSchema, expected_data)
        expected_30s_df = self.buildTestDF(expected_30m_Schema, expected_data_30m)
        barsExpected = self.buildTestDF(expectedBarsSchema, expected_bars)

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
        schema = StructType(
            [
                StructField("symbol", StringType()),
                StructField("date", StringType()),
                StructField("event_ts", StringType()),
                StructField("trade_pr", FloatType()),
                StructField("trade_pr_2", FloatType()),
            ]
        )

        expectedSchema = StructType(
            [
                StructField("symbol", StringType()),
                StructField("event_ts", StringType()),
                StructField("floor_trade_pr", FloatType()),
                StructField("floor_date", StringType()),
                StructField("floor_trade_pr_2", FloatType()),
            ]
        )

        expectedSchemaMS = StructType(
            [
                StructField("symbol", StringType()),
                StructField("event_ts", StringType(), True),
                StructField("date", DoubleType()),
                StructField("trade_pr", DoubleType()),
                StructField("trade_pr_2", DoubleType()),
            ]
        )

        data = [
            ["S1", "SAME_DT", "2020-08-01 00:00:10.12345", 349.21, 10.0],
            ["S1", "SAME_DT", "2020-08-01 00:00:10.123", 340.21, 9.0],
            ["S1", "SAME_DT", "2020-08-01 00:00:10.124", 353.32, 8.0],
        ]

        expected_data_ms = [
            ["S1", "2020-08-01 00:00:10.123", None, 344.71, 9.5],
            ["S1", "2020-08-01 00:00:10.124", None, 353.32, 8.0],
        ]

        # construct dataframes
        df = self.buildTestDF(schema, data)
        dfExpected = self.buildTestDF(expectedSchemaMS, expected_data_ms)

        # convert to TSDF
        tsdf_left = TSDF(df, partition_cols=["symbol"])

        # 30 minute aggregation
        resample_ms = tsdf_left.resample(freq="ms", func="mean").df.withColumn(
            "trade_pr", F.round(F.col("trade_pr"), 2)
        )

        int_df = TSDF(
            tsdf_left.df.withColumn("event_ts", F.col("event_ts").cast("timestamp")),
            partition_cols=["symbol"],
        )
        interpolated = int_df.interpolate(freq="ms", func="floor", method="ffill")
        self.assertDataFramesEqual(resample_ms, dfExpected)

    def test_upsample(self):
        """Test of range stats for 20 minute rolling window"""
        schema = StructType(
            [
                StructField("symbol", StringType()),
                StructField("date", StringType()),
                StructField("event_ts", StringType()),
                StructField("trade_pr", FloatType()),
                StructField("trade_pr_2", FloatType()),
            ]
        )

        expectedSchema = StructType(
            [
                StructField("symbol", StringType()),
                StructField("event_ts", StringType()),
                StructField("floor_trade_pr", FloatType()),
                StructField("floor_date", StringType()),
                StructField("floor_trade_pr_2", FloatType()),
            ]
        )

        expected_30m_Schema = StructType(
            [
                StructField("symbol", StringType()),
                StructField("event_ts", StringType()),
                StructField("date", DoubleType()),
                StructField("trade_pr", DoubleType()),
                StructField("trade_pr_2", DoubleType()),
            ]
        )

        expectedBarsSchema = StructType(
            [
                StructField("symbol", StringType()),
                StructField("event_ts", StringType()),
                StructField("close_trade_pr", FloatType()),
                StructField("close_trade_pr_2", FloatType()),
                StructField("high_trade_pr", FloatType()),
                StructField("high_trade_pr_2", FloatType()),
                StructField("low_trade_pr", FloatType()),
                StructField("low_trade_pr_2", FloatType()),
                StructField("open_trade_pr", FloatType()),
                StructField("open_trade_pr_2", FloatType()),
            ]
        )

        data = [
            ["S1", "SAME_DT", "2020-08-01 00:00:10", 349.21, 10.0],
            ["S1", "SAME_DT", "2020-08-01 00:00:11", 340.21, 9.0],
            ["S1", "SAME_DT", "2020-08-01 00:01:12", 353.32, 8.0],
            ["S1", "SAME_DT", "2020-08-01 00:01:13", 351.32, 7.0],
            ["S1", "SAME_DT", "2020-08-01 00:01:14", 350.32, 6.0],
            ["S1", "SAME_DT", "2020-09-01 00:01:12", 361.1, 5.0],
            ["S1", "SAME_DT", "2020-09-01 00:19:12", 362.1, 4.0],
        ]

        expected_data = [
            ["S1", "2020-08-01 00:00:00", 349.21, "SAME_DT", 10.0],
            ["S1", "2020-08-01 00:01:00", 353.32, "SAME_DT", 8.0],
            ["S1", "2020-09-01 00:01:00", 361.1, "SAME_DT", 5.0],
            ["S1", "2020-09-01 00:19:00", 362.1, "SAME_DT", 4.0],
        ]

        expected_data_30m = [
            ["S1", "2020-08-01 00:00:00", 0.0, 348.88, 8.0],
            ["S1", "2020-08-01 00:05:00", 0.0, 0.0, 0.0],
            ["S1", "2020-09-01 00:00:00", 0.0, 361.1, 5.0],
            ["S1", "2020-09-01 00:15:00", 0.0, 362.1, 4.0],
        ]

        expected_bars = [
            [
                "S1",
                "2020-08-01 00:00:00",
                340.21,
                9.0,
                349.21,
                10.0,
                340.21,
                9.0,
                349.21,
                10.0,
            ],
            [
                "S1",
                "2020-08-01 00:01:00",
                350.32,
                6.0,
                353.32,
                8.0,
                350.32,
                6.0,
                353.32,
                8.0,
            ],
            [
                "S1",
                "2020-09-01 00:01:00",
                361.1,
                5.0,
                361.1,
                5.0,
                361.1,
                5.0,
                361.1,
                5.0,
            ],
            [
                "S1",
                "2020-09-01 00:19:00",
                362.1,
                4.0,
                362.1,
                4.0,
                362.1,
                4.0,
                362.1,
                4.0,
            ],
        ]

        # construct dataframes
        df = self.buildTestDF(schema, data)
        dfExpected = self.buildTestDF(expectedSchema, expected_data)
        expected_30s_df = self.buildTestDF(expected_30m_Schema, expected_data_30m)
        barsExpected = self.buildTestDF(expectedBarsSchema, expected_bars)

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
