import sys
from typing import Tuple

from pyspark.sql import SparkSession as spark
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import (
    col,
    first,
    last,
    lit,
    to_timestamp,
    udf,
    unix_timestamp,
    when,
)
from pyspark.sql.types import FloatType
from pyspark.sql.window import Window
from tempo import *

fill_options = ["zero", "null", "back", "forward", "linear"]


class Interpolation:
    def __init__(self):
        self.linear_udf = udf(Interpolation.__calc_linear, FloatType())

    def get_time_range(self) -> Tuple[str]:
        start_ts, end_ts = self.df.select(min(self.ts_col), max(self.ts_col)).first()
        self.start_ts = start_ts
        self.end_ts = end_ts
        return start_ts, end_ts

    def generate_series(self, start, stop, interval):
        """
        Generate timeseries range.

        :param start  - lower bound, inclusive
        :param stop   - upper bound, exclusive
        :interval int - increment interval in seconds
        """
        # Determine start and stops in epoch seconds
        start, stop = (
            spark.createDataFrame([(start, stop)], ("start", "stop"))
            .select([col(c).cast("timestamp").cast("long") for c in ("start", "stop")])
            .first()
        )
        # Create range with increments and cast to timestamp
        return spark.range(start, stop, interval).select(
            col("id").cast("timestamp").alias("generated_ts")
        )

    @staticmethod
    def __calc_linear(epoch, epoch_ff, epoch_bf, value_ff, value_bf, value):
        if epoch_bf == epoch_ff:
            return value
        else:
            m = (value_ff - value_bf) / (epoch_ff - epoch_bf)
            value_linear = value_bf + m * (epoch - epoch_bf)
            return value_linear

    def __fill_series(
        self, df: DataFrame, partition_col: str, ts_col: str, target_col: str
    ) -> DataFrame:

        """
        Apply forward and backward fill to the timeseries dataset.

        :param df  - input dataframe
        :param partition_col   - partition column for window (must exist in the input dataframe)
        :ts_col int - timeseries column (must exist in the input dataframe)
        :target_col int - column to perform interpolation (must exist in the input dataframe)
        """
        window_ff = (
            Window.partitionBy(partition_col)
            .orderBy("series_ts")
            .rowsBetween(-sys.maxsize, 0)
        )
        window_bf = (
            Window.partitionBy(partition_col)
            .orderBy("series_ts")
            .rowsBetween(0, sys.maxsize)
        )

        read_last = last(df[target_col], ignorenulls=True).over(window_ff)
        read_next = first(df[target_col], ignorenulls=True).over(window_bf)
        readtime_last = last(df[ts_col], ignorenulls=True).over(window_ff)
        readtime_next = first(df[ts_col], ignorenulls=True).over(window_bf)

        filled_series: DataFrame = (
            df.withColumn("readvalue_ff", read_last)
            .withColumn("readvalue_bf", read_next)
            .withColumn("readtime_ff", readtime_last)
            .withColumn("readtime_bf", readtime_next)
            .filter(col("readtime_bf").isNotNull())
        )

        return filled_series

    def __validate_fill(self, fill: str):
        if fill in str(fill_options):
            pass
        else:
            raise ValueError(
                "Please select from one of the following fill options: zero, null, back, forward, linear"
            )

    def interpolate(
        self,
        tsdf: TSDF,
        ts_col: str,
        partition_col: str,
        target_col: str,
        sample_freq: str,
        sample_func: str,
        fill: str,
    ) -> TSDF:

        # Validate Fill
        self.__validate_fill(fill)

        # Resample and Normalize Columns
        no_fill: DataFrame = tsdf.resample(freq=sample_freq, func=sample_func).df
        zero_fill: DataFrame = tsdf.resample(
            freq=sample_freq, func=sample_func, fill=True
        ).df.select(
            col(ts_col).alias("series_ts"), col(partition_col).alias("partition_key")
        )

        # Join Sampled DataFrames - Generate Complete Timeseries
        joined_series: DataFrame = (
            zero_fill.join(
                no_fill,
                (zero_fill["series_ts"] == no_fill[ts_col])
                & (zero_fill["partition_key"] == no_fill[partition_col]),
                "left",
            )
            # Mark timestamps that are interpolated
            .withColumn(
                "is_interpolated", when(col(ts_col).isNull(), True).otherwise(False)
            ).drop(partition_col)
        )

        # Normalize Timestamp to Epoch Seconds
        epoch_series: DataFrame = joined_series.withColumn(
            "series_ts", unix_timestamp("series_ts")
        ).withColumn(ts_col, unix_timestamp(ts_col))

        # Generate Fill for Dataset
        filled_series: DataFrame = self.__fill_series(
            epoch_series, "partition_key", ts_col, target_col
        )

        output_df: DataFrame = filled_series

        # Handle zero fill
        if fill == "zero":
            output_df = filled_series.withColumn(
                target_col,
                when(col(target_col).isNotNull(), col(target_col)).otherwise(lit(0)),
            )

        # Handle forward fill
        if fill == "ff":
            output_df = filled_series.withColumn(target_col, col("readvalue_ff"))

        # Handle backwards fill
        if fill == "bf":
            output_df = filled_series.withColumn(target_col, col("readvalue_bf"))

        # Handle linear fill
        if fill == "linear":
            output_df = output_df.withColumn(
                target_col,
                self.linear_udf(
                    "series_ts",
                    "readtime_ff",
                    "readtime_bf",
                    "readvalue_ff",
                    "readvalue_bf",
                    target_col,
                ),
            )
        # Denormalize output
        output_df = (
            output_df.drop("readvalue_ff", "readvalue_bf", "readtime_ff", "readtime_bf")
            .withColumnRenamed("partition_key", partition_col)
            .withColumn(ts_col, to_timestamp("series_ts"))
            .drop("series_ts")
        )

        return output_df
