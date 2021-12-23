import sys
from datetime import datetime
from typing import List, Tuple

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

# Interpolation fill options
fill_options = ["zero", "null", "back", "forward", "linear"]
supported_target_col_types = ["double", "float"]


class Interpolation:
    def __init__(self):
        """
        Constructor

        :linear_udf register linear calculation UDF
        """
        self.__linear_udf = udf(Interpolation.__calc_linear, FloatType())

    @staticmethod
    def __calc_linear(epoch, epoch_ff, epoch_bf, value_ff, value_bf, value):
        """
        User defined function for calculating linear interpolation on a DataFrame.

        :param epoch  - Original epoch timestamp of the column to be interpolated.
        :param epoch_ff   -  Forward filled epoch timestamp of the column to be interpolated.
        :param epoch_bf  - Backfilled epoch timestamp of the column to be interpolated.
        :param value_ff   -  Forward filled value of the column to be interpolated.
        :param value_bf  - Backfilled value of the column to be interpolated.
        :param value   -  Original value of the column to be interpolated.
        """
        if epoch_bf == epoch_ff:
            return value
        else:
            m = (value_ff - value_bf) / (epoch_ff - epoch_bf)
            value_linear = value_bf + m * (epoch - epoch_bf)
            return value_linear

    # TODO: Currently not being used. But will useful for interpolating arbitrary ranges.
    def get_time_range(self, df: DataFrame, ts_col: str) -> Tuple[str]:
        """
        Get minimum timestamp and maximum timestamp for DataFrame.

        :param df  - Input dataframe (must have timestamp column)
        :param ts_col   -  Timestamp column name
        """
        start_ts, end_ts = df.select(min(ts_col), max(ts_col)).first()
        self.start_ts = start_ts
        self.end_ts = end_ts
        return start_ts, end_ts

    # TODO: Currently not being used. But will useful for interpolating arbitrary ranges.
    def generate_series(
        self, start: datetime, stop: datetime, interval: int
    ) -> DataFrame:
        """
        Generate timeseries for a given range and interval.

        :param start  - lower bound, inclusive
        :param stop   - upper bound, exclusive
        :param interval - increment interval in seconds
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

    def __validate_fill(self, fill: str):
        """
        Validate if the fill provided is within the allowed list of values.

        :param fill - Fill type e.g. "zero", "null", "back", "forward", "linear"
        """
        if fill not in fill_options:
            raise ValueError(
                f"Please select from one of the following fill options: {fill_options}"
            )

    def __validate_col(
        self,
        df: DataFrame,
        partition_cols: List[str],
        target_cols: List[str],
        ts_col: str,
    ):
        """
        Validate if target column exists and is of numeric type, and validates if partition column exists.

        :param df -  DataFrame to be validated
        :param partition_cols -  Partition columns to be validated
        :param target_col -  Target column to be validated
        :param ts_col -  Timestamp column to be validated
        """
        for column in partition_cols:
            if column not in str(df.columns):
                raise ValueError(
                    f"Partition Column: '{column}' does not exist in DataFrame."
                )
        for column in target_cols:
            if column not in str(df.columns):
                raise ValueError(
                    f"Target Column: '{column}' does not exist in DataFrame."
                )
            if df.select(column).dtypes[0][1] not in supported_target_col_types:
                raise ValueError(
                    f"Target Column needs to be one of the following types: {supported_target_col_types}"
                )

        if ts_col not in str(df.columns):
            raise ValueError(
                f"Timestamp Column: '{ts_col}' does not exist in DataFrame."
            )

        if df.select(ts_col).dtypes[0][1] != "timestamp":
            raise ValueError(f"Timestamp Column needs to be of timestamp type.")

    def __fill_series(
        self, df: DataFrame, partition_cols: List[str], ts_col: str, target_col: str
    ) -> DataFrame:
        """
        Apply forward and backward fill (value and timestamp) to the timeseries dataset.

        :param df  - input dataframe
        :param partition_cols   - partition columns for window (columns must exist in the input dataframe)
        :ts_col int - timeseries column (must exist in the input dataframe)
        :target_col int - column to perform interpolation (must exist in the input dataframe)
        """
        window_ff = (
            Window.partitionBy(*partition_cols)
            .orderBy(ts_col)
            .rowsBetween(-sys.maxsize, 0)
        )
        window_bf = (
            Window.partitionBy(*partition_cols)
            .orderBy(ts_col)
            .rowsBetween(0, sys.maxsize)
        )

        read_last = last(df[target_col], ignorenulls=True).over(window_ff)
        read_next = first(df[target_col], ignorenulls=True).over(window_bf)
        readtime_last = last(df["surrogate_ts"], ignorenulls=True).over(window_ff)
        readtime_next = first(df["surrogate_ts"], ignorenulls=True).over(window_bf)

        filled_series: DataFrame = (
            df.withColumn("readvalue_ff", read_last)
            .withColumn("readvalue_bf", read_next)
            .withColumn("readtime_ff", readtime_last)
            .withColumn("readtime_bf", readtime_next)
            .filter(col("readtime_bf").isNotNull())
        )

        return filled_series

    def __interpolate_column(
        self,
        epoch_series: DataFrame,
        partition_cols: List[str],
        ts_col: str,
        target_col: str,
        fill: str,
    ) -> DataFrame:
        """
        Apply interpolation to column.

        :param epoch_series  - input DataFrame in epoch seconds
        :param ts_col   - timestamp column name
        :param target_col   - numeric column to interpolate
        :param fill   - interpolation function to fill missing values
        """
        # Generate Fill for Dataset
        filled_series: DataFrame = self.__fill_series(
            epoch_series, partition_cols, ts_col, target_col
        )

        output_df: DataFrame = filled_series

        # Handle zero fill
        if fill == "zero":
            output_df = filled_series.withColumn(
                target_col,
                when(col(target_col).isNotNull(), col(target_col)).otherwise(lit(0)),
            )

        # Handle forward fill
        if fill == "forward":
            output_df = filled_series.withColumn(target_col, col("readvalue_ff"))

        # Handle backwards fill
        if fill == "back":
            output_df = filled_series.withColumn(target_col, col("readvalue_bf"))

        # Handle linear fill
        if fill == "linear":
            output_df = output_df.withColumn(
                target_col,
                self.__linear_udf(
                    ts_col,
                    "readtime_ff",
                    "readtime_bf",
                    "readvalue_ff",
                    "readvalue_bf",
                    target_col,
                ),
            )

        return output_df

    def interpolate(
        self,
        tsdf: TSDF,
        ts_col: str,
        partition_cols: List[str],
        target_cols: List[str],
        sample_freq: str,
        sample_func: str,
        fill: str,
    ) -> TSDF:
        """
        Apply interpolation to TSDF.

        :param tsdf  - input TSDF
        :param target_cols   - numeric columns to interpolate
        :param sample_freq  - frequency at which to sample
        :param sample_func   - aggregate function for sampling
        :param fill   - interpolation function to fill missing values
        :param ts_col   - timestamp column name
        :param partition_cols  - partition columns names
        """
        # Validate parameters
        self.__validate_fill(fill)
        self.__validate_col(tsdf.df, partition_cols, target_cols, ts_col)

        # Resample and Normalize Columns
        no_fill: DataFrame = tsdf.resample(
            freq=sample_freq, func=sample_func, metricCols=target_cols
        ).df

        # Build columns list to joining the two sampled series
        join_cols: List[str] = partition_cols + [ts_col]

        # Get series zero filled
        zero_fill: DataFrame = tsdf.resample(
            freq=sample_freq, func=sample_func, fill=True, metricCols=target_cols
        ).df.select(*join_cols)

        # Join sampled DataFrames - generate complete timeseries
        joined_series: DataFrame = zero_fill.join(
            no_fill,
            join_cols,
            "left",
        )

        # Perform interpolation on each target column
        output_list: List[DataFrame] = []
        for target_col in target_cols:
            # Mark columns that are interpolated
            marked_series: DataFrame = joined_series.withColumn(
                f"is_{target_col}_interpolated",
                when(col(target_col).isNull(), True).otherwise(False),
            )

            # Add surrogate ts_col to get missing values
            with_surrogate_ts: DataFrame = marked_series.withColumn(
                "surrogate_ts",
                when(col(target_col).isNull(), None).otherwise(col(ts_col)),
            )

            # # Normalize Timestamp to Epoch Seconds
            epoch_series: DataFrame = with_surrogate_ts.withColumn(
                ts_col, unix_timestamp(ts_col)
            ).withColumn("surrogate_ts", unix_timestamp("surrogate_ts"))

            # # Interpolate target columns
            interpolated_result: DataFrame = self.__interpolate_column(
                epoch_series, partition_cols, ts_col, target_col, fill
            )

            # Denormalize output
            columns_to_drop: List[str] = [
                "readvalue_ff",
                "readvalue_bf",
                "readtime_ff",
                "readtime_bf",
                "surrogate_ts",
            ] + target_cols
            columns_to_drop.remove(target_col)
            normalized_output: DataFrame = (
                # Drop intermediatry columns
                interpolated_result.drop(*columns_to_drop)
                # Convert timestamp column back to timestamp
                .withColumn(ts_col, to_timestamp(ts_col))
            )
            output_list.append(normalized_output)

        # Join output_list into single DataFrame to output
        output: DataFrame = output_list[0]
        output.cache()
        for right_df in output_list[1:]:
            output: DataFrame = output.join(right_df, on=join_cols, how="left")

        return output
