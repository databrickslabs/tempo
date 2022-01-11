import sys
from datetime import datetime, time
from typing import List, Tuple

from pyspark.sql import SparkSession as spark
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import (
    col,
    expr,
    first,
    lag,
    last,
    lead,
    lit,
    when,
)
from pyspark.sql.window import Window

# Interpolation fill options
method_options = ["zero", "null", "back", "forward", "linear"]
supported_target_col_types = ["int", "bigint", "float", "double"]


class Interpolation:
    def __validate_fill(self, method: str):
        """
        Validate if the fill provided is within the allowed list of values.

        :param fill - Fill type e.g. "zero", "null", "back", "forward", "linear"
        """
        if method not in method_options:
            raise ValueError(
                f"Please select from one of the following fill options: {method_options}"
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

    def __calc_linear_spark(self, df: DataFrame, ts_col: str, target_col: str):
        """
        Native Spark function for calculating linear interpolation on a DataFrame.

        :param timestamp  - Original timestamp of the column to be interpolated.
        :param next_timestamp   -  Forward filled timestamp of the column to be interpolated.
        :param previous_timestamp  - Backfilled timestamp of the column to be interpolated.
        :param next_value   -  Forward filled value of the column to be interpolated.
        :param value   -  Original value of the column to be interpolated.
        """
        cols: List[str] = df.columns
        cols.remove(target_col)
        expr = f"""
        case when is_interpolated_{target_col} = false then {target_col}
            when {target_col} is null then 
            (next_null_{target_col} - previous_{target_col})
            /(unix_timestamp(next_timestamp_{target_col})-unix_timestamp(previous_timestamp_{target_col}))
            *(unix_timestamp({ts_col}) - unix_timestamp(previous_timestamp_{target_col}))
            + previous_{target_col}
        else 
            (next_{target_col}-{target_col})
            /(unix_timestamp(next_timestamp)-unix_timestamp(previous_timestamp))
            *(unix_timestamp({ts_col}) - unix_timestamp(previous_timestamp)) 
            + {target_col}
        end as {target_col}
        """
        interpolated: DataFrame = df.selectExpr(*cols, expr)
        # Preserve column order
        return interpolated.select(*df.columns)

    def __interpolate_column(
        self,
        freq: str,
        series: DataFrame,
        ts_col: str,
        target_col: str,
        method: str,
    ) -> DataFrame:
        """
        Apply interpolation to column.

        :param series  - input DataFrame
        :param ts_col   - timestamp column name
        :param target_col   - numeric column to interpolate
        :param fill   - interpolation function to fill missing values
        """

        output_df: DataFrame = series

        expr_flag = f"""
        CASE WHEN {target_col} is null and is_ts_interpolated = false THEN true
             WHEN is_ts_interpolated = true THEN true
        ELSE false
        END AS is_interpolated_{target_col}
        """
        output_df = output_df.withColumn(
            f"is_interpolated_{target_col}", expr(expr_flag)
        )

        # Handle zero fill
        if method == "zero":
            output_df = output_df.withColumn(
                target_col,
                when(
                    col(f"is_interpolated_{target_col}") == False, col(target_col)
                ).otherwise(lit(0)),
            )

        # Handle null fill
        if method == "null":
            output_df = output_df.withColumn(
                target_col,
                when(
                    col(f"is_interpolated_{target_col}") == False, col(target_col)
                ).otherwise(None),
            )

        # Handle forward fill
        if method == "forward":
            output_df = output_df.withColumn(
                target_col,
                when(
                    col(f"is_interpolated_{target_col}") == True,
                    col(f"previous_{target_col}"),
                ).otherwise(col(target_col)),
            )
        # Handle backwards fill-  when target column is null use current value, otherwise use next value.
        if method == "back":
            output_df = output_df.withColumn(
                target_col,
                when(
                    (col(f"is_interpolated_{target_col}") == True)
                    & (
                        col(f"next_{target_col}").isNull()
                        & (col(f"{ts_col}_{target_col}").isNull())
                    ),
                    col(f"next_null_{target_col}"),
                ).otherwise(
                    when(
                        col(f"is_interpolated_{target_col}") == True,
                        col(f"next_{target_col}"),
                    ).otherwise(col(f"{target_col}"))
                ),
            )

        # Handle linear fill
        if method == "linear":
            output_df = self.__calc_linear_spark(
                output_df,
                ts_col,
                target_col,
            )

        return output_df

    def generate_time_series_fill(
        self, df: DataFrame, partition_cols: List[str], ts_col: str
    ) -> DataFrame:
        return df.withColumn("previous_timestamp", col(ts_col),).withColumn(
            "next_timestamp",
            lead(df[ts_col]).over(Window.partitionBy(*partition_cols).orderBy(ts_col)),
        )

    def generate_column_time_fill(
        self, df: DataFrame, partition_cols: List[str], ts_col: str, target_col: str
    ) -> DataFrame:
        return df.withColumn(
            f"previous_timestamp_{target_col}",
            last(col(f"{ts_col}_{target_col}"), ignorenulls=True).over(
                Window.partitionBy(*partition_cols)
                .orderBy(ts_col)
                .rowsBetween(-sys.maxsize, 0)
            ),
        ).withColumn(
            f"next_timestamp_{target_col}",
            first(col(f"{ts_col}_{target_col}"), ignorenulls=True).over(
                Window.partitionBy(*partition_cols)
                .orderBy(ts_col)
                .rowsBetween(0, sys.maxsize)
            ),
        )

    def generate_target_fill(
        self, df: DataFrame, partition_cols: List[str], ts_col: str, target_col: str
    ) -> DataFrame:
        return (
            df.withColumn(
                f"previous_{target_col}",
                last(df[target_col], ignorenulls=True).over(
                    Window.partitionBy(*partition_cols)
                    .orderBy(ts_col)
                    .rowsBetween(-sys.maxsize, 0)
                ),
            )
            .withColumn(
                f"next_null_{target_col}",
                first(df[target_col], ignorenulls=True).over(
                    Window.partitionBy(*partition_cols)
                    .orderBy(ts_col)
                    .rowsBetween(0, sys.maxsize)
                ),
            )
            .withColumn(
                f"next_{target_col}",
                lead(df[target_col]).over(
                    Window.partitionBy(*partition_cols).orderBy(ts_col)
                ),
            )
        )

    def interpolate(
        self,
        tsdf,
        ts_col: str,
        partition_cols: List[str],
        target_cols: List[str],
        freq: str,
        func: str,
        method: str,
        show_interpolated: bool,
    ) -> DataFrame:
        """
        Apply interpolation to TSDF.

        :param tsdf  - input TSDF
        :param target_cols   - numeric columns to interpolate
        :param freq  - frequency at which to sample
        :param func   - aggregate function for sampling
        :param fill   - interpolation function to fill missing values
        :param ts_col   - timestamp column name
        :param partition_cols  - partition columns names
        :param show_interpolated  - show if row is interpolated?
        """
        # Validate parameters
        self.__validate_fill(method)
        self.__validate_col(tsdf.df, partition_cols, target_cols, ts_col)

        # Resample and Normalize Input
        resampled_input: DataFrame = tsdf.resample(
            freq=freq, func=func, metricCols=target_cols
        ).df

        # Fill timeseries for nearest values
        time_series_filled = self.generate_time_series_fill(
            resampled_input, partition_cols, ts_col
        )

        # Generate surrogate timestamps for each target column
        add_column_time: DataFrame = time_series_filled
        for column in target_cols:
            add_column_time = add_column_time.withColumn(
                f"event_ts_{column}",
                when(col(column).isNull(), None).otherwise(col(ts_col)),
            )
            add_column_time = self.generate_column_time_fill(
                add_column_time, partition_cols, ts_col, column
            )

        # Handle Right Side Edge
        edge_filled = add_column_time.withColumn(
            "next_timestamp",
            when(
                col("next_timestamp").isNull(), expr(f"{ts_col}+ interval {freq}")
            ).otherwise(col("next_timestamp")),
        )

        # Fill target column for nearest values
        target_column_filled = edge_filled
        for column in target_cols:
            target_column_filled = self.generate_target_fill(
                target_column_filled, partition_cols, ts_col, column
            )

        # Generate missing timeseries values
        exploded_series = target_column_filled.withColumn(
            f"new_{ts_col}",
            expr(
                f"explode(sequence({ts_col}, next_timestamp - interval {freq}, interval {freq} )) as timestamp"
            ),
        )
        # Mark rows that are interpolated if flag is set to True
        flagged_series: DataFrame = exploded_series

        flagged_series = (
            exploded_series.withColumn(
                "is_ts_interpolated",
                when(col(f"new_{ts_col}") != col(ts_col), True).otherwise(False),
            )
            .withColumn(ts_col, col(f"new_{ts_col}"))
            .drop(col(f"new_{ts_col}"))
        )

        # # Perform interpolation on each target column
        interpolated_result: DataFrame = flagged_series
        for target_col in target_cols:
            # Interpolate target columns
            interpolated_result: DataFrame = self.__interpolate_column(
                freq, interpolated_result, ts_col, target_col, method
            )

            interpolated_result = interpolated_result.drop(
                f"previous_timestamp_{target_col}",
                f"next_timestamp_{target_col}",
                f"previous_{target_col}",
                f"next_{target_col}",
                f"next_null_{target_col}",
                f"{ts_col}_{target_col}",
            )

        output: DataFrame = interpolated_result.drop(
            "previous_timestamp", "next_timestamp"
        )

        if show_interpolated is False:
            interpolated_col_names = ["is_ts_interpolated"]
            for column in target_cols:
                interpolated_col_names.append(f"is_interpolated_{column}")
            output = output.drop(*interpolated_col_names)
        return output
