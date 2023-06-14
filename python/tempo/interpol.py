from __future__ import annotations

from typing import Callable, List, Optional, Union

from pyspark.sql.dataframe import DataFrame
import pyspark.sql.functions as sfn

import tempo.resample as t_resample
import tempo.tsdf as t_tsdf
import tempo.utils as t_utils

# Interpolation fill options
method_options = ["zero", "null", "bfill", "ffill", "linear"]
supported_target_col_types = ["int", "bigint", "float", "double"]


class Interpolation:
    def __init__(self, is_resampled: bool):
        self.is_resampled = is_resampled

    def __validate_fill(self, method: str) -> None:
        """
        Validate if the fill provided is within the allowed list of values.

        :param fill: Fill type e.g. "zero", "null", "bfill", "ffill", "linear"
        """
        if method not in method_options:
            raise ValueError(
                f"Please select from one of the following fill options: {method_options}"
            )

    def __calc_linear_spark(
        self, df: DataFrame, ts_col: str, target_col: str
    ) -> DataFrame:
        """
        Native Spark function for calculating linear interpolation on a DataFrame.

        :param df: prepared dataframe to be interpolated
        :param ts_col: timeseries column name
        :param target_col: column to be interpolated
        """
        interpolation_expr = f"""
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

        # remove target column to avoid duplication during interpolation expression
        cols: List[str] = df.columns
        cols.remove(target_col)
        interpolated: DataFrame = df.selectExpr(*cols, interpolation_expr)
        # Preserve column order
        return interpolated.select(*df.columns)

    def __interpolate_column(
        self,
        series: DataFrame,
        ts_col: str,
        target_col: str,
        method: str,
    ) -> DataFrame:
        """
        Apply interpolation to column.

        :param series: input DataFrame
        :param ts_col: timestamp column name
        :param target_col: column to interpolate
        :param method: interpolation function to fill missing values
        """
        output_df: DataFrame = series

        # create new column for if target column is interpolated
        flag_expr = f"""
        CASE WHEN {target_col} is null and is_ts_interpolated = false THEN true
             WHEN is_ts_interpolated = true THEN true
        ELSE false
        END AS is_interpolated_{target_col}
        """
        output_df = output_df.withColumn(
            f"is_interpolated_{target_col}", sfn.expr(flag_expr)
        )

        # Handle zero fill
        if method == "zero":
            output_df = output_df.withColumn(
                target_col,
                sfn.when(
                    sfn.col(f"is_interpolated_{target_col}") == False,  # noqa: E712
                    sfn.col(target_col),
                ).otherwise(sfn.lit(0)),
            )

        # Handle null fill
        if method == "null":
            output_df = output_df.withColumn(
                target_col,
                sfn.when(
                    sfn.col(f"is_interpolated_{target_col}") == False,  # noqa: E712
                    sfn.col(target_col),
                ).otherwise(None),
            )

        # Handle forward fill
        if method == "ffill":
            output_df = output_df.withColumn(
                target_col,
                sfn.when(
                    sfn.col(f"is_interpolated_{target_col}") == True,  # noqa: E712
                    sfn.col(f"previous_{target_col}"),
                ).otherwise(sfn.col(target_col)),
            )
        # Handle backwards fill
        if method == "bfill":
            output_df = output_df.withColumn(
                target_col,
                # Handle case when subsequent value is null
                sfn.when(
                    (sfn.col(f"is_interpolated_{target_col}") == True)  # noqa: E712
                    & (
                        sfn.col(f"next_{target_col}").isNull()
                        & (sfn.col(f"{ts_col}_{target_col}").isNull())
                    ),
                    sfn.col(f"next_null_{target_col}"),
                ).otherwise(
                    # Handle standard backwards fill
                    sfn.when(
                        sfn.col(f"is_interpolated_{target_col}") == True,  # noqa: E712
                        sfn.col(f"next_{target_col}"),
                    ).otherwise(sfn.col(f"{target_col}"))
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

    def __generate_time_series_fill(self, tsdf: t_tsdf.TSDF) -> t_tsdf.TSDF:
        """
        Create additional timeseries columns for previous and next timestamps

        :param df: input DataFrame
        :param partition_cols: partition column names
        :param ts_col: timestamp column name
        """
        return tsdf.withColumn(
            "previous_timestamp",
            sfn.col(tsdf.ts_col),
        ).withColumn(
            "next_timestamp",
            sfn.lead(sfn.col(tsdf.ts_col)).over(tsdf.baseWindow()))

    def __generate_column_time_fill(
        self,
        tsdf: t_tsdf.TSDF,
        target_col: str,
    ) -> t_tsdf.TSDF:
        """
        Create timeseries columns for previous and next timestamps for a specific target column

        :param df: input DataFrame
        :param partition_cols: partition column names
        :param ts_col: timestamp column name
        :param target_col: target column name
        """

        # define forward & backward windows
        fwd_win = tsdf.allBeforeWindow()
        bkwd_win = tsdf.allAfterWindow()

        return tsdf.withColumn(
            f"previous_timestamp_{target_col}",
            sfn.last(sfn.col(f"{tsdf.ts_col}_{target_col}"),
                     ignorenulls=True).over(fwd_win),
        ).withColumn(
            f"next_timestamp_{target_col}",
            sfn.last(sfn.col(f"{tsdf.ts_col}_{target_col}"),
                     ignorenulls=True).over(bkwd_win)
        )

    def __generate_target_fill(
        self,
        tsdf: t_tsdf.TSDF,
        target_col: str,
    ) -> t_tsdf.TSDF:
        """
        Create columns for previous and next value for a specific target column

        :param df: input DataFrame
        :param partition_cols: partition column names
        :param ts_col: timestamp column name
        :param target_col: target column name
        """

        # define forward & backward windows
        fwd_win = tsdf.allBeforeWindow()
        bkwd_win = tsdf.allAfterWindow()

        return (
            tsdf.withColumn(
                f"previous_{target_col}",
                sfn.last(sfn.col(target_col), ignorenulls=True).over(fwd_win)
            )
            # Handle if subsequent value is null
            .withColumn(
                f"next_null_{target_col}",
                sfn.last(sfn.col(target_col), ignorenulls=True).over(bkwd_win)
            ).withColumn(
                f"next_{target_col}",
                sfn.lead(sfn.col(target_col)).over(fwd_win)
            )
        )

    def interpolate(
        self,
        tsdf: t_tsdf.TSDF,
        target_cols: List[str],
        freq: Optional[str],
        func: Optional[Union[Callable | str]],
        method: str,
        show_interpolated: bool,
        perform_checks: bool = True,
    ) -> DataFrame:
        """
        Apply interpolation function.

        :param tsdf: input TSDF
        :param target_cols: numeric columns to interpolate
        :param freq: frequency at which to sample
        :param func: aggregate function used for sampling to the specified interval
        :param method: interpolation function used to fill missing values
        :param show_interpolated: show if row is interpolated?
        :param perform_checks: calculate time horizon and warnings if True (default is True)
        :return: DataFrame containing interpolated data.
        """
        # Validate input parameters
        self.__validate_fill(method)

        if freq is None:
            raise ValueError("freq cannot be None")

        if func is None:
            raise ValueError("func cannot be None")

        if callable(func):
            raise ValueError("func must be a string")

        # Convert Frequency using resample dictionary
        parsed_freq = t_resample.checkAllowableFreq(freq)
        period, unit = parsed_freq[0], parsed_freq[1]
        freq = f"{period} {t_resample.freq_dict[unit]}"  # type: ignore[literal-required]

        # Throw warning for user to validate that the expected number of output rows is valid.
        if perform_checks:
            t_utils.calculate_time_horizon(tsdf, freq)

        # Only select required columns for interpolation
        input_cols: List[str] = [tsdf.ts_col, *target_cols]
        if tsdf.series_ids is not None:
            input_cols += [*tsdf.series_ids]

        sampled_input: t_tsdf.TSDF = tsdf.select(*input_cols)

        if self.is_resampled is False:
            # Resample and Normalize Input
            sampled_input = tsdf.resample(freq=freq,
                                          func=func,
                                          metricCols=target_cols)

        # Fill timeseries for nearest values
        time_series_filled = self.__generate_time_series_fill(sampled_input)

        # Generate surrogate timestamps for each target column
        # This is required if multuple columns are being interpolated and may contain nulls
        add_column_time: t_tsdf.TSDF = time_series_filled
        for column in target_cols:
            add_column_time = add_column_time.withColumn(
                f"{tsdf.ts_col}_{column}",
                sfn.when(sfn.col(column).isNull(), None).otherwise(sfn.col(tsdf.ts_col)),
            )
            add_column_time = self.__generate_column_time_fill(
                add_column_time, column
            )

        # Handle edge case if last value (latest) is null
        edge_filled = add_column_time.withColumn(
            "next_timestamp",
            sfn.when(
                sfn.col("next_timestamp").isNull(),
                sfn.expr(f"{tsdf.ts_col}+ interval {freq}"),
            ).otherwise(sfn.col("next_timestamp")),
        )

        # Fill target column for nearest values
        target_column_filled = edge_filled
        for column in target_cols:
            target_column_filled = self.__generate_target_fill(
                target_column_filled, column
            )

        # Generate missing timeseries values
        exploded_series = target_column_filled.withColumn(
            f"new_{tsdf.ts_col}",
            sfn.expr(
                f"explode(sequence({tsdf.ts_col}, next_timestamp - interval {freq}, interval {freq} )) as timestamp"
            ),
        )
        # Mark rows that are interpolated if flag is set to True
        flagged_series: DataFrame = exploded_series

        flagged_series = (
            exploded_series.withColumn(
                "is_ts_interpolated",
                sfn.when(sfn.col(f"new_{tsdf.ts_col}") != sfn.col(tsdf.ts_col), True).otherwise(
                    False
                ),
            )
            .withColumn(tsdf.ts_col, sfn.col(f"new_{tsdf.ts_col}"))
            .drop(sfn.col(f"new_{tsdf.ts_col}"))
        )

        # # Perform interpolation on each target column
        interpolated_result: DataFrame = flagged_series
        for target_col in target_cols:
            # Interpolate target columns
            interpolated_result = self.__interpolate_column(
                interpolated_result, tsdf.ts_col, target_col, method
            )

            interpolated_result = interpolated_result.drop(
                f"previous_timestamp_{target_col}",
                f"next_timestamp_{target_col}",
                f"previous_{target_col}",
                f"next_{target_col}",
                f"next_null_{target_col}",
                f"{tsdf.ts_col}_{target_col}",
            )

        # Remove non-required columns
        output: DataFrame = interpolated_result.drop(
            "previous_timestamp", "next_timestamp"
        )

        # Hide is_interpolated columns based on flag
        if show_interpolated is False:
            interpolated_col_names = ["is_ts_interpolated"]
            for column in target_cols:
                interpolated_col_names.append(f"is_interpolated_{column}")
            output = output.drop(*interpolated_col_names)

        return output
