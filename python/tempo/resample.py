from __future__ import annotations

import warnings
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple,
    Union,
)

import pyspark.sql.functions as sfn
from pyspark.sql import DataFrame

import tempo.intervals as t_int
import tempo.tsdf as t_tsdf
from tempo.resample_utils import (
    ALLOWED_FREQ_KEYS,
    FreqDict,
    average,
    ceiling,
    checkAllowableFreq,
    floor,
    freq_dict,
    is_valid_allowed_freq_keys,
    max,
    min,
    validateFuncExists,
)

# Column name constants
AGG_KEY = "agg_key"


def _appendAggKey(
    tsdf: t_tsdf.TSDF, freq: Optional[str] = None
) -> Tuple[t_tsdf.TSDF, int | str, Any]:
    """
    :param tsdf: TSDF object as input
    :param freq: frequency at which to upsample
    :return: triple - 1) return a TSDF with a new aggregate key (called agg_key) 2) return the period for use in interpolation, 3) return the time increment (also necessary for interpolation)
    """
    df = tsdf.df
    parsed_freq = checkAllowableFreq(freq)
    period, unit = parsed_freq[0], parsed_freq[1]

    agg_window = sfn.window(
        sfn.col(tsdf.ts_col),
        "{} {}".format(period, freq_dict[unit]),  # type: ignore[literal-required]
    )

    df = df.withColumn(AGG_KEY, agg_window)

    return (
        t_tsdf.TSDF(df, ts_col=tsdf.ts_col, series_ids=tsdf.series_ids),
        period,
        freq_dict[unit],  # type: ignore[literal-required]
    )


class ResampleWarning(Warning):
    """
    This class is a warning that is raised when the interpolate or resample with fill methods are called.
    """

    pass


def calculate_time_horizon(
    tsdf: t_tsdf.TSDF,
    freq: str,
    local_freq_dict: Optional[FreqDict] = None,
) -> None:
    # Convert Frequency using resample dictionary
    if local_freq_dict is None:
        local_freq_dict = freq_dict
    parsed_freq = checkAllowableFreq(freq)
    period, unit = parsed_freq[0], parsed_freq[1]
    if is_valid_allowed_freq_keys(
        unit,
        ALLOWED_FREQ_KEYS,
    ):
        freq = f"{period} {local_freq_dict[unit]}"  # type: ignore[literal-required]
    else:
        raise ValueError(f"Frequency {unit} not supported")

    # Get max and min timestamp per partition
    if tsdf.series_ids:
        grouped_df = tsdf.df.groupBy(*tsdf.series_ids)
    else:
        grouped_df = tsdf.df.groupBy()
    ts_range_per_series: DataFrame = grouped_df.agg(
        sfn.max(tsdf.ts_col).alias("max_ts"),
        sfn.min(tsdf.ts_col).alias("min_ts"),
    )

    # Generate upscale metrics
    normalized_time_df: DataFrame = (
        ts_range_per_series.withColumn("min_epoch_ms", sfn.expr("unix_millis(min_ts)"))
        .withColumn("max_epoch_ms", sfn.expr("unix_millis(max_ts)"))
        .withColumn(
            "interval_ms",
            sfn.expr(
                f"unix_millis(cast('1970-01-01 00:00:00.000+0000' as TIMESTAMP) + INTERVAL {freq})"
            ),
        )
        .withColumn(
            "rounded_min_epoch",
            sfn.expr("min_epoch_ms - (min_epoch_ms % interval_ms)"),
        )
        .withColumn(
            "rounded_max_epoch",
            sfn.expr("max_epoch_ms - (max_epoch_ms % interval_ms)"),
        )
        .withColumn("diff_ms", sfn.expr("rounded_max_epoch - rounded_min_epoch"))
        .withColumn("num_values", sfn.expr("(diff_ms/interval_ms) +1"))
    )

    (
        min_ts,
        max_ts,
        min_value_partition,
        max_value_partition,
        p25_value_partition,
        p50_value_partition,
        p75_value_partition,
        total_values,
    ) = normalized_time_df.select(
        sfn.min("min_ts"),
        sfn.max("max_ts"),
        sfn.min("num_values"),
        sfn.max("num_values"),
        sfn.percentile_approx("num_values", 0.25),
        sfn.percentile_approx("num_values", 0.5),
        sfn.percentile_approx("num_values", 0.75),
        sfn.sum("num_values"),
    ).first()

    warnings.simplefilter("always", ResampleWarning)
    warnings.warn(
        f"""
            Resample Metrics Warning:
                Earliest Timestamp: {min_ts}
                Latest Timestamp: {max_ts}
                No. of Unique Partitions: {normalized_time_df.count()}
                Resampled Min No. Values in Single a Partition: {min_value_partition}
                Resampled Max No. Values in Single a Partition: {max_value_partition}
                Resampled P25 No. Values in Single a Partition: {p25_value_partition}
                Resampled P50 No. Values in Single a Partition: {p50_value_partition}
                Resampled P75 No. Values in Single a Partition: {p75_value_partition}
                Resampled Total No. Values Across All Partitions: {total_values}
        """,
        ResampleWarning,
    )


def downsample(
    tsdf: t_tsdf.TSDF,
    freq: str,
    func: Union[Callable, str],
    metricCols: Optional[List[str]] = None,
) -> t_int.IntervalsDF:
    """
    Downsample a TSDF object to a lower frequency
    
    Note: This function uses a simpler aggregation approach than aggregate().
    If metricCols is None, it defaults to tsdf.metric_cols (numeric columns only).
    Non-metric observational columns are not preserved in the output.

    :param tsdf: input TSDF object
    :param freq: downsample to this frequency
    :param func: aggregate function
    :param metricCols: columns used for aggregates. If None, uses numeric columns only.

    :return: IntervalsDF object with the aggregated values
    """
    if metricCols is None:
        metricCols = tsdf.metric_cols
    if isinstance(func, Callable):
        agg_exprs = [func(col).alias(col) for col in metricCols]
    else:
        agg_exprs = {col: func for col in metricCols}
    return tsdf.aggByCycles(freq, *agg_exprs)


def aggregate(
    tsdf: t_tsdf.TSDF,
    freq: str,
    func: Union[Callable, str],
    metricCols: Optional[List[str]] = None,
    prefix: Optional[str] = None,
    fill: Optional[bool] = None,
) -> t_tsdf.TSDF:
    """
    aggregate a data frame by a coarser timestamp than the initial TSDF ts_col
    
    Column Handling Behavior:
    ------------------------
    This function follows the "explicit is better than implicit" principle for column selection,
    which aligns with industry best practices from pandas, Flint, and other time series libraries.
    
    1. When metricCols is None (default):
       - For column-wise operations (min, max, average): Applies the operation to ALL observational columns
       - For row-wise operations (floor, ceiling): Preserves ALL observational columns in the output
       - This ensures no data is accidentally lost during aggregation
    
    2. When metricCols is explicitly provided:
       - For column-wise operations: Only the specified columns are aggregated and returned
       - For row-wise operations: Only the specified columns are included in the output
       - Non-metric observational columns are NOT preserved
       - This gives users precise control over which columns appear in the output
    
    This design allows users to:
    - Get comprehensive results by default (all columns preserved)
    - Optimize performance and output size when they know exactly which columns they need
    
    :param tsdf: input TSDF object
    :param func: aggregate function (min, max, average/mean, floor, ceiling)
    :param metricCols: columns used for aggregates. If None, uses all observational columns
    :param prefix: the metric columns with the aggregate named function
    :param fill: upsample based on the time increment for 0s in numeric columns
    :return: TSDF object with newly aggregated timestamp as ts_col with aggregated values
    """
    tsdf, period, unit = _appendAggKey(tsdf, freq)

    df = tsdf.df

    groupingCols = tsdf.series_ids + [AGG_KEY]

    # Track whether metricCols was explicitly provided
    # This is crucial for determining column handling behavior
    metric_cols_provided = metricCols is not None
    
    if metricCols is None:
        # Default behavior: use all metric columns (numeric observational columns)
        # This ensures comprehensive aggregation without data loss
        metricCols = tsdf.metric_cols

    if prefix is None:
        prefix = ""
    else:
        prefix = prefix + "_"

    groupingCols = [sfn.col(column) for column in groupingCols]

    if func == floor:
        # Floor is a row-wise operation: it selects the entire row with the earliest timestamp
        # in each aggregation window, preserving relationships between columns
        if metric_cols_provided:
            # Explicit column selection: only include the requested columns
            # This allows users to optimize output size and performance
            cols_to_include = [col for col in metricCols if col in df.columns]
        else:
            # Default behavior: preserve all observational columns to avoid data loss
            # This matches pandas resample().first() behavior
            cols_to_include = [col for col in tsdf.observational_cols if col in df.columns and col != AGG_KEY]

        metricCol = sfn.struct(*([tsdf.ts_col] + cols_to_include))
        res = df.withColumn("struct_cols", metricCol).groupBy(groupingCols)
        res = res.agg(sfn.min("struct_cols").alias("closest_data")).select(
            *groupingCols, sfn.col("closest_data.*")
        )
        new_cols = [sfn.col(tsdf.ts_col)] + [
            sfn.col(c).alias("{}".format(prefix) + c) for c in cols_to_include
        ]
        res = res.select(*groupingCols, *new_cols)
    elif func == average:
        # Average is a column-wise operation: it computes the mean for each column independently
        # This preserves the semantic meaning of each metric
        if metric_cols_provided:
            # Explicit selection: only average the specified columns
            # Non-specified columns are excluded from the output
            cols_to_avg = metricCols
        else:
            # Default behavior: average all observational columns
            # This ensures comprehensive statistics without user needing to list every column
            cols_to_avg = [col for col in tsdf.observational_cols if col in df.columns and col != AGG_KEY]

        exprs = {x: "avg" for x in cols_to_avg}
        res = df.groupBy(groupingCols).agg(exprs)
        agg_metric_cls = list(
            set(res.columns).difference(set(tsdf.series_ids + [tsdf.ts_col, AGG_KEY]))
        )
        new_cols = [
            sfn.col(c).alias(
                "{}".format(prefix) + (c.split("avg(")[1]).replace(")", "")
            )
            for c in agg_metric_cls
        ]
        res = res.select(*groupingCols, *new_cols)
    elif func == min:
        # Min is a column-wise operation: it finds the minimum value for each column independently
        # Unlike floor (which preserves row relationships), min may return values from different rows
        agg_exprs = []

        # Always compute min for the specified metric columns
        for col in metricCols:
            agg_exprs.append(sfn.min(col).alias("{}{}".format(prefix, col)))

        # Preserve non-metric observational columns only in default mode
        # This prevents accidental data loss while allowing explicit column filtering
        if not metric_cols_provided:
            # In default mode, preserve other columns using first() to maintain data completeness
            # Note: first() is used because these columns aren't being aggregated
            non_metric_obs_cols = [col for col in tsdf.observational_cols
                                   if col not in metricCols and col in df.columns and col != AGG_KEY]
            for col in non_metric_obs_cols:
                agg_exprs.append(sfn.first(col).alias("{}{}".format(prefix, col)))

        res = df.groupBy(groupingCols).agg(*agg_exprs)
    elif func == max:
        # Max is a column-wise operation: it finds the maximum value for each column independently
        # This is semantically different from ceiling, which preserves row relationships
        agg_exprs = []

        # Always compute max for the specified metric columns
        for col in metricCols:
            agg_exprs.append(sfn.max(col).alias("{}{}".format(prefix, col)))

        # Preserve non-metric observational columns only in default mode
        # This follows the same pattern as min for consistency
        if not metric_cols_provided:
            # In default mode, preserve other columns to avoid data loss
            # This matches the behavior users expect from pandas/Flint
            non_metric_obs_cols = [col for col in tsdf.observational_cols
                                   if col not in metricCols and col in df.columns and col != AGG_KEY]
            for col in non_metric_obs_cols:
                agg_exprs.append(sfn.first(col).alias("{}{}".format(prefix, col)))

        res = df.groupBy(groupingCols).agg(*agg_exprs)
    elif func == ceiling:
        # Ceiling is a row-wise operation: it selects the entire row with the latest timestamp
        # in each aggregation window, preserving relationships between columns
        if metric_cols_provided:
            # Explicit column selection: only include the requested columns
            # This mirrors the floor behavior for consistency
            cols_to_include = [col for col in metricCols if col in df.columns]
        else:
            # Default behavior: preserve all observational columns
            # This ensures the full context of the latest observation is maintained
            cols_to_include = [col for col in tsdf.observational_cols if col in df.columns and col != AGG_KEY]

        metricCol = sfn.struct([tsdf.ts_col] + cols_to_include)
        res = df.withColumn("struct_cols", metricCol).groupBy(groupingCols)
        res = res.agg(sfn.max("struct_cols").alias("ceil_data")).select(
            *groupingCols, sfn.col("ceil_data.*")
        )
        new_cols = [sfn.col(tsdf.ts_col)] + [
            sfn.col(c).alias("{}".format(prefix) + c) for c in cols_to_include
        ]
        res = res.select(*groupingCols, *new_cols)

    # aggregate by the window and drop the end time (use start time as new ts_col)
    res = (
        res.drop(tsdf.ts_col)
        .withColumnRenamed(AGG_KEY, tsdf.ts_col)
        .withColumn(tsdf.ts_col, sfn.col(tsdf.ts_col).start)
    )

    # sort columns so they are consistent
    non_part_cols = set(set(res.columns) - set(tsdf.series_ids)) - {tsdf.ts_col}
    sel_and_sort = tsdf.series_ids + [tsdf.ts_col] + sorted(non_part_cols)
    res = res.select(sel_and_sort)

    # Calculate time bounds per partition without using window functions to avoid duplicates
    if tsdf.series_ids:
        # Group by series to get min/max per partition
        time_bounds = res.groupBy(*tsdf.series_ids).agg(
            sfn.min(tsdf.ts_col).alias("from"),
            sfn.max(tsdf.ts_col).alias("until")
        )
    else:
        # No series_ids, so get global min/max
        time_bounds = res.agg(
            sfn.min(tsdf.ts_col).alias("from"),
            sfn.max(tsdf.ts_col).alias("until")
        )

    # Generate sequence of timestamps for filling
    imputes = time_bounds.withColumn(
        tsdf.ts_col,
        sfn.explode(
            sfn.expr("sequence(from, until, interval {} {})".format(period, unit))
        ),
    ).drop("from", "until")

    metrics = []
    for col in res.dtypes:
        if col[1] in ["long", "double", "decimal", "integer", "float", "int"]:
            metrics.append(col[0])

    if fill:
        res = imputes.join(res, tsdf.series_ids + [tsdf.ts_col], "leftouter").na.fill(
            0, metrics
        )

    return res


def resample(
    tsdf: t_tsdf.TSDF,
    freq: str,
    func: Union[Callable | str],
    metricCols: Optional[List[str]] = None,
    prefix: Optional[str] = None,
    fill: Optional[bool] = None,
    perform_checks: bool = True,
) -> t_tsdf.TSDF:
    """
    function to upsample based on frequency and aggregate function similar to pandas
    
    Note on Column Handling:
    -----------------------
    This function delegates column handling behavior to the aggregate() function.
    See aggregate() documentation for detailed explanation of how columns are handled
    based on whether metricCols is None (default) or explicitly provided.
    
    :param freq: frequency for upsample - valid inputs are "hr", "min", "sec" corresponding to hour, minute, or second
    :param func: function used to aggregate input
    :param metricCols: supply a smaller list of numeric columns if the entire set of numeric columns should not be 
                       returned for the resample function. If None, all observational columns are included.
    :param prefix - supply a prefix for the newly sampled columns
    :param fill - Boolean - set to True if the desired output should contain filled in gaps (with 0s currently)
    :param perform_checks: calculate time horizon and warnings if True (default is True)
    :return: TSDF object with sample data using aggregate function
    """
    validateFuncExists(func)

    # Throw warning for user to validate that the expected number of output rows is valid.
    if fill is True and perform_checks is True:
        calculate_time_horizon(tsdf, freq)

    enriched_df: DataFrame = aggregate(tsdf, freq, func, metricCols, prefix, fill)

    # Import TSDF here to avoid circular import
    from tempo.tsdf import TSDF

    return TSDF(
        enriched_df,
        ts_col=tsdf.ts_col,
        series_ids=tsdf.series_ids,
        resample_freq=freq,
        resample_func=func,
    )

