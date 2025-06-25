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

    df = df.withColumn("agg_key", agg_window)

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

    :param tsdf: input TSDF object
    :param freq: downsample to this frequency
    :param func: aggregate function
    :param metricCols: columns used for aggregates

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
    :param tsdf: input TSDF object
    :param func: aggregate function
    :param metricCols: columns used for aggregates
    :param prefix: the metric columns with the aggregate named function
    :param fill: upsample based on the time increment for 0s in numeric columns
    :return: TSDF object with newly aggregated timestamp as ts_col with aggregated values
    """
    tsdf, period, unit = _appendAggKey(tsdf, freq)

    df = tsdf.df

    groupingCols = tsdf.series_ids + ["agg_key"]

    if metricCols is None:
        metricCols = tsdf.metric_cols

    if prefix is None:
        prefix = ""
    else:
        prefix = prefix + "_"

    groupingCols = [sfn.col(column) for column in groupingCols]

    if func == floor:
        metricCol = sfn.struct(*([tsdf.ts_col] + metricCols))
        res = df.withColumn("struct_cols", metricCol).groupBy(groupingCols)
        res = res.agg(sfn.min("struct_cols").alias("closest_data")).select(
            *groupingCols, sfn.col("closest_data.*")
        )
        new_cols = [sfn.col(tsdf.ts_col)] + [
            sfn.col(c).alias("{}".format(prefix) + c) for c in metricCols
        ]
        res = res.select(*groupingCols, *new_cols)
    elif func == average:
        exprs = {x: "avg" for x in metricCols}
        res = df.groupBy(groupingCols).agg(exprs)
        agg_metric_cls = list(
            set(res.columns).difference(set(tsdf.series_ids + [tsdf.ts_col, "agg_key"]))
        )
        new_cols = [
            sfn.col(c).alias(
                "{}".format(prefix) + (c.split("avg(")[1]).replace(")", "")
            )
            for c in agg_metric_cls
        ]
        res = res.select(*groupingCols, *new_cols)
    elif func == min:
        exprs = {x: "min" for x in metricCols}
        res = df.groupBy(groupingCols).agg(exprs)
        agg_metric_cls = list(
            set(res.columns).difference(set(tsdf.series_ids + [tsdf.ts_col, "agg_key"]))
        )
        new_cols = [
            sfn.col(c).alias(
                "{}".format(prefix) + (c.split("min(")[1]).replace(")", "")
            )
            for c in agg_metric_cls
        ]
        res = res.select(*groupingCols, *new_cols)
    elif func == max:
        exprs = {x: "max" for x in metricCols}
        res = df.groupBy(groupingCols).agg(exprs)
        agg_metric_cls = list(
            set(res.columns).difference(set(tsdf.series_ids + [tsdf.ts_col, "agg_key"]))
        )
        new_cols = [
            sfn.col(c).alias(
                "{}".format(prefix) + (c.split("max(")[1]).replace(")", "")
            )
            for c in agg_metric_cls
        ]
        res = res.select(*groupingCols, *new_cols)
    elif func == ceiling:
        metricCol = sfn.struct([tsdf.ts_col] + metricCols)
        res = df.withColumn("struct_cols", metricCol).groupBy(groupingCols)
        res = res.agg(sfn.max("struct_cols").alias("ceil_data")).select(
            *groupingCols, sfn.col("ceil_data.*")
        )
        new_cols = [sfn.col(tsdf.ts_col)] + [
            sfn.col(c).alias("{}".format(prefix) + c) for c in metricCols
        ]
        res = res.select(*groupingCols, *new_cols)

    # aggregate by the window and drop the end time (use start time as new ts_col)
    res = (
        res.drop(tsdf.ts_col)
        .withColumnRenamed("agg_key", tsdf.ts_col)
        .withColumn(tsdf.ts_col, sfn.col(tsdf.ts_col).start)
    )

    # sort columns so they are consistent
    non_part_cols = set(set(res.columns) - set(tsdf.series_ids)) - {tsdf.ts_col}
    sel_and_sort = tsdf.series_ids + [tsdf.ts_col] + sorted(non_part_cols)
    res = res.select(sel_and_sort)

    fillW = tsdf.baseWindow()

    imputes = (
        res.select(
            *tsdf.series_ids,
            sfn.min(tsdf.ts_col).over(fillW).alias("from"),
            sfn.max(tsdf.ts_col).over(fillW).alias("until"),
        )
        .distinct()
        .withColumn(
            tsdf.ts_col,
            sfn.explode(
                sfn.expr("sequence(from, until, interval {} {})".format(period, unit))
            ),
        )
        .drop("from", "until")
    )

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
    :param freq: frequency for upsample - valid inputs are "hr", "min", "sec" corresponding to hour, minute, or second
    :param func: function used to aggregate input
    :param metricCols supply a smaller list of numeric columns if the entire set of numeric columns should not be returned for the resample function
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
        _resample_freq=freq,
        _resample_func=func,
    )

