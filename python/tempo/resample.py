from __future__ import annotations

from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple,
    TypedDict,
    Union,
    get_type_hints,
)

import pyspark.sql.functions as sfn
from pyspark.sql import DataFrame
from pyspark.sql.window import Window

import tempo.tsdf as t_tsdf

# define global frequency options
MUSEC = "microsec"
MS = "ms"
SEC = "sec"
MIN = "min"
HR = "hr"
DAY = "day"

# define global aggregate function options for downsampling
floor = "floor"
min = "min"
max = "max"
average = "mean"
ceiling = "ceil"


class FreqDict(TypedDict):
    musec: str
    microsec: str
    microsecond: str
    microseconds: str
    ms: str
    millisecond: str
    milliseconds: str
    sec: str
    second: str
    seconds: str
    min: str
    minute: str
    minutes: str
    hr: str
    hour: str
    hours: str
    day: str
    days: str


freq_dict: FreqDict = {
    "musec": "microseconds",
    "microsec": "microseconds",
    "microsecond": "microseconds",
    "microseconds": "microseconds",
    "ms": "milliseconds",
    "millisecond": "milliseconds",
    "milliseconds": "milliseconds",
    "sec": "seconds",
    "second": "seconds",
    "seconds": "seconds",
    "min": "minutes",
    "minute": "minutes",
    "minutes": "minutes",
    "hr": "hours",
    "hour": "hours",
    "hours": "hours",
    "day": "days",
    "days": "days",
}

ALLOWED_FREQ_KEYS: List[str] = list(get_type_hints(FreqDict).keys())


def is_valid_allowed_freq_keys(val: str, literal_constant: List[str]) -> bool:
    return val in literal_constant


allowableFreqs = [MUSEC, MS, SEC, MIN, HR, DAY]
allowableFuncs = [floor, min, max, average, ceiling]


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
        sfn.col(tsdf.ts_col), "{} {}".format(period, freq_dict[unit])  # type: ignore[literal-required]
    )

    df = df.withColumn("agg_key", agg_window)

    return (
        t_tsdf.TSDF(df, tsdf.ts_col, partition_cols=tsdf.partitionCols),
        period,
        freq_dict[unit],  # type: ignore[literal-required]
    )


def aggregate(
    tsdf: t_tsdf.TSDF,
    freq: str,
    func: Union[Callable, str],
    metricCols: Optional[List[str]] = None,
    prefix: Optional[str] = None,
    fill: Optional[bool] = None,
) -> DataFrame:
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

    groupingCols = tsdf.partitionCols + ["agg_key"]

    if metricCols is None:
        metricCols = list(set(df.columns).difference(set(groupingCols + [tsdf.ts_col])))

    if prefix is None:
        prefix = ""
    else:
        prefix = prefix + "_"

    groupingCols = [sfn.col(column) for column in groupingCols]

    if func == floor:
        metricCol = sfn.struct([tsdf.ts_col] + metricCols)
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
            set(res.columns).difference(
                set(tsdf.partitionCols + [tsdf.ts_col, "agg_key"])
            )
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
            set(res.columns).difference(
                set(tsdf.partitionCols + [tsdf.ts_col, "agg_key"])
            )
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
            set(res.columns).difference(
                set(tsdf.partitionCols + [tsdf.ts_col, "agg_key"])
            )
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
    non_part_cols = set(set(res.columns) - set(tsdf.partitionCols)) - set([tsdf.ts_col])
    sel_and_sort = tsdf.partitionCols + [tsdf.ts_col] + sorted(non_part_cols)
    res = res.select(sel_and_sort)

    fillW = Window.partitionBy(tsdf.partitionCols)

    imputes = (
        res.select(
            *tsdf.partitionCols,
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
        res = imputes.join(
            res, tsdf.partitionCols + [tsdf.ts_col], "leftouter"
        ).na.fill(0, metrics)

    return res


def checkAllowableFreq(freq: Optional[str]) -> Tuple[Union[int | str], str]:
    """
    Parses frequency and checks against allowable frequencies
    :param freq: frequncy at which to upsample/downsample, declared in resample function
    :return: list of parsed frequency value and time suffix
    """
    if not isinstance(freq, str):
        raise TypeError(f"Invalid type for `freq` argument: {freq}.")

    # TODO - return either int OR str for first argument
    allowable_freq: Tuple[Union[int | str], str] = (
        0,
        "will_always_fail_if_not_overwritten",
    )

    if is_valid_allowed_freq_keys(
        freq.lower(),
        ALLOWED_FREQ_KEYS,
    ):
        allowable_freq = 1, freq
        return allowable_freq

    try:
        periods = freq.lower().split(" ")[0].strip()
        units = freq.lower().split(" ")[1].strip()
    except IndexError:
        raise ValueError(
            "Allowable grouping frequencies are microsecond (musec), millisecond (ms), sec (second), min (minute), hr (hour), day. Reformat your frequency as <integer> <day/hour/minute/second>"
        )

    if is_valid_allowed_freq_keys(
        units.lower(),
        ALLOWED_FREQ_KEYS,
    ):
        if units.startswith(MUSEC):
            allowable_freq = periods, MUSEC
        elif units.startswith(MS) | units.startswith("millis"):
            allowable_freq = periods, MS
        elif units.startswith(SEC):
            allowable_freq = periods, SEC
        elif units.startswith(MIN):
            allowable_freq = periods, MIN
        elif units.startswith("hour") | units.startswith(HR):
            allowable_freq = periods, "hour"
        elif units.startswith(DAY):
            allowable_freq = periods, DAY
    else:
        raise ValueError(f"Invalid value for `freq` argument: {freq}.")

    return allowable_freq


def validateFuncExists(func: Union[Callable | str]) -> None:
    if func is None:
        raise TypeError(
            "Aggregate function missing. Provide one of the allowable functions: "
            + ", ".join(allowableFuncs)
        )
    elif func not in allowableFuncs:
        raise ValueError(
            "Aggregate function is not in the valid list. Provide one of the allowable functions: "
            + ", ".join(allowableFuncs)
        )
