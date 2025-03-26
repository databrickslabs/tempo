import copy
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq  # type: ignore

import pyspark.sql.functions as sfn
from pyspark.sql import Column

from tempo.tsdf import TSDF
from tempo.resample import resample, checkAllowableFreq, freq_dict


def vwap(
    tsdf: TSDF,
    frequency: str = "m",
    volume_col: str = "volume",
    price_col: str = "price",
) -> TSDF:
    # set pre_vwap as self or enrich with the frequency
    pre_vwap = tsdf.df
    if frequency == "m":
        pre_vwap = tsdf.df.withColumn(
            "time_group",
            sfn.concat(
                sfn.lpad(sfn.hour(sfn.col(tsdf.ts_col)), 2, "0"),
                sfn.lit(":"),
                sfn.lpad(sfn.minute(sfn.col(tsdf.ts_col)), 2, "0"),
            ),
        )
    elif frequency == "H":
        pre_vwap = tsdf.df.withColumn(
            "time_group",
            sfn.concat(sfn.lpad(sfn.hour(sfn.col(tsdf.ts_col)), 2, "0")),
        )
    elif frequency == "D":
        pre_vwap = tsdf.df.withColumn(
            "time_group",
            sfn.concat(sfn.lpad(sfn.dayofyear(sfn.col(tsdf.ts_col)), 2, "0")),
        )

    group_cols = ["time_group"]
    if tsdf.series_ids:
        group_cols.extend(tsdf.series_ids)
    vwapped = (
        pre_vwap.withColumn("dllr_value", sfn.col(price_col) * sfn.col(volume_col))
        .groupby(group_cols)
        .agg(
            sfn.sum("dllr_value").alias("dllr_value"),
            sfn.sum(volume_col).alias(volume_col),
            sfn.max(price_col).alias("_".join(["max", price_col])),
        )
        .withColumn("vwap", sfn.col("dllr_value") / sfn.col(volume_col))
    )

    return TSDF(vwapped, ts_schema=copy.deepcopy(tsdf.ts_schema))


def EMA(tsdf: TSDF, colName: str, window: int = 30, exp_factor: float = 0.2) -> TSDF:
    """
    Constructs an approximate EMA in the fashion of:
    EMA = e * lag(col,0) + e * (1 - e) * lag(col, 1) + e * (1 - e)^2 * lag(col, 2) etc, up until window
    TODO: replace case when statement with coalesce
    TODO: add in time partitions functionality (what is the overlap fraction?)
    """

    emaColName = "_".join(["EMA", colName])
    df = tsdf.df.withColumn(emaColName, sfn.lit(0)).orderBy(tsdf.ts_col)
    w = tsdf.baseWindow()
    # Generate all the lag columns:
    for i in range(window):
        lagColName = "_".join(["lag", colName, str(i)])
        weight = exp_factor * (1 - exp_factor) ** i
        df = df.withColumn(lagColName, weight * sfn.lag(sfn.col(colName), i).over(w))
        df = df.withColumn(
            emaColName,
            sfn.col(emaColName)
            + sfn.when(sfn.col(lagColName).isNull(), sfn.lit(0)).otherwise(
                sfn.col(lagColName)
            ),
        ).drop(lagColName)
    # Nulls are currently removed

    return TSDF(df, ts_schema=copy.deepcopy(tsdf.ts_schema))


def withLookbackFeatures(
    tsdf: TSDF,
    feature_cols: List[str],
    lookback_window_size: int,
    exact_size: bool = True,
    feature_col_name: str = "features",
) -> TSDF:
    """
    Creates a 2-D feature tensor suitable for training an ML model to predict current values from the history of
    some set of features. This function creates a new column containing, for each observation, a 2-D array of the values
    of some number of other columns over a trailing "lookback" window from the previous observation up to some maximum
    number of past observations.

    :param tsdf: the TSDF to be enriched with the lookback feature column
    :param feature_cols: the names of one or more feature columns to be aggregated into the feature column
    :param lookback_window_size: The size of lookback window (in terms of past observations). Must be an integer >= 1
    :param exact_size: If True (the default), then the resulting DataFrame will only include observations where the
        generated feature column contains arrays of length lookbackWindowSize. This implies that it will truncate
        observations that occurred less than lookbackWindowSize from the start of the timeseries. If False, no truncation
        occurs, and the column may contain arrays less than lookbackWindowSize in length.
    :param feature_col_name: The name of the feature column to be generated. Defaults to "features"

    :return: a DataFrame with a feature column named featureColName containing the lookback feature tensor
    """
    # first, join all featureCols into a single array column
    temp_array_col_name = "__TempArrayCol"
    feat_array_tsdf = tsdf.withColumn(temp_array_col_name, sfn.array(*feature_cols))

    # construct a lookback array
    lookback_win = tsdf.rowsBetweenWindow(-lookback_window_size, -1)
    lookback_tsdf = feat_array_tsdf.withColumn(
        feature_col_name,
        sfn.collect_list(sfn.col(temp_array_col_name)).over(lookback_win),
    ).drop(temp_array_col_name)

    # make sure only windows of exact size are allowed
    if exact_size:
        return lookback_tsdf.where(sfn.size(feature_col_name) == lookback_window_size)

    return lookback_tsdf


def withRangeStats(
    tsdf: TSDF,
    type: str = "range",
    cols_to_summarize: Optional[List[Column]] = None,
    range_back_window_secs: int = 1000,
) -> TSDF:
    """
    Create a wider set of stats based on all numeric columns by default
    Users can choose which columns they want to summarize also. These stats are:
    mean/count/min/max/sum/std deviation/zscore

    :param tsdf: the input dataframe
    :param type: this is created in case we want to extend these stats to lookback over a fixed number of rows instead of ranging over column values
    :param cols_to_summarize: list of user-supplied columns to compute stats for. All numeric columns are used if no list is provided
    :param range_back_window_secs: lookback this many seconds in time to summarize all stats. Note this will look back from the floor of the base event timestamp (as opposed to the exact time since we cast to long)

    Assumptions:

    1. The features are summarized over a rolling window that ranges back
    2. The range back window can be specified by the user
    3. Sequence numbers are not yet supported for the sort
    4. There is a cast to long from timestamp so microseconds or more likely breaks down - this could be more easily handled with a string timestamp or sorting the timestamp itself. If using a 'rows preceding' window, this wouldn't be a problem
    """

    # by default summarize all metric columns
    if not cols_to_summarize:
        cols_to_summarize = tsdf.metric_cols

    # build window
    w = tsdf.rangeBetweenWindow(-1 * range_back_window_secs, 0)

    # compute column summaries
    selected_cols: List[Column] = [sfn.col(c) for c in tsdf.columns]
    derived_cols = []
    for metric in cols_to_summarize:
        selected_cols.append(sfn.mean(metric).over(w).alias("mean_" + metric))
        selected_cols.append(sfn.count(metric).over(w).alias("count_" + metric))
        selected_cols.append(sfn.min(metric).over(w).alias("min_" + metric))
        selected_cols.append(sfn.max(metric).over(w).alias("max_" + metric))
        selected_cols.append(sfn.sum(metric).over(w).alias("sum_" + metric))
        selected_cols.append(sfn.stddev(metric).over(w).alias("stddev_" + metric))
        derived_cols.append(
            (
                (sfn.col(metric) - sfn.col("mean_" + metric))
                / sfn.col("stddev_" + metric)
            ).alias("zscore_" + metric)
        )
    selected_df = tsdf.df.select(*selected_cols)
    summary_df = selected_df.select(*selected_df.columns, *derived_cols).drop(
        "double_ts"
    )

    return TSDF(summary_df, ts_schema=copy.deepcopy(tsdf.ts_schema))


def withGroupedStats(
    tsdf: TSDF,
    freq: str,
    metric_cols: Optional[List[str]] = None,
) -> TSDF:
    """
    Create a wider set of stats based on all numeric columns by default
    Users can choose which columns they want to summarize also. These stats are:
    mean/count/min/max/sum/std deviation

    :param tsdf: the input dataframe
    :param metric_cols - list of user-supplied columns to compute stats for. All numeric columns are used if no list is provided
    :param freq - frequency (provide a string of the form '1 min', '30 seconds' and we interpret the window to use to aggregate
    """

    # identify columns to summarize if not provided
    # these should include all numeric columns that
    # are not the timestamp column and not any of the partition columns
    if not metric_cols:
        # columns we should never summarize
        prohibited_cols = [tsdf.ts_col.lower()]
        if tsdf.series_ids:
            prohibited_cols.extend([pc.lower() for pc in tsdf.series_ids])
        # types that can be summarized
        summarizable_types = ["int", "bigint", "float", "double"]
        # filter columns to find summarizable columns
        metric_cols = [
            datatype[0]
            for datatype in tsdf.df.dtypes
            if (
                (datatype[1] in summarizable_types)
                and (datatype[0].lower() not in prohibited_cols)
            )
        ]

    # build window
    parsed_freq = checkAllowableFreq(freq)
    period, unit = parsed_freq[0], parsed_freq[1]
    agg_window = sfn.window(
        sfn.col(tsdf.ts_col),
        "{} {}".format(
            period, freq_dict[unit]  # type: ignore[literal-required]
        ),
    )

    # compute column summaries
    selected_cols = []
    for metric in metric_cols:
        selected_cols.extend(
            [
                sfn.mean(sfn.col(metric)).alias("mean_" + metric),
                sfn.count(sfn.col(metric)).alias("count_" + metric),
                sfn.min(sfn.col(metric)).alias("min_" + metric),
                sfn.max(sfn.col(metric)).alias("max_" + metric),
                sfn.sum(sfn.col(metric)).alias("sum_" + metric),
                sfn.stddev(sfn.col(metric)).alias("stddev_" + metric),
            ]
        )

    grouping_cols = [sfn.col(c) for c in tsdf.series_ids] + [agg_window]
    selected_df = tsdf.df.groupBy(grouping_cols).agg(*selected_cols)
    summary_df = (
        selected_df.select(*selected_df.columns)
        .withColumn(tsdf.ts_col, sfn.col("window").start)
        .drop("window")
    )

    return TSDF(summary_df, ts_schema=copy.deepcopy(tsdf.ts_schema))

def calc_bars(
        tsdf: TSDF,
        freq: str,
        metric_cols: Optional[List[str]] = None,
        fill: Optional[bool] = None,
) -> TSDF:
    """
    Calculate OHLC (Open, High, Low, Close) bars from time series data.

    :param tsdf: the input time series dataframe
    :param freq: frequency to resample the data
    :param metric_cols: list of metric columns to calculate bars for
    :param fill: whether to fill missing values
    :return: a TSDF with OHLC bars
    """
    # Import resample functionality from tempo.resample
    import tempo.resample as t_resample

    # Create resample instances using the t_resample module directly
    resample_open = resample(
        tsdf, freq=freq, func="floor", metricCols=metric_cols, prefix="open", fill=fill
    )

    resample_low = resample(
        tsdf, freq=freq, func="min", metricCols=metric_cols, prefix="low", fill=fill
    )

    resample_high = resample(
        tsdf, freq=freq, func="max", metricCols=metric_cols, prefix="high", fill=fill
    )

    resample_close = resample(
        tsdf, freq=freq, func="ceil", metricCols=metric_cols, prefix="close", fill=fill
    )

    join_cols = resample_open.series_ids + [resample_open.ts_col]
    bars = (
        resample_open.df.join(resample_high.df, join_cols)
        .join(resample_low.df, join_cols)
        .join(resample_close.df, join_cols)
    )
    non_part_cols = (
            set(bars.columns) - set(resample_open.series_ids) - {resample_open.ts_col}
    )
    sel_and_sort = (
            resample_open.series_ids + [resample_open.ts_col] + sorted(non_part_cols)
    )
    bars = bars.select(sel_and_sort)

    return TSDF(bars, ts_col=resample_open.ts_col, series_ids=resample_open.series_ids)


def fourier_transform(
    tsdf: TSDF, timestep: Union[int, float, complex], value_col: str
) -> TSDF:
    """
    Function to fourier transform the time series to its frequency domain representation.

    :param tsdf: input time series dataframe
    :param timestep: timestep value to be used for getting the frequency scale
    :param value_col: name of the time domain data column which will be transformed

    :return: TSDF with the fourier transform columns added
    """

    def tempo_fourier_util(
        pdf: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        This method is a vanilla python logic implementing fourier transform on a numpy array using the scipy module.
        This method is meant to be called from Tempo TSDF as a pandas function API on Spark
        """
        select_cols = list(pdf.columns)
        pdf.sort_values(by=["tpoints"], inplace=True, ascending=True)
        y = np.array(pdf["tdval"])
        tran = fft(y)
        r = tran.real
        i = tran.imag
        pdf["ft_real"] = r
        pdf["ft_imag"] = i
        N = tran.shape
        xf = fftfreq(N[0], timestep)
        pdf["freq"] = xf
        return pdf[select_cols + ["freq", "ft_real", "ft_imag"]]

    data = tsdf.df

    if not tsdf.series_ids:
        data = data.withColumn("dummy_group", sfn.lit("dummy_val"))
        data = (
            data.select(sfn.col("dummy_group"), tsdf.ts_col, sfn.col(value_col))
            .withColumn("tdval", sfn.col(value_col))
            .withColumn("tpoints", sfn.col(tsdf.ts_col))
        )
        return_schema = ",".join(
            [f"{i[0]} {i[1]}" for i in data.dtypes]
            + ["freq double", "ft_real double", "ft_imag double"]
        )
        result = data.groupBy("dummy_group").applyInPandas(
            tempo_fourier_util, return_schema
        )
        result = result.drop("dummy_group", "tdval", "tpoints")
    else:
        group_cols = tsdf.series_ids
        data = (
            data.select(*group_cols, tsdf.ts_col, sfn.col(value_col))
            .withColumn("tdval", sfn.col(value_col))
            .withColumn("tpoints", sfn.col(tsdf.ts_col))
        )
        return_schema = ",".join(
            [f"{i[0]} {i[1]}" for i in data.dtypes]
            + ["freq double", "ft_real double", "ft_imag double"]
        )
        result = data.groupBy(*group_cols).applyInPandas(
            tempo_fourier_util, return_schema
        )
        result = result.drop("tdval", "tpoints")

    return TSDF(result, ts_schema=copy.deepcopy(tsdf.ts_schema))
