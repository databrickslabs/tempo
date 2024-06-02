from typing import Union, Callable
import copy

import pandas as pd

import pyspark.sql.functions as sfn
from pyspark.sql import Window

from tempo.tsdf import TSDF

# Some common interpolation functions


def zero_fill(null_series: pd.Series) -> pd.Series:
    return null_series.fillna(0)


def forward_fill(null_series: pd.Series) -> pd.Series:
    return null_series.ffill()


def backward_fill(null_series: pd.Series) -> pd.Series:
    return null_series.bfill()


# The interpolation

def _build_interpolator(
        interpol_col: str,
        interpol_fn: Callable[[pd.Series], pd.Series]
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def interpolator_fn(pdf: pd.DataFrame) -> pd.DataFrame:
        # first we identify non-gap sections and return them as-is
        if pdf["needs_interpol"].sum() < 1:
            return pdf
        # otherwise we interpolate the missing values
        pdf[interpol_col] = interpol_fn(pdf[interpol_col])
        # return only the rows that were missing (others are margins)
        return pdf[pdf["needs_interpol"]]

    return interpolator_fn


def interpolate(
    tsdf: TSDF,
    col: str,
    fn: Union[Callable[[pd.Series], pd.Series], str],
    leading_margin: int = 1,
    lagging_margin: int = 0
) -> TSDF:
    """
    Interpolate missing values in a time series column.

    For the given column, null values are assumed to be missing, and
    this method will attempt to interpolate them using the given function.
    The interpolation function can be a string representing a valid method
    for the pandas Series.interpolate method, or a custom function that takes
    a pandas Series and returns a pandas Series of the same length.
    The Series given may include a "margin" of
    leading or trailing non-missing (i.e. non-null) values to help the
    interpolation function. The exact size of the leading and trailing margins are
    configurable. The function should not attempt to modify these margin values.

    :param tsdf: the :class:`TSDF` timeseries dataframe
    :param col: the name of the column to interpolate
    :param fn: the interpolation function
    :param leading_margin: the number of non-missing values
    to include before the first missing value
    :param lagging_margin: the number of non-missing values
    to include after the last missing value

    :return: a new :class:`TSDF` with the missing values of the given
    column interpolated
    """

    # flag columns that need interpolation
    needs_intpl_col = "__tmp_needs_interpol"
    segments = tsdf.df.withColumn(needs_intpl_col, sfn.col(col).isNull())

    # identify transitions between segments
    seg_trans_col = "__tmp_seg_transition"
    all_win = Window.partitionBy("symbol").orderBy("timestamp")
    segments = segments.withColumn(seg_trans_col,
                                   sfn.lag(needs_intpl_col, 1).over(all_win)
                                   != sfn.col(needs_intpl_col))

    # assign a group number to each segment
    seg_group_col = "__tmp_seg_group"
    all_prev_win = Window.partitionBy("symbol")\
        .orderBy("timestamp")\
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    segments = segments.withColumn(seg_group_col,
                                   sfn.count_if(seg_trans_col).over(all_prev_win))

    # build margins around intepolation segments
    if leading_margin > 0 or lagging_margin > 0:
        # collect the group number of each segment with a margin
        margin_col = "__tmp_group_with_margin"
        margin_win = Window.partitionBy("symbol")\
            .orderBy("timestamp")\
            .rowsBetween(-leading_margin, lagging_margin)
        segments = segments.withColumn(margin_col,
                                       sfn.when(~sfn.col(needs_intpl_col),
                                                sfn.collect_set(seg_group_col)
                                                .over(margin_win))
                                       .otherwise(sfn.array(seg_group_col)))
        # explode the groups with margins
        explode_exprs = (tsdf.df.columns
                         + [sfn.explode(margin_col).alias(seg_group_col),
                            needs_intpl_col])
        segments = segments.select(*explode_exprs)

    # build the interpolator function
    if isinstance(fn, str):
        interpolator = _build_interpolator(col, lambda x: x.interpolate(method=fn))
    else:
        interpolator = _build_interpolator(col, fn)

    # apply the interpolator to each segment
    interpolated_df = segments.groupBy(tsdf.series_ids + [seg_group_col]) \
        .applyInPandas(interpolator, segments.schema) \
        .drop(seg_group_col, needs_intpl_col)

    # return it as a new TSDF
    return TSDF(interpolated_df, ts_schema=copy.deepcopy(tsdf.ts_schema))
