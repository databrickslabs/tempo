from __future__ import annotations

import logging
import math
import os
from datetime import datetime as dt
from datetime import timedelta as td
from typing import Optional, Union, overload

import pyspark.sql.functions as sfn
from IPython import get_ipython  # type: ignore[import-not-found]
from IPython.core.display import HTML  # type: ignore[import-not-found]
from IPython.display import display as ipydisplay  # type: ignore[import-not-found]
from pandas.core.frame import DataFrame as pandasDataFrame
from pyspark.sql import DataFrame, SparkSession

import tempo.tsdf as t_tsdf

logger = logging.getLogger(__name__)
IS_DATABRICKS = "DB_HOME" in os.environ.keys()

"""
DB_HOME env variable has been chosen and that's because this variable is a special variable that will be available in DBR.

This constant is to ensure the correct behaviour of the show and display methods are called based on the platform
where the code is running from.
"""


def time_range(
    spark: SparkSession,
    start_time: dt,
    end_time: Optional[dt] = None,
    step_size: Optional[td] = None,
    num_intervals: Optional[int] = None,
    ts_colname: str = "ts",
    include_interval_ends: bool = False,
) -> DataFrame:
    """
    Generate a DataFrame of a range of timestamps with a regular interval,
    similar to pandas.date_range, but for Spark DataFrames.
    The DataFrame will have a single column named `ts_colname` (default is "ts")
    that contains timestamps starting at `start_time` and ending at `end_time`
    (if provided), with a step size of `step_size` (if provided) or
    `num_intervals` (if provided). At least 2 of the 3 arguments `end_time`,
    `step_size`, and `num_intervals` must be provided. The third
    argument can be computed based on the other two, if needed. Optionally, the end of
    each time interval can be included as a separate column in the DataFrame.

    :param spark: SparkSession object
    :param start_time: start time of the range
    :param end_time: end time of the range (optional)
    :param step_size: time step size (optional)
    :param num_intervals: number of intervals (optional)
    :param ts_colname: name of the timestamp column, default is "ts"
    :param include_interval_ends: whether to include the end of each time
    interval as a separate column in the DataFrame

    :return: DataFrame with a time range of timestamps
    """

    # compute step_size if not provided
    if not step_size:
        # must have both end_time and num_intervals defined
        assert (
            end_time and num_intervals
        ), "must provide at least 2 of: end_time, step_size, num_intervals"
        diff_time = end_time - start_time
        step_size = diff_time / num_intervals

    # compute the number of intervals if not provided
    if not num_intervals:
        # must have both end_time and num_intervals defined
        assert (
            end_time and step_size
        ), "must provide at least 2 of: end_time, step_size, num_intervals"
        diff_time = end_time - start_time
        num_intervals = math.ceil(diff_time / step_size)

    # define expressions for the time range
    start_time_expr = sfn.to_timestamp(sfn.lit(str(start_time)))
    step_fractional_seconds = step_size.seconds + (step_size.microseconds / 1e6)
    interval_expr = sfn.make_dt_interval(
        days=sfn.lit(step_size.days), secs=sfn.lit(step_fractional_seconds)
    )

    # create the DataFrame
    range_df = spark.range(0, num_intervals).withColumn(
        ts_colname, start_time_expr + sfn.col("id") * interval_expr
    )
    if include_interval_ends:
        interval_end_colname = ts_colname + "_interval_end"
        range_df = range_df.withColumn(
            interval_end_colname,
            start_time_expr + (sfn.col("id") + sfn.lit(1)) * interval_expr,
        )
    return range_df.drop("id")


class ResampleWarning(Warning):
    """
    This class is a warning that is raised when the interpolate or resample with fill methods are called.
    """


def _is_capable_of_html_rendering() -> bool:
    """
    This method returns a boolean value signifying whether the environment is a notebook environment
    capable of rendering HTML or not.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


@overload
def display_html(df: pandasDataFrame) -> None: ...


@overload
def display_html(df: DataFrame) -> None: ...


def display_html(df: Union[pandasDataFrame, DataFrame]) -> None:
    """
    Display method capable of displaying the dataframe in a formatted HTML structured output
    """
    ipydisplay(HTML("<style>pre { white-space: pre !important; }</style>"))
    if isinstance(df, DataFrame):
        df.show(truncate=False, vertical=False)
    elif isinstance(df, pandasDataFrame):
        print(df.head())
    else:
        logger.error("'display' method not available for this object")


def display_unavailable() -> None:
    """
    This method is called when display method is not available in the environment.
    """
    logger.error(
        "'display' method not available in this environment. Use 'show' method instead."
    )


def get_display_df(tsdf, k):
    return tsdf.latest(k).withNaturalOrdering().df


@overload
def display_improvised(obj: t_tsdf.TSDF) -> None: ...


@overload
def display_improvised(obj: pandasDataFrame) -> None: ...


@overload
def display_improvised(obj: DataFrame) -> None: ...


def display_improvised(obj: Union[t_tsdf.TSDF, pandasDataFrame, DataFrame]) -> None:
    if isinstance(obj, t_tsdf.TSDF):
        method(get_display_df(obj, k=5))
    else:
        method(obj)


@overload
def display_html_improvised(obj: Optional[t_tsdf.TSDF]) -> None: ...


@overload
def display_html_improvised(obj: Optional[pandasDataFrame]) -> None: ...


@overload
def display_html_improvised(obj: Optional[DataFrame]) -> None: ...


def display_html_improvised(
    obj: Union[t_tsdf.TSDF, pandasDataFrame, DataFrame]
) -> None:
    if isinstance(obj, t_tsdf.TSDF):
        display_html(get_display_df(obj, k=5))
    else:
        display_html(obj)


ENV_CAN_RENDER_HTML = _is_capable_of_html_rendering()

if (
    IS_DATABRICKS
        and get_ipython() is not None
    and ("display" in get_ipython().user_ns.keys())
):
    method = get_ipython().user_ns["display"]

    # Under 'display' key in user_ns the original databricks display method is present
    # to know more refer: /databricks/python_shell/scripts/db_ipykernel_launcher.py

    display = display_improvised

elif ENV_CAN_RENDER_HTML:

    display = display_html_improvised

else:
    display = display_unavailable  # type: ignore

"""
display method's equivalent for TSDF object

Example to show usage
---------------------
from pyspark.sql.functions import *

phone_accel_df = spark.read.format("csv").option("header", "true").load("dbfs:/home/tempo/Phones_accelerometer").withColumn("event_ts", (col("Arrival_Time").cast("double")/1000).cast("timestamp")).withColumn("x", col("x").cast("double")).withColumn("y", col("y").cast("double")).withColumn("z", col("z").cast("double")).withColumn("event_ts_dbl", col("event_ts").cast("double"))

from tempo import *

phone_accel_tsdf = TSDF(phone_accel_df, ts_col="event_ts", partition_cols = ["User"])

# Calling display method here
display(phone_accel_tsdf)
"""
