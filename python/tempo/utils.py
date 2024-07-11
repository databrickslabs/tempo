from __future__ import annotations

import logging
import os
import warnings
from typing import List, Optional, Union, overload

import pyspark.sql.functions as sfn
import tempo.resample as t_resample
import tempo.tsdf as t_tsdf
from IPython import get_ipython  # type: ignore
from IPython.core.display import HTML  # type: ignore
from IPython.display import display as ipydisplay  # type: ignore
from pandas.core.frame import DataFrame as pandasDataFrame
from pyspark.sql.dataframe import DataFrame

logger = logging.getLogger(__name__)
IS_DATABRICKS = "DB_HOME" in os.environ.keys()

"""
DB_HOME env variable has been chosen and that's because this variable is a special variable that will be available in DBR.

This constant is to ensure the correct behaviour of the show and display methods are called based on the platform
where the code is running from.
"""


class ResampleWarning(Warning):
    """
    This class is a warning that is raised when the interpolate or resample with fill methods are called.
    """

    pass


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


def calculate_time_horizon(
    df: DataFrame,
    ts_col: str,
    freq: str,
    partition_cols: Optional[List[str]],
    local_freq_dict: Optional[t_resample.FreqDict] = None,
) -> None:
    # Convert Frequency using resample dictionary
    if local_freq_dict is None:
        local_freq_dict = t_resample.freq_dict
    parsed_freq = t_resample.checkAllowableFreq(freq)
    period, unit = parsed_freq[0], parsed_freq[1]
    if t_resample.is_valid_allowed_freq_keys(
        unit,
        t_resample.ALLOWED_FREQ_KEYS,
    ):
        freq = f"{period} {local_freq_dict[unit]}"  # type: ignore[literal-required]
    else:
        raise ValueError(f"Frequency {unit} not supported")

    # Get max and min timestamp per partition
    partitioned_df: DataFrame = df.groupBy(*partition_cols).agg(
        sfn.max(ts_col).alias("max_ts"),
        sfn.min(ts_col).alias("min_ts"),
    )

    # Generate upscale metrics
    normalized_time_df: DataFrame = (
        partitioned_df.withColumn("min_epoch_ms", sfn.expr("unix_millis(min_ts)"))
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


def get_display_df(tsdf: t_tsdf.TSDF, k: int) -> DataFrame:
    # let's show the n most recent records per series, in order:
    orderCols = tsdf.partitionCols.copy()
    orderCols.append(tsdf.ts_col)
    if tsdf.sequence_col:
        orderCols.append(tsdf.sequence_col)
    return tsdf.latest(k).df.orderBy(orderCols)


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
    and not (get_ipython() is None)
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
