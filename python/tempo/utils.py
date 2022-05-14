from typing import List
import logging
import os
import warnings
from IPython import get_ipython
from IPython.core.display import HTML
from IPython.display import display as ipydisplay
from pandas import DataFrame as pandasDataFrame
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import expr, max, min, sum
from tempo.resample import checkAllowableFreq, freq_dict

logger = logging.getLogger(__name__)
PLATFORM = "DATABRICKS" if "DB_HOME" in os.environ.keys() else "NON_DATABRICKS"

"""
DB_HOME env variable has been chosen and that's because this variable is a special variable that will be available in DBR.

This constant is to ensure the correct behaviour of the show and display methods are called based on the platform 
where the code is running from. 
"""


class UpsampleWarning(Warning):
    """
    This class is a warning that is raised when the interpolate or resample with fill methods are called.
    """
    pass


def __is_capable_of_html_rendering():
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
    df: DataFrame, ts_col: str, freq: str, partition_cols: List[str]
):
    # Convert Frequency using resample dictionary
    parsed_freq = checkAllowableFreq(freq)
    freq = f"{parsed_freq[0]} {freq_dict[parsed_freq[1]]}"

    # Get max and min timestamp per partition
    partitioned_df: DataFrame = df.groupBy(*partition_cols).agg(
        max(ts_col).alias("max_ts"),
        min(ts_col).alias("min_ts"),
    )

    # Generate upscale metrics
    normalized_time_df: DataFrame = (
        partitioned_df.withColumn("min_epoch_ms", expr("unix_millis(min_ts)"))
        .withColumn("max_epoch_ms", expr("unix_millis(max_ts)"))
        .withColumn(
            "interval_ms",
            expr(
                f"unix_millis(cast('1970-01-01 00:00:00.000+0000' as TIMESTAMP) + INTERVAL {freq})"
            ),
        )
        .withColumn(
            "rounded_min_epoch", expr("min_epoch_ms - (min_epoch_ms % interval_ms)")
        )
        .withColumn(
            "rounded_max_epoch", expr("max_epoch_ms - (max_epoch_ms % interval_ms)")
        )
        .withColumn("diff_ms", expr("rounded_max_epoch - rounded_min_epoch"))
        .withColumn("num_values", expr("(diff_ms/interval_ms) +1"))
    )

    min_ts, max_ts, max_value_partition, total_values = normalized_time_df.select(
        min("min_ts"), max("max_ts"), max("num_values"), sum("num_values")
    ).first()

    warnings.simplefilter("always", UpsampleWarning)
    warnings.warn(
        f"""
            Upsample Metrics Warning: 
                Earliest Timestamp: {min_ts}
                Latest Timestamp: {max_ts}
                No. of Unique Partitions: {normalized_time_df.count()}
                Max No. Values in Single a Partition: {max_value_partition}
                Total No. Values Across All Partitions: {total_values}
        """,
        UpsampleWarning,
    )


def display_html(df):
    """
    Display method capable of displaying the dataframe in a formatted HTML structured output
    """
    ipydisplay(HTML("<style>pre { white-space: pre !important; }</style>"))
    if isinstance(df, DataFrame):
        df.show(truncate=False, vertical=False)
    elif isinstance(df, pandasDataFrame):
        df.head()
    else:
        logger.error("'display' method not available for this object")


def display_unavailable(df):
    """
    This method is called when display method is not available in the environment.
    """
    logger.error(
        "'display' method not available in this environment. Use 'show' method instead."
    )


ENV_BOOLEAN = __is_capable_of_html_rendering()


if (
    (PLATFORM == "DATABRICKS")
    and (type(get_ipython()) != type(None))
    and ("display" in get_ipython().user_ns.keys())
):
    method = get_ipython().user_ns["display"]
    # Under 'display' key in user_ns the original databricks display method is present
    # to know more refer: /databricks/python_shell/scripts/db_ipykernel_launcher.py
    def display_improvised(obj):
        if type(obj).__name__ == "TSDF":
            method(obj.df)
        else:
            method(obj)

    display = display_improvised

elif ENV_BOOLEAN:

    def display_html_improvised(obj):
        if type(obj).__name__ == "TSDF":
            display_html(obj.df)
        else:
            display_html(obj)

    display = display_html_improvised

else:
    display = display_unavailable

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
