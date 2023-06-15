from __future__ import annotations

import logging
import os
from typing import Optional, Union, overload

from IPython import get_ipython
from IPython.core.display import HTML
from IPython.display import display as ipydisplay
from pandas.core.frame import DataFrame as pandasDataFrame
from pyspark.sql.dataframe import DataFrame

import tempo.tsdf as t_tsdf

logger = logging.getLogger(__name__)
IS_DATABRICKS = "DB_HOME" in os.environ.keys()

"""
DB_HOME env variable has been chosen and that's because this variable is a special variable that will be available in DBR.

This constant is to ensure the correct behaviour of the show and display methods are called based on the platform
where the code is running from.
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
def display_html(df: pandasDataFrame) -> None:
    ...


@overload
def display_html(df: DataFrame) -> None:
    ...


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


ENV_CAN_RENDER_HTML = _is_capable_of_html_rendering()

if (
    IS_DATABRICKS
    and not (get_ipython() is None)
    and ("display" in get_ipython().user_ns.keys())
):
    method = get_ipython().user_ns["display"]

    # Under 'display' key in user_ns the original databricks display method is present
    # to know more refer: /databricks/python_shell/scripts/db_ipykernel_launcher.py

    @overload
    def display_improvised(obj: t_tsdf.TSDF) -> None:
        ...

    @overload
    def display_improvised(obj: pandasDataFrame) -> None:
        ...

    @overload
    def display_improvised(obj: DataFrame) -> None:
        ...

    def display_improvised(obj: Union[t_tsdf.TSDF, pandasDataFrame, DataFrame]) -> None:
        if isinstance(obj, t_tsdf.TSDF):
            method(get_display_df(obj, k=5))
        else:
            method(obj)

    display = display_improvised

elif ENV_CAN_RENDER_HTML:

    @overload
    def display_html_improvised(obj: Optional[t_tsdf.TSDF]) -> None:
        ...

    @overload
    def display_html_improvised(obj: Optional[pandasDataFrame]) -> None:
        ...

    @overload
    def display_html_improvised(obj: Optional[DataFrame]) -> None:
        ...

    def display_html_improvised(
        obj: Union[t_tsdf.TSDF, pandasDataFrame, DataFrame]
    ) -> None:
        if isinstance(obj, t_tsdf.TSDF):
            display_html(get_display_df(obj, k=5))
        else:
            display_html(obj)

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
