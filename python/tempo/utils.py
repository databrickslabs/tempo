import logging
import os

from IPython import get_ipython
from IPython.core.display import HTML
from IPython.display import display as ipydisplay
from pandas import DataFrame as pandasDataFrame
from pyspark.sql.dataframe import DataFrame

logger = logging.getLogger(__name__)
PLATFORM = "DATABRICKS" if "DB_HOME" in os.environ.keys() else "NON_DATABRICKS"
"""
DB_HOME env variable has been chosen and that's because this variable is a special variable that will be available in DBR.

This constant is to ensure the correct behaviour of the show and display methods are called based on the platform 
where the code is running from. 
"""


def __not_dlt_runtime():
    if type(get_ipython()) != type(None):
        user_namespace_keys = list(get_ipython().user_ns.keys())
        check_bool = (('create_dlt_table_fn' not in user_namespace_keys) and ('dlt_sql_fn' not in user_namespace_keys))
        return check_bool
    else:
        return False


def __isnotebookenv():
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
    logger.error("'display' method not available in this environment. Use 'show' method instead.")


ENV_BOOLEAN = __isnotebookenv()

if PLATFORM == "DATABRICKS":

    # This below check is for ensuring compatibility with Databricks DLT runtimes
    # This if logic ensures that the custom user's namespace of DLT runtimes
    # which doesn't have PythonShell's Display object in the namespace doesn't result in an error.
    if __not_dlt_runtime():
        method = get_ipython().user_ns['display']


        # Under 'display' key in user_ns the original databricks display method is present
        # to know more refer: /databricks/python_shell/scripts/db_ipykernel_launcher.py
        def display_improvised(obj):
            if type(obj).__name__ == 'TSDF':
                method(obj.df)
            else:
                method(obj)


        display = display_improvised
    else:
        display = display_unavailable
elif __isnotebookenv():
    def display_html_improvised(obj):
        if type(obj).__name__ == 'TSDF':
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
