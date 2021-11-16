from IPython.display import display as ipydisplay
from IPython.core.display import HTML
from IPython import get_ipython
import os
import logging
from pyspark.sql.dataframe import DataFrame
from pandas import DataFrame as pandasDataFrame
import numpy as np
from scipy.fft import fft, fftfreq

logger = logging.getLogger(__name__)
PLATFORM = "DATABRICKS" if "DATABRICKS_RUNTIME_VERSION" in os.environ.keys() else "NON_DATABRICKS"
"""
This constant is to ensure the correct behaviour of the show and display methods are called based on the platform 
where the code is running from. 
"""


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
    method = get_ipython().user_ns['display']


    # Under 'display' key in user_ns the original databricks display method is present
    # to know more refer: /databricks/python_shell/scripts/db_ipykernel_launcher.py
    def display_improvised(obj):
        if type(obj).__name__ == 'TSDF':
            method(obj.df)
        else:
            method(obj)


    display = display_improvised
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

TIMESTEP = 1
"""
This constant is for initializing the TIMESTEP value of a time series as 1 sec, by default   
"""


def set_timestep(n=1):
    """
    This method is called to set the TIMESTEP value for a Time series.
    """
    global TIMESTEP
    TIMESTEP = n


def get_timestep():
    """
    This method is called to get TIMESTEP value for a Time series inside the tempo_fourier_util method
    """
    global TIMESTEP
    return TIMESTEP


def tempo_fourier_util(pdf):
    """
    This method is a vanilla python logic implementing fourier transform on a numpy array using the scipy module.
    This method is meant to be called from Tempo TSDF as a pandas function API on Spark
    """
    select_cols = list(pdf.columns)
    y = np.array(pdf['val'])
    tran = fft(y)
    r = tran.real
    i = tran.imag
    pdf['ft_real'] = r
    pdf['ft_imag'] = i
    N = tran.shape
    timestep = get_timestep()
    xf = fftfreq(N[0], timestep)
    pdf['freq'] = xf
    return pdf[select_cols + ['freq', 'ft_real', 'ft_imag']]
