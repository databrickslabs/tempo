from tempo.utils import __not_dlt_runtime

from tempo.tsdf import TSDF
if __not_dlt_runtime():
    from tempo.utils import display

    '''
    Conditonal import of display so that it doesn't gets imported in runtimes where display is not required.
    For example in DATABRICKS Delta Live Tables Runtimes.
    '''
