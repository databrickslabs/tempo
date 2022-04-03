from IPython import get_ipython

from tempo.tsdf import TSDF
if (
            ('create_dlt_table_fn' not in list(get_ipython().user_ns.keys()))
            and
            ('dlt_sql_fn' not in list(get_ipython().user_ns.keys()))
    ):
    from tempo.utils import display

    '''
    Conditonal import of display so that it doesn't gets imported in runtimes where display is not required.
    For example in DATABRICKS Delta Live Tables Runtimes.
    '''
