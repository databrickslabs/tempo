from utils import __notdltruntime

from tempo.tsdf import TSDF
if __notdltruntime():
    from tempo.utils import display

    '''
    Conditonal import of display so that it doesn't gets imported in runtimes where display is not required.
    For example in DATABRICKS Delta Live Tables Runtimes.
    '''
