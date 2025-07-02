from datetime import datetime
from typing import TypeVar, Union

from pandas import Timestamp

IntervalBoundary = Union[str, int, float, Timestamp, datetime, None]
MetricValue = Union[int, float, bool]
T = TypeVar("T")
