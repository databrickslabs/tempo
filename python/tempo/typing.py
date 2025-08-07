from collections.abc import Iterable
from typing import Any, Callable, Union

from pandas.core.frame import DataFrame as PandasDataFrame
from pyspark.sql import Column

# These definitions were copied from private pypark modules:
# - pyspark.sql._typing
# - pyspark.sql.pandas._typing

ColumnOrName = Union[Column, str]

PandasMapIterFunction = Callable[[Iterable[PandasDataFrame]], Iterable[PandasDataFrame]]

PandasGroupedMapFunction = Union[
    Callable[[PandasDataFrame], PandasDataFrame],
    Callable[[Any, PandasDataFrame], PandasDataFrame],
]
