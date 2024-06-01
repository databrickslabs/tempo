from typing import Union, Callable, Iterable, Any

from pyspark.sql import Column

from pandas.core.frame import DataFrame as PandasDataFrame

# These definitions were copied from private pypark modules:
# - pyspark.sql._typing
# - pyspark.sql.pandas._typing

ColumnOrName = Union[Column, str]

PandasMapIterFunction = Callable[[Iterable[PandasDataFrame]], Iterable[PandasDataFrame]]

PandasGroupedMapFunction = Union[
    Callable[[PandasDataFrame], PandasDataFrame],
    Callable[[Any, PandasDataFrame], PandasDataFrame],
]
