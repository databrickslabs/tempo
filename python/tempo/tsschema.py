from abc import ABC, abstractmethod
from typing import Union, Collection, List

from pyspark.sql import Column
import pyspark.sql.functions as Fn
from pyspark.sql.types import *


class TSIndex(ABC):
    """
    Abstract base class for all Timeseries Index types
    """

    # Valid types for time index columns
    __valid_ts_types = (
        DateType(),
        TimestampType(),
        ByteType(),
        ShortType(),
        IntegerType(),
        LongType(),
        DecimalType(),
        FloatType(),
        DoubleType(),
    )

    @classmethod
    def isValidTSType(cls, dataType: DataType) -> bool:
        return dataType in cls.__valid_ts_types

    def __init__(self, name: str, dataType: DataType) -> None:
        self.name = name
        self.dataType = dataType

    @abstractmethod
    def orderByExpr(self, reverse: bool = False) -> Union[Column, List[Column]]:
        """
        Returns a :class:`Column` expression that will order the :class:`TSDF` according to the timeseries index.

        :param reverse: whether or not the ordering should be reversed (backwards in time)

        :return: an expression appropriate for ordering the :class:`TSDF` according to this index
        """
        pass

class SimpleTSIndex(TSIndex):
    """
    Timeseries index based on a single column of a numeric or temporal type.
    """

    def __init__(self, ts_col: StructField) -> None:
        if not self.isValidTSType(ts_col.dataType):
            raise TypeError(f"DataType {ts_col.dataType} of column {ts_col.name} is not valid for a timeseries Index")
        super().__init__(ts_col.name, ts_col.dataType)

    def orderByExpr(self, reverse: bool = False) -> Column:
        expr = Fn.col(self.name)
        if reverse:
            return expr.desc()
        return expr


class SubSequenceTSIndex(TSIndex):
    """
    Special timeseries index for columns that involve a secondary sequencing column
    """

    # default name for our timeseries index
    __ts_idx_name = "ts_index"
    # Valid types for sub-sequence columns
    __valid_subseq_types = (
        ByteType(),
        ShortType(),
        IntegerType(),
        LongType()
    )

    def __init__(self, primary_ts_col: StructField, subsequence_col: StructField) -> None:
        # validate these column types
        if primary_ts_col.dataType not in self.__valid_ts_types:
            raise TypeError(f"DataType {primary_ts_col.dataType} of column {primary_ts_col.name} is not valid for a timeseries Index")
        if subsequence_col.dataType not in self.__valid_subseq_types:
            raise TypeError(f"DataType {subsequence_col.dataType} of column {subsequence_col.name} is not valid for a sub-sequencing column")
        # construct a struct for these
        ts_struct = StructType([primary_ts_col, subsequence_col])
        super().__init__(self.__ts_idx_name, ts_struct)
        # set colnames for primary & subsequence
        self.primary_ts_col = primary_ts_col.name
        self.subsequence_col = subsequence_col.name

    def orderByExpr(self, reverse: bool = False) -> List[Column]:
        expr = [ Fn.col(self.primary_ts_col), Fn.col(self.subsequence_col) ]
        if reverse:
            return [col.desc() for col in expr]
        return expr


class TSSchema:
    """
    Schema type for a :class:`TSDF` class.
    """

    # Valid types for metric columns
    __metric_types = (
        BooleanType(),
        ByteType(),
        ShortType(),
        IntegerType(),
        LongType(),
        DecimalType(),
        FloatType(),
        DoubleType(),
    )

    def __init__(
        self,
        ts_idx: TSIndex,
        series_ids: Collection[str] = None
    ) -> None:
        self.ts_idx = ts_idx
        if series_ids:
            self.series_ids = list(series_ids)
        else:
            self.series_ids = None

    @classmethod
    def fromDFSchema(
        cls, df_schema: StructType, ts_col: str, series_ids: Collection[str] = None
    ) -> "TSSchema":
        # construct a TSIndex for the given ts_col
        ts_idx = SimpleTSIndex(df_schema[ts_col])
        return cls(ts_idx, series_ids)

    @property
    def structural_columns(self) -> set[str]:
        """
        Structural columns are those that define the structure of the :class:`TSDF`. This includes the timeseries column,
        a timeseries index (if different), any subsequence column (if present), and the series ID columns.

        :return: a set of column names corresponding the structural columns of a :class:`TSDF`
        """
        struct_cols = {self.ts_index}.union(self.series_ids)
        struct_cols.discard(None)
        return struct_cols

    def validate(self, df_schema: StructType) -> None:
        pass

    def find_observational_columns(self, df_schema: StructType) -> set[str]:
        return set(df_schema.fieldNames()) - self.structural_columns

    def find_metric_columns(self, df_schema: StructType) -> list[str]:
        return [
            col.name
            for col in df_schema.fields
            if (col.dataType in self.__metric_types)
                and
               (col.name in self.find_observational_columns(df_schema))
        ]
