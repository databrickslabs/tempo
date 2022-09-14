from abc import ABC, abstractmethod
from typing import Union, Collection, List

from pyspark.sql import Column
import pyspark.sql.functions as Fn
from pyspark.sql.types import *
from pyspark.sql.types import NumericType

#
# Timeseries Index Classes
#

class TSIndex(ABC):
    """
    Abstract base class for all Timeseries Index types
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        :return: the column name of the timeseries index
        """

    @property
    @abstractmethod
    def ts_col(self) -> str:
        """
        :return: the name of the primary timeseries column (may or may not be the same as the name)
        """

    def _reverseOrNot(self, expr: Union[Column, List[Column]], reverse: bool) -> Union[Column, List[Column]]:
        if not reverse:
            return expr # just return the expression as-is if we're not reversing
        elif type(expr) == Column:
            return expr.desc() # reverse a single-expression
        elif type(expr) == List[Column]:
            return [col.desc() for col in expr] # reverse all columns in the expression

    @abstractmethod
    def orderByExpr(self, reverse: bool = False) -> Union[Column, List[Column]]:
        """
        Gets an expression that will order the :class:`TSDF` according to the timeseries index.

        :param reverse: whether the ordering should be reversed (backwards in time)

        :return: an expression appropriate for ordering the :class:`TSDF` according to this index
        """

    def rangeOrderByExpr(self, reverse: bool = False) -> Union[Column, List[Column]]:
        """
        Gets an expression appropriate for performing range operations on the :class:`TSDF` records.
        Defaults to the same expression giving by :method:`TSIndex.orderByExpr`

        :param reverse: whether the ordering should be reversed (backwards in time)

        :return: an expression appropriate for operforming range operations on the :class:`TSDF` records
        """
        return self.orderByExpr(reverse=reverse)

#
# Simple TS Index types
#

class SimpleTSIndex(TSIndex, ABC):
    """
    Abstract base class for simple Timeseries Index types
    that only reference a single column for maintaining the temporal structure
    """

    def __init__(self, ts_col: StructField) -> None:
        self.__name = ts_col.name
        self.dataType = ts_col.dataType

    @property
    def name(self):
        return self.__name

    @property
    def ts_col(self) -> str:
        return self.name

    def orderByExpr(self, reverse: bool = False) -> Column:
        expr = Fn.col(self.name)
        return self._reverseOrNot(expr, reverse)

    @classmethod
    def fromTSCol(cls, ts_col: StructField) -> "SimpleTSIndex":
        # pick our implementation based on the column type
        if isinstance(ts_col.dataType, NumericType):
            return NumericIndex(ts_col)
        elif isinstance(ts_col.dataType, TimestampType):
            return SimpleTimestampIndex(ts_col)
        elif isinstance(ts_col.dataType, DateType):
            return SimpleDateIndex(ts_col)
        else:
            raise TypeError(f"A SimpleTSIndex must be a Numeric, Timestamp or Date type, but column {ts_col.name} is of type {ts_col.dataType}")


class NumericIndex(SimpleTSIndex):
    """
    Timeseries index based on a single column of a numeric or temporal type.
    """

    def __init__(self, ts_col: StructField) -> None:
        if not isinstance(ts_col.dataType, NumericType):
            raise TypeError(f"NumericIndex must be of a numeric type, but ts_col {ts_col.name} has type {ts_col.dataType}")
        super().__init__(ts_col)


class SimpleTimestampIndex(SimpleTSIndex):
    """
    Timeseries index based on a single Timestamp column
    """

    def __init__(self, ts_col: StructField) -> None:
        if not isinstance(ts_col.dataType, TimestampType):
            raise TypeError(f"SimpleTimestampIndex must be of TimestampType, but given ts_col {ts_col.name} has type {ts_col.dataType}")
        super().__init__(ts_col)

    def rangeOrderByExpr(self, reverse: bool = False) -> Column:
        # cast timestamp to double (fractional seconds since epoch)
        expr = Fn.col(self.name).cast("double")
        return self._reverseOrNot(expr, reverse)


class SimpleDateIndex(SimpleTSIndex):
    """
    Timeseries index based on a single Date column
    """

    def __init__(self, ts_col: StructField) -> None:
        if not isinstance(ts_col.dataType, DateType):
            raise TypeError(f"DateIndex must be of DateType, but given ts_col {ts_col.name} has type {ts_col.dataType}")
        super().__init__(ts_col)

    def rangeOrderByExpr(self, reverse: bool = False) -> Column:
        # convert date to number of days since the epoch
        expr = Fn.datediff(Fn.col(self.name), Fn.lit("1970-01-01").cast("date"))
        self._reverseOrNot(expr,reverse)

#
# Compound TS Index Types
#

class CompositeTSIndex(TSIndex, ABC):
    """
    Abstract base class for complex Timeseries Index classes
    that involve two or more columns organized into a StructType column
    """

    def __init__(self, composite_ts_idx: StructField, primary_ts_col: str) -> None:
        if not isinstance(composite_ts_idx.dataType, StructType):
            raise TypeError(f"CompoundTSIndex must be of type StructType, but given compound_ts_idx {composite_ts_idx.name} has type {composite_ts_idx.dataType}")
        self.ts_idx: str = composite_ts_idx.name
        self.struct: StructType = composite_ts_idx.dataType
        # construct a simple TS index object for the primary column
        self.primary_ts_idx: SimpleTSIndex = SimpleTSIndex.fromTSCol(self.struct[primary_ts_col])

    @property
    def name(self) -> str:
        return self.ts_idx

    @property
    def ts_col(self) -> str:
        return self.primary_ts_col

    @property
    def primary_ts_col(self) -> str:
        return self.component(self.primary_ts_idx.name)

    def component(self, component_name):
        """
        Returns the full path to a component column that is within the composite index

        :param component_name: the name of the component element within the composite index

        :return: a column name that can be used to reference the component column from the :class:`TSDF`
        """
        return f"{self.name}.{self.struct[component_name].name}"

    def orderByExpr(self, reverse: bool = False) -> Column:
        # default to using the primary column
        expr = Fn.col(self.primary_ts_col)
        return self._reverseOrNot(expr,reverse)


class SubSequenceTSIndex(CompositeTSIndex):
    """
    Timeseries Index when we have a primary timeseries column and a secondary sequencing
    column that indicates the
    """

    def __init__(self, composite_ts_idx: StructField, primary_ts_col: str, sub_seq_col: str) -> None:
        super().__init__(composite_ts_idx, primary_ts_col)
        # construct a simple index for the sub-sequence column
        self.sub_sequence_idx = NumericIndex(self.struct[sub_seq_col])

    @property
    def sub_seq_col(self) -> str:
        return self.component(self.sub_sequence_idx.name)

    def orderByExpr(self, reverse: bool = False) -> List[Column]:
        # build a composite expression of the primary index followed by the sub-sequence index
        exprs = [ Fn.col(self.primary_ts_col), Fn.col(self.sub_seq_col) ]
        return self._reverseOrNot(exprs, reverse)


class ParsedTSIndex(CompositeTSIndex, ABC):
    """
    Abstract base class for timeseries indices that are parsed from a string column.
    Retains the original string form as well as the parsed column.
    """

    def __init__(self, composite_ts_idx: StructField, src_str_col: str, parsed_col: str) -> None:
        super().__init__(composite_ts_idx, primary_ts_col=parsed_col)
        src_str_field = self.struct[src_str_col]
        if not isinstance(src_str_field.dataType, StringType):
            raise TypeError(f"Source string column must be of StringType, but given column {src_str_field.name} is of type {src_str_field.dataType}")
        self.__src_str_col = src_str_col

    @property
    def src_str_col(self):
        return self.component(self.__src_str_col)


class ParsedTimestampIndex(ParsedTSIndex):
    """
    Timeseries index class for timestamps parsed from a string column
    """

    def __init__(self, composite_ts_idx: StructField, src_str_col: str, parsed_col: str) -> None:
        super().__init__(composite_ts_idx, src_str_col, parsed_col)
        if not isinstance(self.primary_ts_idx.dataType, TimestampType):
            raise TypeError(f"ParsedTimestampIndex must be of TimestampType, but given ts_col {self.primary_ts_idx.name} has type {self.primary_ts_idx.dataType}")

    def rangeOrderByExpr(self, reverse: bool = False) -> Column:
        # cast timestamp to double (fractional seconds since epoch)
        expr = Fn.col(self.primary_ts_col).cast("double")
        return self._reverseOrNot(expr, reverse)


class ParsedDateIndex(ParsedTSIndex):
    """
    Timeseries index class for dates parsed from a string column
    """

    def __init__(self, composite_ts_idx: StructField, src_str_col: str, parsed_col: str) -> None:
        super().__init__(composite_ts_idx, src_str_col, parsed_col)
        if not isinstance(self.primary_ts_idx.dataType, DateType):
            raise TypeError(f"ParsedDateIndex must be of DateType, but given ts_col {self.primary_ts_idx.name} has type {self.primary_ts_idx.dataType}")

    def rangeOrderByExpr(self, reverse: bool = False) -> Column:
        # convert date to number of days since the epoch
        expr = Fn.datediff(Fn.col(self.primary_ts_col), Fn.lit("1970-01-01").cast("date"))
        self._reverseOrNot(expr,reverse)

#
# Timseries Schema
#

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
        ts_idx = SimpleTSIndex.fromTSCol(df_schema[ts_col])
        return cls(ts_idx, series_ids)

    @property
    def structural_columns(self) -> set[str]:
        """
        Structural columns are those that define the structure of the :class:`TSDF`. This includes the timeseries column,
        a timeseries index (if different), any subsequence column (if present), and the series ID columns.

        :return: a set of column names corresponding the structural columns of a :class:`TSDF`
        """
        struct_cols = {self.ts_idx.name}.union(self.series_ids)
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
