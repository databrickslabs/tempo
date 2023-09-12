from enum import Enum, auto
from abc import ABC, abstractmethod
from typing import cast, Any, Union, Optional, Collection, List

import pyspark.sql.functions as sfn
from pyspark.sql import Column, WindowSpec, Window
from pyspark.sql.types import *
from pyspark.sql.types import NumericType
from pyspark.sql._typing import LiteralType, DecimalLiteral, DateTimeLiteral

#
# Time Units
#

class TimeUnits(Enum):
    YEARS = auto()
    MONTHS = auto()
    DAYS = auto()
    HOURS = auto()
    MINUTES = auto()
    SECONDS = auto()
    MICROSECONDS = auto()
    NANOSECONDS = auto()


#
# Timeseries Index Classes
#


class TSIndex(ABC):
    """
    Abstract base class for all Timeseries Index types
    """

    def __eq__(self, o: object) -> bool:
        # must be a SimpleTSIndex
        if not isinstance(o, TSIndex):
            return False
        return self._indexAttributes == o._indexAttributes

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"""{self.__class__.__name__}({self._indexAttributes})"""

    @property
    @abstractmethod
    def _indexAttributes(self) -> dict[str, Any]:
        """
        :return: key attributes of this index
        """

    @property
    @abstractmethod
    def colname(self) -> str:
        """
        :return: the column name of the timeseries index
        """

    @property
    @abstractmethod
    def unit(self) -> Optional[TimeUnits]:
        """
        :return: the unit of this index, that is, the unit that a range value of 1 represents (Days, seconds, etc.)
        """

    @abstractmethod
    def validate(self, df_schema: StructType) -> None:
        """
        Validate that this TSIndex is correctly represented in the given schema
        :param df_schema: the schema for a :class:`DataFrame`
        """

    @abstractmethod
    def renamed(self, new_name: str) -> "TSIndex":
        """
        Renames the index

        :param new_name: new name of the index

        :return: a copy of this :class:`TSIndex` object with the new name
        """

    def _reverseOrNot(
        self, expr: Union[Column, List[Column]], reverse: bool
    ) -> Union[Column, List[Column]]:
        if not reverse:
            return expr  # just return the expression as-is if we're not reversing
        elif isinstance(expr, Column):
            return expr.desc()  # reverse a single-expression
        elif isinstance(expr, list):
            return [col.desc() for col in expr]  # reverse all columns in the expression
        else:
            raise TypeError(f"Type for expr argument must be either Column or List[Column], instead received: {type(expr)}")

    @abstractmethod
    def orderByExpr(self, reverse: bool = False) -> Union[Column, List[Column]]:
        """
        Gets an expression that will order the :class:`TSDF` according to the timeseries index.

        :param reverse: whether the ordering should be reversed (backwards in time)

        :return: an expression appropriate for ordering the :class:`TSDF` according to this index
        """

    @abstractmethod
    def rangeExpr(self, reverse: bool = False) -> Column:
        """
        Gets an expression appropriate for performing range operations on the :class:`TSDF` records.

        :param reverse: whether the ordering should be reversed (backwards in time)

        :return: an expression appropriate for operforming range operations on the :class:`TSDF` records
        """

    @abstractmethod
    def betweenExpr(self,
                    lowerBound: Union["Column", "LiteralType", "DateTimeLiteral", "DecimalLiteral"],
                    upperBound: Union["Column", "LiteralType", "DateTimeLiteral", "DecimalLiteral"]
                    ) -> Column:
        """
        Gets an expression appropriate for performing upper and lower bound comparisons
        on the values of a timeseries index.

        :param lowerBound: the lower bound for the range comparison
        :param upperBound: the upper bound for the range comparison

        :return: an expression appropriate for performing upper and lower bound
        comparisons on the values of a timeseries index
        """

#
# Simple TS Index types
#


class SimpleTSIndex(TSIndex, ABC):
    """
    Abstract base class for simple Timeseries Index types
    that only reference a single column for maintaining the temporal structure
    """

    def __init__(self, ts_idx: StructField) -> None:
        self.__name = ts_idx.name
        self.dataType = ts_idx.dataType

    @property
    def _indexAttributes(self) -> dict[str, Any]:
        return {"name": self.colname, "dataType": self.dataType}

    @property
    def colname(self):
        return self.__name

    @property
    def ts_col(self) -> str:
        return self.colname

    def validate(self, df_schema: StructType) -> None:
        # the ts column must exist
        assert(self.colname in df_schema.fieldNames(),
               f"The TSIndex column {self.colname} "
               f"does not exist in the given DataFrame")
        schema_ts_col = df_schema[self.colname]
        # it must have the right type
        schema_ts_type = schema_ts_col.dataType
        assert(isinstance(schema_ts_type, type(self.dataType)),
               f"The TSIndex column is of type {schema_ts_type}, "
               f"but the expected type is {self.dataType}")

    def renamed(self, new_name: str) -> "TSIndex":
        self.__name = new_name
        return self

    def orderByExpr(self, reverse: bool = False) -> Column:
        expr = sfn.col(self.colname)
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
            raise TypeError(
                f"A SimpleTSIndex must be a Numeric, Timestamp or Date type, but column {ts_col.name} is of type {ts_col.dataType}"
            )


class NumericIndex(SimpleTSIndex):
    """
    Timeseries index based on a single column of a numeric or temporal type.
    """

    def __init__(self, ts_idx: StructField) -> None:
        if not isinstance(ts_idx.dataType, NumericType):
            raise TypeError(
                f"NumericIndex must be of a numeric type, "
                f"but ts_col {ts_idx.name} has type {ts_idx.dataType}"
            )
        super().__init__(ts_idx)

    @property
    def unit(self) -> Optional[TimeUnits]:
        return None

    def rangeExpr(self, reverse: bool = False) -> Column:
        return self.orderByExpr(reverse)


class SimpleTimestampIndex(SimpleTSIndex):
    """
    Timeseries index based on a single Timestamp column
    """

    def __init__(self, ts_idx: StructField) -> None:
        if not isinstance(ts_idx.dataType, TimestampType):
            raise TypeError(
                f"SimpleTimestampIndex must be of TimestampType, "
                f"but given ts_col {ts_idx.name} has type {ts_idx.dataType}"
            )
        super().__init__(ts_idx)

    @property
    def unit(self) -> Optional[TimeUnits]:
        return TimeUnits.SECONDS

    def rangeExpr(self, reverse: bool = False) -> Column:
        # cast timestamp to double (fractional seconds since epoch)
        expr = sfn.col(self.colname).cast("double")
        return self._reverseOrNot(expr, reverse)


class SimpleDateIndex(SimpleTSIndex):
    """
    Timeseries index based on a single Date column
    """

    def __init__(self, ts_idx: StructField) -> None:
        if not isinstance(ts_idx.dataType, DateType):
            raise TypeError(
                f"DateIndex must be of DateType, but given ts_col {ts_idx.name} has type {ts_idx.dataType}"
            )
        super().__init__(ts_idx)

    @property
    def unit(self) -> Optional[TimeUnits]:
        return TimeUnits.DAYS

    def rangeExpr(self, reverse: bool = False) -> Column:
        # convert date to number of days since the epoch
        expr = sfn.datediff(sfn.col(self.colname), sfn.lit("1970-01-01").cast("date"))
        return self._reverseOrNot(expr, reverse)


#
# Complex (Multi-Field) TS Index Types
#


class CompositeTSIndex(TSIndex):
    """
    Abstract base class for complex Timeseries Index classes
    that involve two or more columns organized into a StructType column
    """

    def __init__(self, ts_idx: StructField, *ts_fields: str) -> None:
        if not isinstance(ts_idx.dataType, StructType):
            raise TypeError(
                f"CompoundTSIndex must be of type StructType, "
                f"but given compound_ts_idx {ts_idx.name} has type {ts_idx.dataType}"
            )
        self.__name: str = ts_idx.name
        self.struct: StructType = ts_idx.dataType
        # handle the timestamp fields
        if ts_fields is None or len(ts_fields) < 1:
            raise ValueError("A CompoundTSIndex must have "
                             "at least one ts_field specified!")
        self.ts_components = [SimpleTSIndex.fromTSCol(self.struct[field])
                              for field in ts_fields]
        self.primary_ts_idx = self.ts_components[0]


    @property
    def _indexAttributes(self) -> dict[str, Any]:
        return {
            "name": self.colname,
            "struct": self.struct,
            "ts_components": self.ts_components
        }

    @property
    def colname(self) -> str:
        return self.__name

    @property
    def ts_col(self) -> str:
        return self.primary_ts_col

    @property
    def primary_ts_col(self) -> str:
        return self.ts_component(0)

    @property
    def unit(self) -> Optional[TimeUnits]:
        return self.primary_ts_idx.unit

    def validate(self, df_schema: StructType) -> None:
        # validate that the composite field exists
        assert(self.colname in df_schema.fieldNames(),
               f"The TSIndex column {self.colname} "
               f"does not exist in the given DataFrame")
        schema_ts_col = df_schema[self.colname]
        # it must have the right type
        schema_ts_type = schema_ts_col.dataType
        assert(isinstance(schema_ts_type, StructType),
               f"The TSIndex column is of type {schema_ts_type}, "
               f"but the expected type is {StructType}")
        # validate all the TS components
        for comp in self.ts_components:
            comp.validate(schema_ts_type)

    def renamed(self, new_name: str) -> "TSIndex":
        self.__name = new_name
        return self

    def component(self, component_name: str) -> str:
        """
        Returns the full path to a component field that is within the composite index

        :param component_name: the name of the component element within the composite index

        :return: a column name that can be used to reference the component field in PySpark expressions
        """
        return f"{self.colname}.{self.struct[component_name].name}"

    def ts_component(self, component_index: int) -> str:
        """
        Returns the full path to a component field that is a functional part of the timeseries.

        :param component_index: the index giving the ordering of the component field within the timeseries

        :return: a column name that can be used to reference the component field in PySpark expressions
        """
        return self.component(self.ts_components[component_index].colname)

    def orderByExpr(self, reverse: bool = False) -> Column:
        # build an expression for each TS component, in order
        exprs = [sfn.col(self.component(comp.colname)) for comp in self.ts_components]
        return self._reverseOrNot(exprs, reverse)

    def rangeExpr(self, reverse: bool = False) -> Column:
        return self.primary_ts_idx.rangeExpr(reverse)


class ParsedTSIndex(CompositeTSIndex, ABC):
    """
    Abstract base class for timeseries indices that are parsed from a string column.
    Retains the original string form as well as the parsed column.
    """

    def __init__(
        self, ts_idx: StructField, src_str_col: str, parsed_col: str
    ) -> None:
        super().__init__(ts_idx, parsed_col)
        src_str_field = self.struct[src_str_col]
        if not isinstance(src_str_field.dataType, StringType):
            raise TypeError(
                f"Source string column must be of StringType, but given column {src_str_field.name} is of type {src_str_field.dataType}"
            )
        self.__src_str_col = src_str_col

    @property
    def _indexAttributes(self) -> dict[str, Any]:
        attrs = super()._indexAttributes
        attrs["src_str_col"] = self.src_str_col
        return attrs

    @property
    def src_str_col(self):
        return self.component(self.__src_str_col)

    def validate(self, df_schema: StructType) -> None:
        super().validate(df_schema)
        # make sure the parsed field exists
        composite_idx_type: StructType = cast(StructType, df_schema[self.colname].dataType)
        assert(self.__src_str_col in composite_idx_type,
               f"The src_str_col column {self.src_str_col} "
               f"does not exist in the composite field {composite_idx_type}")
        # make sure it's StringType
        src_str_field_type = composite_idx_type[self.__src_str_col].dataType
        assert(isinstance(src_str_field_type, StringType),
               f"The src_str_col column {self.src_str_col} "
               f"should be of StringType, but found {src_str_field_type} instead")


class ParsedTimestampIndex(ParsedTSIndex):
    """
    Timeseries index class for timestamps parsed from a string column
    """

    def __init__(
        self, ts_idx: StructField, src_str_col: str, parsed_col: str
    ) -> None:
        super().__init__(ts_idx, src_str_col, parsed_col)
        if not isinstance(self.primary_ts_idx.dataType, TimestampType):
            raise TypeError(
                f"ParsedTimestampIndex must be of TimestampType, but given ts_col {self.primary_ts_idx.colname} has type {self.primary_ts_idx.dataType}"
            )

    def rangeExpr(self, reverse: bool = False) -> Column:
        # cast timestamp to double (fractional seconds since epoch)
        expr = sfn.col(self.primary_ts_col).cast("double")
        return self._reverseOrNot(expr, reverse)


class ParsedDateIndex(ParsedTSIndex):
    """
    Timeseries index class for dates parsed from a string column
    """

    def __init__(
        self, ts_idx: StructField, src_str_col: str, parsed_col: str
    ) -> None:
        super().__init__(ts_idx, src_str_col, parsed_col)
        if not isinstance(self.primary_ts_idx.dataType, DateType):
            raise TypeError(
                f"ParsedDateIndex must be of DateType, but given ts_col {self.primary_ts_idx.colname} has type {self.primary_ts_idx.dataType}"
            )

    def rangeExpr(self, reverse: bool = False) -> Column:
        # convert date to number of days since the epoch
        expr = sfn.datediff(
            sfn.col(self.primary_ts_col), sfn.lit("1970-01-01").cast("date")
        )
        return self._reverseOrNot(expr, reverse)


#
# Window Builder Interface
#

class WindowBuilder(ABC):
    """
    Abstract base class for window builders.
    """

    @abstractmethod
    def baseWindow(self, reverse: bool = False) -> WindowSpec:
        """
        build a basic window for sorting the Timeseries

        :param reverse: if True, sort in reverse order

        :return: a WindowSpec object
        """
        pass

    @abstractmethod
    def rowsBetweenWindow(self,
                          start: int,
                          end: int,
                          reverse: bool = False) -> WindowSpec:
        """
        build a row-based window with the given start and end offsets

        :param start: the start offset
        :param end: the end offset
        :param reverse: if True, sort in reverse order

        :return: a WindowSpec object
        """
        pass

    def allBeforeWindow(self, inclusive: bool = True) -> WindowSpec:
        """
        build a window that includes all rows before the current row

        :param inclusive: if True, include the current row,
        otherwise end with the last row before the current row

        :return: a WindowSpec object
        """
        return self.rowsBetweenWindow(Window.unboundedPreceding,
                                      0 if inclusive else -1)

    def allAfterWindow(self, inclusive: bool = True) -> WindowSpec:
        """
        build a window that includes all rows after the current row

        :param inclusive: if True, include the current row,
        otherwise begin with the first row after the current row

        :return: a WindowSpec object
        """
        return self.rowsBetweenWindow(0 if inclusive else 1,
                                      Window.unboundedFollowing)

    @abstractmethod
    def rangeBetweenWindow(self,
                           start: int,
                           end: int,
                           reverse: bool = False) -> WindowSpec:
        """
        build a range-based window with the given start and end offsets

        :param start: the start offset
        :param end: the end offset
        :param reverse: if True, sort in reverse order

        :return: a WindowSpec object
        """
        pass


#
# Timseries Schema
#


class TSSchema(WindowBuilder):
    """
    Schema type for a :class:`TSDF` class.
    """

    def __init__(self, ts_idx: TSIndex, series_ids: Collection[str] = None) -> None:
        self.__ts_idx = ts_idx
        if series_ids:
            self.__series_ids = list(series_ids)
        else:
            self.__series_ids = []

    @property
    def ts_idx(self):
        return self.__ts_idx

    @property
    def series_ids(self) -> List[str]:
        return self.__series_ids

    def __eq__(self, o: object) -> bool:
        # must be of TSSchema type
        if not isinstance(o, TSSchema):
            return False
        # must have same TSIndex
        if self.ts_idx != o.ts_idx:
            return False
        # must have the same series IDs
        if self.series_ids != o.series_ids:
            return False
        return True

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"""TSSchema({id(self)})
    TSIndex: {self.ts_idx}
    Series IDs: {self.series_ids}"""

    @classmethod
    def fromDFSchema(
        cls, df_schema: StructType, ts_col: str, series_ids: Collection[str] = None
    ) -> "TSSchema":
        # construct a TSIndex for the given ts_col
        ts_idx = SimpleTSIndex.fromTSCol(df_schema[ts_col])
        return cls(ts_idx, series_ids)

    @property
    def structural_columns(self) -> list[str]:
        """
        Structural columns are those that define the structure of the :class:`TSDF`. This includes the timeseries column,
        a timeseries index (if different), any subsequence column (if present), and the series ID columns.

        :return: a set of column names corresponding the structural columns of a :class:`TSDF`
        """
        return list({self.ts_idx.colname}.union(self.series_ids))

    def validate(self, df_schema: StructType) -> None:
        # ensure that the TSIndex is valid
        self.ts_idx.validate(df_schema)
        # check series IDs
        for sid in self.series_ids:
            assert( sid in df_schema.fieldNames(),
                    f"Series ID {sid} does not exist in the given DataFrame" )

    def find_observational_columns(self, df_schema: StructType) -> list[str]:
        return list(set(df_schema.fieldNames()) - set(self.structural_columns))

    @classmethod
    def is_metric_col(cls, col: StructField) -> bool:
        return isinstance(col.dataType, NumericType) or isinstance(
            col.dataType, BooleanType
        )

    def find_metric_columns(self, df_schema: StructType) -> list[str]:
        return [
            col.name
            for col in df_schema.fields
            if self.is_metric_col(col)
            and (col.name in self.find_observational_columns(df_schema))
        ]

    def baseWindow(self, reverse: bool = False) -> WindowSpec:
        # The index will determine the appropriate sort order
        w = Window().orderBy(self.ts_idx.orderByExpr(reverse))

        # and partitioned by any series IDs
        if self.series_ids:
            w = w.partitionBy([sfn.col(sid) for sid in self.series_ids])
        return w

    def rowsBetweenWindow(self, start: int, end: int, reverse: bool = False) -> WindowSpec:
        return self.baseWindow(reverse=reverse).rowsBetween(start, end)

    def rangeBetweenWindow(self, start: int, end: int, reverse: bool = False) -> WindowSpec:
        return (
            self.baseWindow(reverse=reverse)
            .orderBy(self.ts_idx.rangeExpr(reverse=reverse))
            .rangeBetween(start, end)
        )


