import warnings
import re
from abc import ABC, abstractmethod
from typing import Any, Collection, List, Optional, Union

import pyspark.sql.functions as sfn
from pyspark.sql import Column, Window, WindowSpec
from pyspark.sql.types import *
from pyspark.sql.types import NumericType

from tempo.timeunit import TimeUnit, StandardTimeUnits

#
# Timestamp parsing helpers
#

DEFAULT_TIMESTAMP_FORMAT = "yyyy-MM-dd HH:mm:ss"
__time_pattern_components = "hHkKmsS"


def is_time_format(ts_fmt: str) -> bool:
    """
    Checcks whether the given format string contains time elements,
    or if it is just a date format

    :param ts_fmt: the format string to check

    :return: whether the given format string contains time elements
    """
    return any(c in ts_fmt for c in __time_pattern_components)


def sub_seconds_precision_digits(ts_fmt: str) -> int:
    """
    Returns the number of digits of precision for a timestamp format string
    """
    # pattern for matching the sub-second precision digits
    sub_seconds_ptrn = r"\.(\S+)"
    # find the sub-second precision digits
    match = re.search(sub_seconds_ptrn, ts_fmt)
    if match is None:
        return 0
    else:
        return len(match.group(1))

#
# Abstract Timeseries Index Classes
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
    def ts_col(self) -> str:
        """
        :return: the name of the primary timeseries column (may or may not be the same as the name)
        """

    @property
    @abstractmethod
    def unit(self) -> Optional[TimeUnit]:
        """
        :return: the unit of this index, that is, the unit that a range value of 1 represents (Days, seconds, etc.)
        """

    @property
    def has_unit(self) -> bool:
        """
        :return: whether this index has a unit
        """
        return self.unit is not None

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


class SimpleTSIndex(TSIndex, ABC):
    """
    Abstract base class for simple Timeseries Index types
    that only reference a single column for maintaining the temporal structure
    """

    def __init__(self, ts_col: StructField) -> None:
        self.__name = ts_col.name
        self.dataType = ts_col.dataType

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
        assert self.colname in df_schema.fieldNames(), \
            f"The TSIndex column {self.colname} does not exist in the given DataFrame"
        schema_ts_col = df_schema[self.colname]
        # it must have the right type
        schema_ts_type = schema_ts_col.dataType
        assert isinstance(schema_ts_type, type(self.dataType)), \
            f"The TSIndex column is of type {schema_ts_type}, but the expected type is {self.dataType}"

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
            return OrdinalTSIndex(ts_col)
        elif isinstance(ts_col.dataType, TimestampType):
            return SimpleTimestampIndex(ts_col)
        elif isinstance(ts_col.dataType, DateType):
            return SimpleDateIndex(ts_col)
        else:
            raise TypeError(
                f"A SimpleTSIndex must be a Numeric, Timestamp or Date type, but column {ts_col.name} is of type {ts_col.dataType}"
            )


#
# Simple TS Index types
#

class OrdinalTSIndex(SimpleTSIndex):
    """
    Timeseries index based on a single column of a numeric or temporal type.
    This index is "unitless", meaning that it is not associated with any
    particular unit of time. It can provide ordering of records, but not
    range operations.
    """

    def __init__(self, ts_col: StructField) -> None:
        if not isinstance(ts_col.dataType, NumericType):
            raise TypeError(
                f"OrdinalTSIndex must be of a numeric type, but ts_col {ts_col.name} "
                f"has type {ts_col.dataType}"
            )
        super().__init__(ts_col)

    @property
    def unit(self) -> Optional[TimeUnit]:
        return None

    def rangeExpr(self, reverse: bool = False) -> Column:
        raise TypeError("Cannot perform range operations on an OrdinalTSIndex")


class SimpleTimestampIndex(SimpleTSIndex):
    """
    Timeseries index based on a single Timestamp column
    """

    def __init__(self, ts_col: StructField) -> None:
        if not isinstance(ts_col.dataType, TimestampType):
            raise TypeError(
                f"SimpleTimestampIndex must be of TimestampType, "
                f"but given ts_col {ts_col.name} has type {ts_col.dataType}"
            )
        super().__init__(ts_col)

    @property
    def unit(self) -> Optional[TimeUnit]:
        return StandardTimeUnits.SECONDS

    def rangeExpr(self, reverse: bool = False) -> Column:
        # cast timestamp to double (fractional seconds since epoch)
        expr = sfn.col(self.colname).cast("double")
        return self._reverseOrNot(expr, reverse)


class SimpleDateIndex(SimpleTSIndex):
    """
    Timeseries index based on a single Date column
    """

    def __init__(self, ts_col: StructField) -> None:
        if not isinstance(ts_col.dataType, DateType):
            raise TypeError(
                f"DateIndex must be of DateType, "
                f"but given ts_col {ts_col.name} has type {ts_col.dataType}"
            )
        super().__init__(ts_col)

    @property
    def unit(self) -> Optional[TimeUnit]:
        return StandardTimeUnits.DAYS

    def rangeExpr(self, reverse: bool = False) -> Column:
        # convert date to number of days since the epoch
        expr = sfn.datediff(sfn.col(self.colname), sfn.lit("1970-01-01").cast("date"))
        return self._reverseOrNot(expr, reverse)


#
# Multi-Part TS Index types
#

class MultiPartTSIndex(TSIndex, ABC):
    """
    Abstract base class for Timeseries Index types that reference multiple columns.
    Such columns are organized as a StructType column with multiple fields.
    """

    def __init__(self, ts_struct: StructField) -> None:
        if not isinstance(ts_struct.dataType, StructType):
            raise TypeError(
                f"CompoundTSIndex must be of type StructType, but given "
                f"ts_struct {ts_struct.name} has type {ts_struct.dataType}"
            )
        self.__name: str = ts_struct.name
        self.schema: StructType = ts_struct.dataType

    @property
    def _indexAttributes(self) -> dict[str, Any]:
        return {"name": self.colname, "schema": self.schema}

    @property
    def colname(self) -> str:
        return self.__name

    def validate(self, df_schema: StructType) -> None:
        # validate that the composite field exists
        assert self.colname in df_schema.fieldNames(),\
            f"The TSIndex column {self.colname} does not exist in the given DataFrame"
        schema_ts_col = df_schema[self.colname]
        # it must have the right type
        schema_ts_type = schema_ts_col.dataType
        assert schema_ts_type == self.schema,\
            f"The TSIndex column is of type {schema_ts_type}, "\
            f"but the expected type is {self.schema}"

    def renamed(self, new_name: str) -> "TSIndex":
        self.__name = new_name
        return self

    def fieldPath(self, field: str) -> str:
        """
        :param field: The name of a field within the TSIndex column

        :return: A dot-separated path to the given field within the TSIndex column
        """
        assert field in self.schema.fieldNames(),\
            f"Field {field} does not exist in the TSIndex schema {self.schema}"
        return f"{self.colname}.{field}"


#
# Parsed TS Index types
#


class ParsedTSIndex(MultiPartTSIndex, ABC):
    """
    Abstract base class for timeseries indices that are parsed from a string column.
    Retains the original string form as well as the parsed column.
    """

    def __init__(
        self, ts_struct: StructField, parsed_ts_col: str, src_str_col: str
    ) -> None:
        super().__init__(ts_struct)
        # validate the source string column
        src_str_field = self.schema[src_str_col]
        if not isinstance(src_str_field.dataType, StringType):
            raise TypeError(
                f"Source string column must be of StringType, "
                f"but given column {src_str_field.name} "
                f"is of type {src_str_field.dataType}"
            )
        self._src_str_col = src_str_col
        # validate the parsed column
        assert parsed_ts_col in self.schema.fieldNames(),\
            f"The parsed timestamp index field {parsed_ts_col} does not exist in the " \
            f"MultiPart TSIndex schema {self.schema}"
        self._parsed_ts_col = parsed_ts_col

    @property
    def src_str_col(self):
        return self.fieldPath(self._src_str_col)

    @property
    def parsed_ts_col(self):
        return self.fieldPath(self._parsed_ts_col)

    @property
    def ts_col(self) -> str:
        return self.parsed_ts_col

    @property
    def _indexAttributes(self) -> dict[str, Any]:
        attrs = super()._indexAttributes
        attrs["parsed_ts_col"] = self.parsed_ts_col
        attrs["src_str_col"] = self.src_str_col
        return attrs

    def orderByExpr(self, reverse: bool = False) -> Union[Column, List[Column]]:
        expr = sfn.col(self.parsed_ts_col)
        return self._reverseOrNot(expr, reverse)

    @classmethod
    def fromParsedTimestamp(cls,
                            ts_struct: StructField,
                            parsed_ts_col: str,
                            src_str_col: str,
                            double_ts_col: Optional[str] = None,
                            num_precision_digits: int = 6) -> "ParsedTSIndex":
        """
        Create a ParsedTimestampIndex from a string column containing timestamps or dates

        :param ts_struct: The StructField for the TSIndex column
        :param parsed_ts_col: The name of the parsed timestamp column
        :param src_str_col: The name of the source string column
        :param double_ts_col: The name of the double-precision timestamp column
        :param num_precision_digits: The number of digits that make up the precision of

        :return: A ParsedTSIndex object
        """

        # if a double timestamp column is given
        # then we are building a SubMicrosecondPrecisionTimestampIndex
        if double_ts_col is not None:
            return SubMicrosecondPrecisionTimestampIndex(ts_struct,
                                                         double_ts_col,
                                                         parsed_ts_col,
                                                         src_str_col,
                                                         num_precision_digits)
        # otherwise, we base it on the standard timestamp type
        # find the schema of the ts_struct column
        ts_schema = ts_struct.dataType
        if not isinstance(ts_schema, StructType):
            raise TypeError(
                f"A ParsedTSIndex must be of type StructType, but given "
                f"ts_struct {ts_struct.name} has type {ts_struct.dataType}"
            )
        # get the type of the parsed timestamp column
        parsed_ts_type = ts_schema[parsed_ts_col].dataType
        if isinstance(parsed_ts_type, TimestampType):
            return ParsedTimestampIndex(ts_struct, parsed_ts_col, src_str_col)
        elif isinstance(parsed_ts_type, DateType):
            return ParsedDateIndex(ts_struct, parsed_ts_col, src_str_col)
        else:
            raise TypeError(
                f"ParsedTimestampIndex must be of TimestampType or DateType, "
                f"but given ts_col {parsed_ts_col} "
                f"has type {parsed_ts_type}"
            )



class ParsedTimestampIndex(ParsedTSIndex):
    """
    Timeseries index class for timestamps parsed from a string column
    """

    def __init__(
        self, ts_struct: StructField, parsed_ts_col: str, src_str_col: str
    ) -> None:
        super().__init__(ts_struct, parsed_ts_col, src_str_col)
        # validate the parsed column as a timestamp column
        parsed_ts_field = self.schema[self._parsed_ts_col]
        if not isinstance(parsed_ts_field.dataType, TimestampType):
            raise TypeError(
                f"ParsedTimestampIndex must be of TimestampType, "
                f"but given ts_col {self.parsed_ts_col} "
                f"has type {parsed_ts_field.dataType}"
            )

    def rangeExpr(self, reverse: bool = False) -> Column:
        # cast timestamp to double (fractional seconds since epoch)
        expr = sfn.col(self.parsed_ts_col).cast("double")
        return self._reverseOrNot(expr, reverse)

    @property
    def unit(self) -> Optional[TimeUnit]:
        return StandardTimeUnits.SECONDS


class SubMicrosecondPrecisionTimestampIndex(ParsedTimestampIndex):
    """
    Timeseries index class for timestamps with sub-microsecond precision
    parsed from a string column. Internally, the timestamps are stored as
    doubles (fractional seconds since epoch), as well as the original string
    and a micro-second precision (standard) timestamp field.
    """

    def __init__(self,
                 ts_struct: StructField,
                 double_ts_col: str,
                 parsed_ts_col: str,
                 src_str_col: str,
                 num_precision_digits: int = 9) -> None:
        """
        :param ts_struct: The StructField for the TSIndex column
        :param double_ts_col: The name of the double-precision timestamp column
        :param parsed_ts_col: The name of the parsed timestamp column
        :param src_str_col: The name of the source string column
        :param num_precision_digits: The number of digits that make up the precision of
        the timestamp. Ie. 9 for nanoseconds (default), 12 for picoseconds, etc.
        You will receive a warning if this value is 6 or less, as this is the precision
        of the standard timestamp type.
        """
        super().__init__(ts_struct, parsed_ts_col, src_str_col)
        # set & validate the double timestamp column
        self.double_ts_col = double_ts_col
        # validate the double timestamp column
        double_ts_field = self.schema[self.double_ts_col]
        if not isinstance(double_ts_field.dataType, DoubleType):
            raise TypeError(
                f"The double_ts_col must be of DoubleType, "
                f"but the given double_ts_col {self.double_ts_col} "
                f"has type {double_ts_field.dataType}"
            )
        # validate the number of precision digits
        if num_precision_digits <= 6:
            warnings.warn(
                f"SubMicrosecondPrecisionTimestampIndex has a num_precision_digits "
                f"of {num_precision_digits} which is within the range of the "
                f"standard timestamp precision of 6 digits (microseconds). "
                f"Consider using a ParsedTimestampIndex instead."
            )
        self.__unit = TimeUnit(
            f"custom_subsecond_unit (precision: {num_precision_digits})",
            10 ** (-num_precision_digits),
            num_precision_digits,
        )

    def orderByExpr(self, reverse: bool = False) -> Union[Column, List[Column]]:
        expr = sfn.col(self.double_ts_col)
        return self._reverseOrNot(expr, reverse)

    def rangeExpr(self, reverse: bool = False) -> Column:
        # just use the order by expression, since this is the same
        return self.orderByExpr(reverse)

    @property
    def unit(self) -> Optional[TimeUnit]:
        return self.__unit


class ParsedDateIndex(ParsedTSIndex):
    """
    Timeseries index class for dates parsed from a string column
    """

    def __init__(
        self, ts_struct: StructField, parsed_ts_col: str, src_str_col: str
    ) -> None:
        super().__init__(ts_struct, parsed_ts_col, src_str_col)
        # validate the parsed column as a date column
        parsed_ts_field = self.schema[self._parsed_ts_col]
        if not isinstance(parsed_ts_field.dataType, DateType):
            raise TypeError(
                f"ParsedTimestampIndex must be of DateType, "
                f"but given ts_col {self.parsed_ts_col} "
                f"has type {parsed_ts_field.dataType}"
            )

    @property
    def unit(self) -> Optional[TimeUnit]:
        return StandardTimeUnits.DAYS

    def rangeExpr(self, reverse: bool = False) -> Column:
        # convert date to number of days since the epoch
        expr = sfn.datediff(
            sfn.col(self.parsed_ts_col), sfn.lit("1970-01-01").cast("date")
        )
        return self._reverseOrNot(expr, reverse)


#
# Complex (Multi-Field) TS Index Types
#


class CompositeTSIndex(MultiPartTSIndex, ABC):
    """
    Abstract base class for complex Timeseries Index classes
    that involve two or more columns organized into a StructType column
    """

    def __init__(self, ts_struct: StructField, *ts_fields: str) -> None:
        super().__init__(ts_struct)
        # handle the timestamp fields
        assert len(ts_fields) > 1,\
            f"CompositeTSIndex must have at least two timestamp fields, " \
            f"but only {len(ts_fields)} were given"
        self.ts_components = \
            [SimpleTSIndex.fromTSCol(self.schema[field]) for field in ts_fields]

    @property
    def _indexAttributes(self) -> dict[str, Any]:
        attrs = super()._indexAttributes
        attrs["ts_components"] = [str(c) for c in self.ts_components]
        return attrs

    def primary_ts_col(self) -> str:
        return self.get_ts_component(0)

    @property
    def primary_ts_idx(self) -> TSIndex:
        return self.ts_components[0]

    @property
    def unit(self) -> Optional[TimeUnit]:
        return self.primary_ts_idx.unit

    def validate(self, df_schema: StructType) -> None:
        super().validate(df_schema)
        # validate all the TS components
        schema_ts_type = df_schema[self.colname].dataType
        assert isinstance(schema_ts_type, StructType),\
            f"CompositeTSIndex must be of StructType, " \
            f"but given ts_col {self.colname} " \
            f"has type {schema_ts_type}"
        for comp in self.ts_components:
            comp.validate(schema_ts_type)

    def get_ts_component(self, component_index: int) -> str:
        """
        Returns the full path to a component field that is a functional part of the timeseries.

        :param component_index: the index giving the ordering of the component field within the timeseries

        :return: a column name that can be used to reference the component field in PySpark expressions
        """
        return self.fieldPath(self.ts_components[component_index].colname)

    def orderByExpr(self, reverse: bool = False) -> Column:
        # build an expression for each TS component, in order
        exprs = [sfn.col(self.fieldPath(comp.colname)) for comp in self.ts_components]
        return self._reverseOrNot(exprs, reverse)

    def rangeExpr(self, reverse: bool = False) -> Column:
        return self.primary_ts_idx.rangeExpr(reverse)


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
            assert sid in df_schema.fieldNames(), \
                f"Series ID {sid} does not exist in the given DataFrame"

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


