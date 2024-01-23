import re
import warnings
from abc import ABC, abstractmethod
from typing import Collection, Tuple, List, Optional, Union, Iterator

import pyspark.sql.functions as sfn
from pyspark.sql import Column, Window, WindowSpec
from pyspark.sql.types import *
from pyspark.sql.types import NumericType

from tempo.typing import AnyLiteral
from tempo.timeunit import TimeUnit, StandardTimeUnits

#
# Timestamp parsing helpers
#

EPOCH_START_DATE = "1970-01-01"
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


def _col_or_lit(other) -> Column:
    """
    Helper function for managing unknown argument types into
    Column expressions

    :param other: the argument to convert to a Column expression

    :return: a Column expression
    """
    if isinstance(other, (list, tuple)):
        if len(other) != 1:
            raise ValueError(
                "Cannot compare a TSIndex with a list or tuple "
                f"of length {len(other)}: {other}"
            )
        return _col_or_lit(other[0])
    if isinstance(other, Column):
        return other
    else:
        return sfn.lit(other)


def _reverse_or_not(
    expr: Union[Column, List[Column]], reverse: bool
) -> Union[Column, List[Column]]:
    """
    Helper function for reversing the ordering of an expression, if necessary

    :param expr: the expression to reverse
    :param reverse: whether to reverse the expression

    :return: the expression, reversed if necessary
    """
    if not reverse:
        return expr  # just return the expression as-is if we're not reversing
    elif isinstance(expr, Column):
        return expr.desc()  # reverse a single-expression
    elif isinstance(expr, list):
        return [col.desc() for col in expr]  # reverse all columns in the expression
    else:
        raise TypeError(
            "Type for expr argument must be either Column or "
            f"List[Column], instead received: {type(expr)}"
        )


#
# Abstract Timeseries Index Classes
#


class TSIndex(ABC):
    """
    Abstract base class for all Timeseries Index types
    """

    @property
    @abstractmethod
    def colname(self) -> str:
        """
        :return: the column name of the timeseries index
        """

    @property
    @abstractmethod
    def dataType(self) -> DataType:
        """
        :return: the data type of the timeseries index
        """

    @property
    def is_composite(self) -> bool:
        """
        :return: whether this index is a composite index
        """
        return isinstance(self, CompositeTSIndex)

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

    # comparators
    # Generate column expressions that compare the index
    # with other columns, expressions or values

    @abstractmethod
    def comparableExpr(self) -> Union[Column, List[Column]]:
        """
        :return: an expression that can be used to compare an index with
        other columns, expressions or values
        """

    @abstractmethod
    def is_comparable(self, other: "TSIndex") -> bool:
        """
        :param other: the other TSIndex to compare with

        :return: whether this TSIndex can be compared with the other TSIndex
        """

    def __eq__(self, other) -> Column:
        return self.comparableExpr() == _col_or_lit(other)

    def __ne__(self, other) -> Column:
        return self.comparableExpr() != _col_or_lit(other)

    def __lt__(self, other) -> Column:
        return self.comparableExpr() < _col_or_lit(other)

    def __le__(self, other) -> Column:
        return self.comparableExpr() <= _col_or_lit(other)

    def __gt__(self, other) -> Column:
        return self.comparableExpr() > _col_or_lit(other)

    def __ge__(self, other) -> Column:
        return self.comparableExpr() >= _col_or_lit(other)

    def between(self, lowerBound, upperBound) -> "Column":
        """
        A boolean expression that is evaluated to true if the value of this expression is between the given columns.

        :param lowerBound: The lower bound of the range
        :param upperBound: The upper bound of the range

        :return: A boolean expression
        """
        return self.comparableExpr().between(lowerBound, upperBound)

    # other expression builder methods

    @abstractmethod
    def orderByExpr(self, reverse: bool = False) -> Union[Column, List[Column]]:
        """
        Gets an expression that will order the :class:`TSDF`
        according to the timeseries index.

        :param reverse: whether the ordering should be reversed (backwards in time)

        :return: an expression appropriate for ordering the :class:`TSDF`
        according to this index
        """

    @abstractmethod
    def rangeExpr(self, reverse: bool = False) -> Column:
        """
        Gets an expression appropriate for performing range operations
        on the :class:`TSDF` records.

        :param reverse: whether the ordering should be reversed (backwards in time)

        :return: an expression appropriate for performing range operations
        on the :class:`TSDF` records
        """


class SimpleTSIndex(TSIndex, ABC):
    """
    Abstract base class for simple Timeseries Index types
    that only reference a single column for maintaining the temporal structure
    """

    def __init__(self, ts_col: StructField) -> None:
        self.__name = ts_col.name
        self.__dataType = ts_col.dataType

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.colname}, "
            f"type={self.dataType}, unit={self.unit})"
        )

    @property
    def colname(self):
        return self.__name

    @property
    def dataType(self) -> DataType:
        return self.__dataType

    def validate(self, df_schema: StructType) -> None:
        # the ts column must exist
        assert (
            self.colname in df_schema.fieldNames()
        ), f"The TSIndex column {self.colname} does not exist in the given DataFrame"
        schema_ts_col = df_schema[self.colname]
        # it must have the right type
        schema_ts_type = schema_ts_col.dataType
        assert isinstance(schema_ts_type, type(self.dataType)), (
            f"The TSIndex column is of type {schema_ts_type}, but the expected type is"
            f" {self.dataType}"
        )

    def renamed(self, new_name: str) -> "TSIndex":
        self.__name = new_name
        return self

    def comparableExpr(self) -> Column:
        return sfn.col(self.colname)

    def is_comparable(self, other: "TSIndex") -> bool:
        if isinstance(other, SimpleTSIndex):
            return self.dataType == other.dataType
        return other.is_comparable(self)

    def orderByExpr(self, reverse: bool = False) -> Column:
        return _reverse_or_not(self.comparableExpr(), reverse)

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
                "A SimpleTSIndex must be a Numeric, Timestamp or Date type, but column"
                f" {ts_col.name} is of type {ts_col.dataType}"
            )


#
# Simple TS Index types
#


class OrdinalTSIndex(SimpleTSIndex):
    """
    Timeseries index based on a single column of a numeric type.
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
        raise NotImplementedError(
            "Cannot perform range operations on an OrdinalTSIndex"
        )


class SimpleTimestampIndex(SimpleTSIndex):
    """
    Timeseries index based on a single Timestamp column
    """

    def __init__(self, ts_col: StructField) -> None:
        if not isinstance(ts_col.dataType, TimestampType):
            raise TypeError(
                "SimpleTimestampIndex must be of TimestampType, "
                f"but given ts_col {ts_col.name} has type {ts_col.dataType}"
            )
        super().__init__(ts_col)

    @property
    def unit(self) -> Optional[TimeUnit]:
        return StandardTimeUnits.SECONDS

    def rangeExpr(self, reverse: bool = False) -> Column:
        # cast timestamp to double (fractional seconds since epoch)
        expr = self.comparableExpr().cast("double")
        return _reverse_or_not(expr, reverse)


class SimpleDateIndex(SimpleTSIndex):
    """
    Timeseries index based on a single Date column
    """

    def __init__(self, ts_col: StructField) -> None:
        if not isinstance(ts_col.dataType, DateType):
            raise TypeError(
                "DateIndex must be of DateType, "
                f"but given ts_col {ts_col.name} has type {ts_col.dataType}"
            )
        super().__init__(ts_col)

    @property
    def unit(self) -> Optional[TimeUnit]:
        return StandardTimeUnits.DAYS

    def rangeExpr(self, reverse: bool = False) -> Column:
        # convert date to number of days since the epoch
        expr = sfn.datediff(
            self.comparableExpr(), sfn.lit(EPOCH_START_DATE).cast("date")
        )
        return _reverse_or_not(expr, reverse)


#
# Complex (Multi-Field) TS Index types
#


class CompositeTSIndex(TSIndex, ABC):
    """
    Abstract base class for Timeseries Index types that reference multiple columns.
    Such columns are organized as a StructType column with multiple fields.
    Some subset of these columns (at least 1) is considered to be a "component field",
    the others are called "accessory fields".
    """

    def __init__(self, ts_struct: StructField, *component_fields: str) -> None:
        if not isinstance(ts_struct.dataType, StructType):
            raise TypeError(
                "CompoundTSIndex must be of type StructType, but given "
                f"ts_struct {ts_struct.name} has type {ts_struct.dataType}"
            )
        # validate the index fields
        assert len(component_fields) > 0, (
            "A MultiFieldTSIndex must have at least 1 index component field, "
            f"but {len(component_fields)} were given"
        )
        for ind_f in component_fields:
            assert (
                ind_f in ts_struct.dataType.fieldNames()
            ), f"Index field {ind_f} does not exist in the given TSIndex schema"
        # assign local attributes
        self.__name: str = ts_struct.name
        self.schema: StructType = ts_struct.dataType
        self.component_fields: List[str] = list(component_fields)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.colname}, "
            f"schema={self.schema}, unit={self.unit}, "
            f"component_fields={self.component_fields})"
        )

    @property
    def colname(self) -> str:
        return self.__name

    @property
    def dataType(self) -> DataType:
        return self.schema

    @property
    def fieldNames(self) -> List[str]:
        return self.schema.fieldNames()

    @property
    def accessory_fields(self) -> List[str]:
        return list(set(self.fieldNames) - set(self.component_fields))

    def renamed(self, new_name: str) -> "TSIndex":
        self.__name = new_name
        return self

    def fieldPath(self, field: str) -> str:
        """
        :param field: The name of a field within the TSIndex column

        :return: A dot-separated path to the given field within the TSIndex column
        """
        assert (
            field in self.fieldNames
        ), f"Field {field} does not exist in the TSIndex schema {self.schema}"
        return f"{self.colname}.{field}"

    def validate(self, df_schema: StructType) -> None:
        # validate that the composite field exists
        assert (
            self.colname in df_schema.fieldNames()
        ), f"The TSIndex column {self.colname} does not exist in the given DataFrame"
        schema_ts_col = df_schema[self.colname]
        # it must have the right type
        schema_ts_type = schema_ts_col.dataType
        assert schema_ts_type == self.schema, (
            f"The TSIndex column is of type {schema_ts_type}, "
            f"but the expected type is {self.schema}"
        )

    # expression builder methods

    def comparableExpr(self) -> List[Column]:
        return [sfn.col(self.fieldPath(comp)) for comp in self.component_fields]

    def is_comparable(self, other: "TSIndex") -> bool:
        # the types of our component fields
        my_comp_types = [self.schema[f].dataType for f in self.component_fields]
        # if other is a CompositeTSIndex,
        if isinstance(other, CompositeTSIndex):
            # then we compare the types of the component fields
            other_comp_types = [other.schema[f].dataType for f in other.component_fields]
            return my_comp_types == other_comp_types
        else:
            # otherwise, we compare to a single type
            return my_comp_types == [other.dataType]

    def orderByExpr(self, reverse: bool = False) -> Union[Column, List[Column]]:
        return _reverse_or_not(self.comparableExpr(), reverse)

    # comparators

    def _validate_other(self, other: Union[Tuple, List]) -> None:
        if len(other) != len(self.component_fields):
            raise ValueError(
                f"{self.__class__.__name__} has {len(self.component_fields)} "
                "component fields, and requires this many arguments for comparison, "
                f"but received {len(other)}"
            )

    def _expand_comps(self, other) -> List[Column]:
        # if other is another TSIndex,
        # then we evaluate against its comparable expressions
        if isinstance(other, TSIndex):
            return self._expand_comps(other.comparableExpr())
        # try to compare the whole index to a single value
        if not isinstance(other, (tuple, list)):
            return self._expand_comps([other])
        # validate the number of arguments
        self._validate_other(other)
        # if not a column, then a literal
        return [_col_or_lit(o) for o in other]

    def _build_comps(self, other) -> Iterator[Column]:
        # match each component field with its corresponding comparison value
        return zip(self.comparableExpr(), self._expand_comps(other))

    def __eq__(self, other) -> Column:
        # match each component field with its corresponding comparison value
        comps = self._build_comps(other)
        # build comparison expressions for each pair
        comp_exprs: list[Column] = [(c == o) for (c, o) in comps]
        # conjunction of all expressions (AND)
        if len(comp_exprs) > 1:
            return sfn.expr(" AND ".join(comp_exprs))
        else:
            return comp_exprs[0]

    def __ne__(self, other) -> Column:
        # match each component field with its corresponding comparison value
        comps = self._build_comps(other)
        # build comparison expressions for each pair
        comp_exprs = [(c != o) for (c, o) in comps]
        # disjunction of all expressions (OR)
        if len(comp_exprs) > 1:
            return sfn.expr(" OR ".join(comp_exprs))
        else:
            return comp_exprs[0]

    def __lt__(self, other) -> Column:
        # match each component field with its corresponding comparison value
        comps = list(self._build_comps(other))
        # do a leq for all but the last component
        comp_exprs = []
        if len(comps) > 1:
            comp_exprs = [(c <= o) for (c, o) in comps[:-1]]
        # strict lt for the last component
        comp_exprs += [(c < o) for (c, o) in comps[-1:]]
        # conjunction of all expressions (AND)
        if len(comp_exprs) > 1:
            return sfn.expr(" AND ".join(comp_exprs))
        else:
            return comp_exprs[0]

    def __le__(self, other) -> Column:
        # match each component field with its corresponding comparison value
        comps = self._build_comps(other)
        # build comparison expressions for each pair
        comp_exprs = [(c <= o) for (c, o) in comps]
        # conjunction of all expressions (AND)
        if len(comp_exprs) > 1:
            return sfn.expr(" AND ".join(comp_exprs))
        else:
            return comp_exprs[0]

    def __gt__(self, other) -> Column:
        # match each component field with its corresponding comparison value
        comps = list(self._build_comps(other))
        # do a geq for all but the last component
        comp_exprs = []
        if len(comps) > 1:
            comp_exprs = [(c >= o) for (c, o) in comps[:-1]]
        # strict gt for the last component
        comp_exprs += [(c > o) for (c, o) in comps[-1:]]
        # conjunction of all expressions (AND)
        if len(comp_exprs) > 1:
            return sfn.expr(" AND ".join(comp_exprs))
        else:
            return comp_exprs[0]

    def __ge__(self, other) -> Column:
        # match each component field with its corresponding comparison value
        comps = self._build_comps(other)
        # build comparison expressions for each pair
        comp_exprs = [(c >= o) for (c, o) in comps]
        # conjunction of all expressions (AND)
        if len(comp_exprs) > 1:
            return sfn.expr(" AND ".join(comp_exprs))
        else:
            return comp_exprs[0]

    def between(self, lowerBound, upperBound) -> "Column":
        # match each component field with its
        # corresponding lower and upper bound values
        comps = zip(self.comparableExpr(),
                    self._expand_comps(lowerBound),
                    self._expand_comps(upperBound))
        # build comparison expressions for each triple
        comp_exprs = [(c.between(lb, ub)) for (c, lb, ub) in comps]
        # conjunction of all expressions (AND)
        if len(comp_exprs) > 1:
            return sfn.expr(" AND ".join(comp_exprs))
        else:
            return comp_exprs[0]

#
# Parsed TS Index types
#


class ParsedTSIndex(CompositeTSIndex, ABC):
    """
    Abstract base class for timeseries indices that are parsed from a string column.
    Retains the original string form as well as the parsed column.
    """

    def __init__(
        self, ts_struct: StructField, parsed_ts_field: str, src_str_field: str
    ) -> None:
        super().__init__(ts_struct, parsed_ts_field)
        # validate the source string column
        src_str_type = self.schema[src_str_field].dataType
        if not isinstance(src_str_type, StringType):
            raise TypeError(
                "Source string column must be of StringType, "
                f"but given column {src_str_field} "
                f"is of type {src_str_type}"
            )
        self._src_str_field = src_str_field
        # validate the parsed column
        assert parsed_ts_field in self.schema.fieldNames(), (
            f"The parsed timestamp index field {parsed_ts_field} does not exist in the "
            f"MultiPart TSIndex schema {self.schema}"
        )
        self._parsed_ts_field = parsed_ts_field

    @property
    def src_str_field(self):
        return self.fieldPath(self._src_str_field)

    @property
    def parsed_ts_field(self):
        return self.fieldPath(self._parsed_ts_field)

    @classmethod
    def fromParsedTimestamp(
        cls,
        ts_struct: StructField,
        parsed_ts_col: str,
        src_str_col: str,
        double_ts_col: Optional[str] = None,
        num_precision_digits: int = 6,
    ) -> "ParsedTSIndex":
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
        # if double_ts_col is not None:
        #     return SubMicrosecondPrecisionTimestampIndex(
        #         ts_struct,
        #         double_ts_col,
        #         parsed_ts_col,
        #         src_str_col,
        #         num_precision_digits,
        #     )
        # otherwise, we base it on the standard timestamp type
        # find the schema of the ts_struct column
        ts_schema = ts_struct.dataType
        if not isinstance(ts_schema, StructType):
            raise TypeError(
                "A ParsedTSIndex must be of type StructType, but given "
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
                "ParsedTimestampIndex must be of TimestampType or DateType, "
                f"but given ts_col {parsed_ts_col} "
                f"has type {parsed_ts_type}"
            )


class ParsedTimestampIndex(ParsedTSIndex):
    """
    Timeseries index class for timestamps parsed from a string column
    """

    @property
    def unit(self) -> Optional[TimeUnit]:
        return StandardTimeUnits.SECONDS

    def rangeExpr(self, reverse: bool = False) -> Column:
        # cast timestamp to double (fractional seconds since epoch)
        expr = sfn.col(self.parsed_ts_field).cast("double")
        return _reverse_or_not(expr, reverse)


class ParsedDateIndex(ParsedTSIndex):
    """
    Timeseries index class for dates parsed from a string column
    """

    @property
    def unit(self) -> Optional[TimeUnit]:
        return StandardTimeUnits.DAYS

    def rangeExpr(self, reverse: bool = False) -> Column:
        # convert date to number of days since the epoch
        expr = sfn.datediff(
            sfn.col(self.parsed_ts_field),
            sfn.lit(EPOCH_START_DATE).cast("date"),
        )
        return _reverse_or_not(expr, reverse)


class SubMicrosecondPrecisionTimestampIndex(ParsedTSIndex):
    """
    Timeseries index class for timestamps with sub-microsecond precision
    parsed from a string column. Internally, the timestamps are stored as
    doubles (fractional seconds since epoch), as well as the original string
    and a micro-second precision (standard) timestamp field.
    """

    def __init__(
        self,
        ts_struct: StructField,
        double_ts_field: str,
        secondary_parsed_ts_field: str,
        src_str_field: str,
        num_precision_digits: int = 9,
    ) -> None:
        """
        :param ts_struct: The StructField for the TSIndex column
        :param double_ts_field: The name of the double-precision timestamp column
        :param secondary_parsed_ts_field: The name of the parsed timestamp column
        :param src_str_field: The name of the source string column
        :param num_precision_digits: The number of digits that make up the precision of
        the timestamp. Ie. 9 for nanoseconds (default), 12 for picoseconds, etc.
        You will receive a warning if this value is 6 or less, as this is the precision
        of the standard timestamp type.
        """
        super().__init__(ts_struct, double_ts_field, src_str_field)
        # validate the double timestamp column
        double_ts_type = self.schema[double_ts_field].dataType
        if not isinstance(double_ts_type, DoubleType):
            raise TypeError(
                "The double_ts_col must be of DoubleType, "
                f"but the given double_ts_col {double_ts_field} "
                f"has type {double_ts_type}"
            )
        self._double_ts_field = double_ts_field
        # validate the number of precision digits
        if num_precision_digits <= 6:
            warnings.warn(
                "SubMicrosecondPrecisionTimestampIndex has a num_precision_digits "
                f"of {num_precision_digits} which is within the range of the "
                "standard timestamp precision of 6 digits (microseconds). "
                "Consider using a ParsedTimestampIndex instead."
            )
        self._num_precision_digits = num_precision_digits
        # validate the parsed column as a timestamp column
        parsed_ts_type = self.schema[secondary_parsed_ts_field].dataType
        if not isinstance(parsed_ts_type, TimestampType):
            raise TypeError(
                "parsed_ts_col field must be of TimestampType, "
                f"but the given parsed_ts_col {secondary_parsed_ts_field} "
                f"has type {parsed_ts_type}"
            )
        self.secondary_parsed_ts_field = secondary_parsed_ts_field

    @property
    def double_ts_field(self):
        return self.fieldPath(self._double_ts_field)

    @property
    def num_precision_digits(self):
        return self._num_precision_digits

    @property
    def unit(self) -> Optional[TimeUnit]:
        return StandardTimeUnits.SECONDS

    def rangeExpr(self, reverse: bool = False) -> Column:
        # just use the order by expression, since this is the same
        return _reverse_or_not(sfn.col(self.double_ts_field), reverse)


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
    def rowsBetweenWindow(
        self, start: int, end: int, reverse: bool = False
    ) -> WindowSpec:
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
        return self.rowsBetweenWindow(Window.unboundedPreceding, 0 if inclusive else -1)

    def allAfterWindow(self, inclusive: bool = True) -> WindowSpec:
        """
        build a window that includes all rows after the current row

        :param inclusive: if True, include the current row,
        otherwise begin with the first row after the current row

        :return: a WindowSpec object
        """
        return self.rowsBetweenWindow(0 if inclusive else 1, Window.unboundedFollowing)

    @abstractmethod
    def rangeBetweenWindow(
        self, start: int, end: int, reverse: bool = False
    ) -> WindowSpec:
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
        # must have comparable TSIndex types
        if not self.ts_idx.is_comparable(o.ts_idx):
            return False
        # must have the same series IDs
        if self.series_ids != o.series_ids:
            return False
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ts_idx={self.ts_idx}, series_ids={self.series_ids})"

    @classmethod
    def fromDFSchema(cls,
                     df_schema: StructType,
                     ts_col: str,
                     series_ids: Optional[Collection[str]] = None) -> "TSSchema":
        # construct a TSIndex for the given ts_col
        ts_idx = SimpleTSIndex.fromTSCol(df_schema[ts_col])
        return cls(ts_idx, series_ids)

    @classmethod
    def fromParsedTimestamp(cls,
                            df_schema: StructType,
                            ts_col: str,
                            parsed_field: str,
                            src_str_field: str,
                            series_ids: Optional[Collection[str]] = None,
                            secondary_parsed_field: Optional[str] = None) -> "TSSchema":
        ts_idx_schema = df_schema[ts_col].dataType
        assert isinstance(ts_idx_schema, StructType), \
            f"Expected a StructType for ts_col {ts_col}, but got {ts_idx_schema}"
        # construct the TSIndex
        parsed_type = ts_idx_schema[parsed_field].dataType
        if isinstance(parsed_type, DoubleType):
            ts_idx = SubMicrosecondPrecisionTimestampIndex(
                df_schema[ts_col],
                parsed_field,
                secondary_parsed_field,
                src_str_field,
            )
        elif isinstance(parsed_type, TimestampType):
            ts_idx = ParsedTimestampIndex(
                df_schema[ts_col],
                parsed_field,
                src_str_field,
            )
        elif isinstance(parsed_type, DateType):
            ts_idx = ParsedDateIndex(
                df_schema[ts_col],
                parsed_field,
                src_str_field,
            )
        else:
            raise TypeError(
                f"Expected a DoubleType, TimestampType or DateType "
                f"for parsed_field {parsed_field}, but got {parsed_type}"
            )
        # construct the TSSchema
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
            assert (
                sid in df_schema.fieldNames()
            ), f"Series ID {sid} does not exist in the given DataFrame"

    def find_observational_columns(self, df_schema: StructType) -> list[str]:
        return list(set(df_schema.fieldNames()) - set(self.structural_columns))

    @classmethod
    def __is_metric_col_type(cls, col: StructField) -> bool:
        return isinstance(col.dataType, NumericType) or isinstance(
            col.dataType, BooleanType
        )

    def find_metric_columns(self, df_schema: StructType) -> list[str]:
        return [
            col.name
            for col in df_schema.fields
            if self.__is_metric_col_type(col)
            and (col.name in self.find_observational_columns(df_schema))
        ]

    def baseWindow(self, reverse: bool = False) -> WindowSpec:
        # The index will determine the appropriate sort order
        w = Window().orderBy(self.ts_idx.orderByExpr(reverse))

        # and partitioned by any series IDs
        if self.series_ids:
            w = w.partitionBy([sfn.col(sid) for sid in self.series_ids])
        return w

    def rowsBetweenWindow(
        self, start: int, end: int, reverse: bool = False
    ) -> WindowSpec:
        return self.baseWindow(reverse=reverse).rowsBetween(start, end)

    def rangeBetweenWindow(
        self, start: int, end: int, reverse: bool = False
    ) -> WindowSpec:
        return (
            self.baseWindow(reverse=reverse)
            .orderBy(self.ts_idx.rangeExpr(reverse=reverse))
            .rangeBetween(start, end)
        )
