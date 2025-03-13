from dataclasses import dataclass
from datetime import datetime
from typing import (
    Optional,
    Union,
    cast,
    Any,
    Protocol,
    Type,
)

from numpy import integer, floating
from pandas import Series, Timestamp

from tempo.intervals.core.types import IntervalBoundary
from tempo.intervals.datetime.utils import infer_datetime_format


# Define a protocol for the converter functions
class ToTimestampProtocol(Protocol):
    def __call__(self, value: Optional[Any]) -> Optional[Timestamp]: ...


class FromTimestampProtocol(Protocol):
    def __call__(self, timestamp: Optional[Timestamp]) -> Optional[Any]: ...


@dataclass
class BoundaryConverter:
    """
    Handles conversion between user-provided boundary types and internal pd.Timestamp.
    Maintains original format for converting back to user format.
    """

    to_timestamp: ToTimestampProtocol
    from_timestamp: FromTimestampProtocol
    original_type: Type[Any]
    original_format: Optional[str] = None  # Store original string format if applicable

    @classmethod
    def for_type(cls, sample_value: IntervalBoundary) -> "BoundaryConverter":
        """Factory method to create appropriate converter based on input type"""

        def _check_negative_timestamp(
                timestamp: Optional[Timestamp],
        ) -> Optional[Timestamp]:
            if timestamp is not None and timestamp.value < 0:
                raise ValueError("Timestamps cannot be negative.")
            return timestamp

        if isinstance(sample_value, str):
            # Determine the format from the sample value
            format_str = infer_datetime_format(sample_value)

            def str_to_timestamp(value: Optional[str]) -> Optional[Timestamp]:
                if value is None:
                    return None
                return _check_negative_timestamp(Timestamp(value))

            def timestamp_to_str(timestamp: Optional[Timestamp]) -> Optional[str]:
                if timestamp is None:
                    return None
                return timestamp.strftime(format_str)

            return cls(
                to_timestamp=str_to_timestamp,
                from_timestamp=cast(FromTimestampProtocol, timestamp_to_str),
                original_type=str,
                original_format=format_str,  # Store just the format string
            )
        elif isinstance(sample_value, (int, float, integer, floating)):
            # Convert numpy types to Python native types
            original_type = int if isinstance(sample_value, (int, integer)) else float

            def numeric_to_timestamp(
                    value: Optional[Union[int, float]]
            ) -> Optional[Timestamp]:
                if value is None:
                    return None
                value_converted = (
                    int(value) if isinstance(value, (int, integer)) else float(value)
                )
                return _check_negative_timestamp(Timestamp(value_converted, unit="s"))

            def timestamp_to_numeric(
                    timestamp: Optional[Timestamp],
            ) -> Optional[Union[int, float]]:
                if timestamp is None:
                    return None
                return original_type(timestamp.timestamp())

            return cls(
                to_timestamp=cast(ToTimestampProtocol, numeric_to_timestamp),
                from_timestamp=cast(FromTimestampProtocol, timestamp_to_numeric),
                original_type=original_type,
            )

        elif sample_value is None:

            def none_to_timestamp(value: Optional[None]) -> Optional[Timestamp]:
                return None

            def timestamp_to_none(timestamp: Optional[Timestamp]) -> Optional[None]:
                return None

            return cls(
                to_timestamp=none_to_timestamp,
                from_timestamp=cast(FromTimestampProtocol, timestamp_to_none),
                original_type=type(None),
            )
        # Handle Timestamp first because pandas.Timestamp is a subclass of datetime
        elif isinstance(sample_value, Timestamp):

            def timestamp_to_timestamp(
                    value: Optional[Timestamp],
            ) -> Optional[Timestamp]:
                if value is None:
                    return None
                return _check_negative_timestamp(value)

            def timestamp_identity(
                    timestamp: Optional[Timestamp],
            ) -> Optional[Timestamp]:
                return timestamp

            return cls(
                to_timestamp=timestamp_to_timestamp,
                from_timestamp=cast(FromTimestampProtocol, timestamp_identity),
                original_type=Timestamp,
            )
        elif isinstance(sample_value, datetime):

            def datetime_to_timestamp(value: Optional[datetime]) -> Optional[Timestamp]:
                if value is None:
                    return None
                return _check_negative_timestamp(Timestamp(value))

            def timestamp_to_datetime(
                    timestamp: Optional[Timestamp],
            ) -> Optional[datetime]:
                if timestamp is None:
                    return None
                return timestamp.to_pydatetime()

            return cls(
                to_timestamp=cast(ToTimestampProtocol, datetime_to_timestamp),
                from_timestamp=cast(FromTimestampProtocol, timestamp_to_datetime),
                original_type=datetime,
            )
        else:
            raise ValueError(f"Unsupported boundary type: {type(sample_value)}")


@dataclass
class BoundaryValue:
    """
    Wrapper class that maintains both internal timestamp and original format.
    """

    _timestamp: Optional[Timestamp]
    _converter: BoundaryConverter

    @classmethod
    def from_user_value(cls, value: IntervalBoundary) -> "BoundaryValue":
        converter = BoundaryConverter.for_type(value)
        timestamp = converter.to_timestamp(value)
        return cls(_timestamp=timestamp, _converter=converter)

    @property
    def internal_value(self) -> Optional[Timestamp]:
        """Get the internal timestamp representation"""
        return self._timestamp

    def to_user_value(self) -> IntervalBoundary:
        """Convert back to the original user format"""
        return self._converter.from_timestamp(self._timestamp)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoundaryValue):
            return NotImplemented
        # Handle None cases
        if self._timestamp is None and other._timestamp is None:
            return True
        if self._timestamp is None or other._timestamp is None:
            return False
        return self._timestamp == other._timestamp

    def __lt__(self, other: "BoundaryValue") -> bool:
        # Handle None cases
        if self._timestamp is None:
            return False  # None is considered less than any value
        if other._timestamp is None:
            return False  # Nothing is less than None
        return self._timestamp < other._timestamp

    def __le__(self, other: "BoundaryValue") -> bool:
        # Handle None cases
        if self._timestamp is None and other._timestamp is None:
            return True  # None == None
        if self._timestamp is None:
            return False  # None is not <= timestamp
        if other._timestamp is None:
            return False  # timestamp is not <= None
        return self._timestamp <= other._timestamp

    def __gt__(self, other: "BoundaryValue") -> bool:
        # Handle None cases
        if self._timestamp is None:
            return False  # None is not greater than anything
        if other._timestamp is None:
            return False  # Nothing is greater than None
        return self._timestamp > other._timestamp

    def __ge__(self, other: "BoundaryValue") -> bool:
        # Handle None cases
        if self._timestamp is None and other._timestamp is None:
            return True  # None == None
        if self._timestamp is None:
            return False  # None is not >= timestamp
        if other._timestamp is None:
            return False  # timestamp is not >= None
        return self._timestamp >= other._timestamp

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, BoundaryValue):
            return NotImplemented
        return not self.__eq__(other)


@dataclass
class IntervalBoundaries:
    _start: BoundaryValue
    _end: BoundaryValue

    @classmethod
    def create(
            cls,
            start: Union[IntervalBoundary, BoundaryValue],
            end: Union[IntervalBoundary, BoundaryValue],
    ) -> "IntervalBoundaries":
        # Convert only if not already a BoundaryValue
        start_boundary = (
            start
            if isinstance(start, BoundaryValue)
            else BoundaryValue.from_user_value(start)
        )
        end_boundary = (
            end
            if isinstance(end, BoundaryValue)
            else BoundaryValue.from_user_value(end)
        )

        return cls(_start=start_boundary, _end=end_boundary)

    @property
    def start(self) -> IntervalBoundary:
        """Get start boundary in user format"""
        return self._start.to_user_value()

    @property
    def end(self) -> IntervalBoundary:
        """Get end boundary in user format"""
        return self._end.to_user_value()

    @property
    def internal_start(self) -> BoundaryValue:
        """Get internal timestamp representation of start"""
        return self._start

    @property
    def internal_end(self) -> BoundaryValue:
        """Get internal timestamp representation of end"""
        return self._end


# This class handles the mapping between user-provided field names and our internal structure
class _BoundaryAccessor:
    """
    Manages access to interval boundaries using user-provided field names.

    This class serves as an adapter between the user's data structure (which uses
    field names to access values) and our internal representation (which uses
    proper objects). It maintains the flexibility of user-defined fields while
    providing a clean interface for the rest of our code.
    """

    def __init__(self, start_field: str, end_field: str):
        self.start_field = start_field
        self.end_field = end_field

    def get_boundaries(self, data: Series) -> IntervalBoundaries:
        """Extract boundary values from data using configured field names"""
        return IntervalBoundaries.create(
            start=data[self.start_field],
            end=data[self.end_field],
        )

    def set_boundaries(self, data: Series, boundaries: IntervalBoundaries) -> Series:
        """Update data with boundary values using configured field names"""
        data = data.copy()
        data[self.start_field] = boundaries.start
        data[self.end_field] = boundaries.end
        return data
