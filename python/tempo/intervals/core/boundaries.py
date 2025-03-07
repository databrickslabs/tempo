from dataclasses import dataclass
from datetime import datetime
from typing import Generic, Callable, Optional, Union

from numpy import integer, floating
from pandas import Series, Timestamp

from tempo.intervals.core.types import T, IntervalBoundary
from tempo.intervals.datetime.utils import infer_datetime_format


@dataclass
class BoundaryConverter(Generic[T]):
    """
    Handles conversion between user-provided boundary types and internal pd.Timestamp.
    Maintains original format for converting back to user format.
    """
    to_timestamp: Callable[[Optional[T]], Optional[Timestamp]]
    from_timestamp: Callable[[Optional[Timestamp]], Optional[T]]
    original_type: type
    original_format: Optional[str] = None  # Store original string format if applicable

    @classmethod
    def for_type(cls, sample_value: IntervalBoundary) -> 'BoundaryConverter':
        """Factory method to create appropriate converter based on input type"""
        if isinstance(sample_value, str):
            # Determine the format from the sample value
            format_str = infer_datetime_format(sample_value)

            return cls(
                to_timestamp=lambda x: Timestamp(x),
                # Use the inferred format to convert back
                from_timestamp=lambda x: x.strftime(format_str),
                original_type=str,
                original_format=format_str  # Store just the format string
            )
        elif isinstance(sample_value, (int, float, integer, floating)):
            # Convert numpy types to Python native types
            original_type = int if isinstance(sample_value, (int, integer)) else float
            return cls(
                to_timestamp=lambda x: Timestamp(int(x) if isinstance(x, (int, integer)) else float(x), unit='s'),
                from_timestamp=lambda x: original_type(x.timestamp()),
                original_type=original_type
            )

        elif sample_value is None:
            return cls(
                to_timestamp=lambda x: None,
                from_timestamp=lambda x: None,
                original_type=type(sample_value)
            )
        # Handle Timestamp first because pandas.Timestamp is a subclass of datetime
        elif isinstance(sample_value, Timestamp):
            return cls(
                to_timestamp=lambda x: x,
                from_timestamp=lambda x: x,
                original_type=Timestamp
            )
        elif isinstance(sample_value, datetime):
            return cls(
                to_timestamp=lambda x: Timestamp(x),
                from_timestamp=lambda x: x.to_pydatetime(),
                original_type=datetime
            )
        else:
            raise ValueError(f"Unsupported boundary type: {type(sample_value)}")

@dataclass
class BoundaryValue:
    """
    Wrapper class that maintains both internal timestamp and original format.
    """
    _timestamp: Timestamp
    _converter: BoundaryConverter

    @classmethod
    def from_user_value(cls, value: IntervalBoundary) -> "BoundaryValue":
        converter = BoundaryConverter.for_type(value)
        return cls(
            _timestamp=converter.to_timestamp(value),
            _converter=converter
        )

    @property
    def internal_value(self) -> Timestamp:
        """Get the internal timestamp representation"""
        return self._timestamp

    def to_user_value(self) -> IntervalBoundary:
        """Convert back to the original user format"""
        return self._converter.from_timestamp(self._timestamp)

    def __eq__(self, other: 'BoundaryValue') -> bool:
        return self._timestamp == other._timestamp

    def __lt__(self, other: 'BoundaryValue') -> bool:
        return self._timestamp < other._timestamp

    def __le__(self, other: 'BoundaryValue') -> bool:
        return self._timestamp <= other._timestamp

    def __gt__(self, other: 'BoundaryValue') -> bool:
        return self._timestamp > other._timestamp

    def __ge__(self, other: 'BoundaryValue') -> bool:
        return self._timestamp >= other._timestamp

    def __ne__(self, other: 'BoundaryValue') -> bool:
        return self._timestamp != other._timestamp


@dataclass
class IntervalBoundaries:
    _start: BoundaryValue
    _end: BoundaryValue

    @classmethod
    def create(
            cls,
            start: Union[IntervalBoundary, BoundaryValue],
            end: Union[IntervalBoundary, BoundaryValue],
    ) -> 'IntervalBoundaries':
        # Convert only if not already a BoundaryValue
        start_boundary = start if isinstance(start, BoundaryValue) else BoundaryValue.from_user_value(start)
        end_boundary = end if isinstance(end, BoundaryValue) else BoundaryValue.from_user_value(end)

        return cls(
            _start=start_boundary,
            _end=end_boundary
        )

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
