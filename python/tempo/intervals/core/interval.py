from typing import Optional, Sequence, Union

from pandas import Series

from tempo.intervals.core.boundaries import _BoundaryAccessor, BoundaryValue, IntervalBoundaries
from tempo.intervals.core.exceptions import InvalidDataTypeError, EmptyIntervalError, InvalidMetricColumnError, \
    InvalidSeriesColumnError
from tempo.intervals.core.types import IntervalBoundary
from tempo.intervals.core.validation import ValidationResult
from tempo.intervals.metrics.merger import DefaultMetricMerger
from tempo.intervals.metrics.operations import MetricMergeConfig


class Interval:
    """
    Represents a single point in a time series with start/end boundaries and associated metrics.

    An Interval is the fundamental unit of time series data in this system, containing:
    - Time boundaries (start and end timestamps)
    - Series identifiers (dimensional values that identify this series)
    - Metric values (measurements or observations for this time point)

    The class provides operations for:
    - Validating interval data and structure
    - Comparing and manipulating time boundaries
    - Managing metric data
    - Checking relationships with other intervals
    """

    @classmethod
    def create(
            cls,
            data: Series,
            start_field: str,
            end_field: str,
            series_fields: Optional[Sequence[str]] = None,
            metric_fields: Optional[Sequence[str]] = None,
    ) -> 'Interval':
        """
        Creates a new Interval instance from a pandas Series and field mappings.

        This is the preferred way to construct an Interval, as it handles all the
        internal setup of field mappings and data validation.
        """
        # Create the accessor internally - users never need to know about it
        boundary_accessor = _BoundaryAccessor(start_field, end_field)

        return cls(
            data=data,
            boundary_accessor=boundary_accessor,
            series_fields=series_fields,
            metric_fields=metric_fields
        )

    def __init__(
            self,
            data: Series,
            boundary_accessor: _BoundaryAccessor,
            series_fields: Optional[Sequence[str]] = None,
            metric_fields: Optional[Sequence[str]] = None,
    ):
        """
        Initialize an interval with its data and metadata.

        Args:
            data: Series containing the interval's data
            boundary_accessor: Accessor for interval boundaries
            series_fields: Names of columns that identify the series
            metric_fields: Names of columns containing metric values
        """
        self._validate_initialization(
            data,
            boundary_accessor,
            series_fields,
            metric_fields,
        )

        self.data = data
        self.boundary_accessor = boundary_accessor
        self.series_fields = series_fields or []
        self.metric_fields = metric_fields or []

        # Internal cached representation for clean access
        self._boundaries = self.boundary_accessor.get_boundaries(data)

    # Validation Methods
    # -----------------

    def _validate_initialization(
            self,
            data: Series,
            boundary_accessor: _BoundaryAccessor,
            series_fields: Optional[Sequence[str]],
            metric_fields: Optional[Sequence[str]]
    ) -> None:
        """Validates all components during initialization"""
        # Validate data type and emptiness first
        self._validate_data(data)
        # Then validate other aspects
        self._validate_not_point_in_time(data, boundary_accessor)
        self._validate_series_fields(series_fields)
        self._validate_metric_columns(metric_fields)

    @property
    def _start(self) -> BoundaryValue:
        """Returns the start timestamp of the interval"""
        return self._boundaries.internal_start

    @property
    def start(self):
        return self._boundaries.start

    @property
    def start_field(self):
        return self.boundary_accessor.start_field

    @property
    def _end(self) -> BoundaryValue:
        """Returns the end timestamp of the interval"""
        return self._boundaries.internal_end

    @property
    def end(self):
        return self._boundaries.end

    @property
    def end_field(self) -> str:

        return self.boundary_accessor.end_field

    @property
    def boundaries(self) -> tuple:
        """Returns the start and end timestamps as a Series"""
        return self._boundaries.start, self._boundaries.end

    @staticmethod
    def _validate_not_point_in_time(data: Series, boundary_accessor: _BoundaryAccessor) -> None:
        """Validates that the interval is not a point in time"""
        if data[boundary_accessor.start_field] == data[boundary_accessor.end_field]:
            raise InvalidDataTypeError(
                "Start and end field values cannot be the same. Point-in-Time Intervals are not supported."
            )

    @staticmethod
    def _validate_data(data: Series) -> None:
        """Validates the basic data structure"""
        if not isinstance(data, Series):
            raise InvalidDataTypeError("Data must be a pandas Series")
        if data.empty:
            raise EmptyIntervalError("Data cannot be empty")

    @staticmethod
    def _validate_series_fields(series_fields: Optional[Sequence[str]]) -> None:
        """Validates series identifier columns"""
        if series_fields is not None:
            if not isinstance(series_fields, Sequence) or isinstance(series_fields, str):
                raise InvalidSeriesColumnError("series_fields must be a sequence")
            if not all(isinstance(col, str) for col in series_fields):
                raise InvalidSeriesColumnError("All series_fields must be strings")

    @staticmethod
    def _validate_metric_columns(metric_fields: Optional[Sequence[str]]) -> None:
        """Validates metric column names"""
        if metric_fields is not None:
            if not isinstance(metric_fields, Sequence) or isinstance(metric_fields, str):
                raise InvalidMetricColumnError("metric_fields must be a sequence")
            if not all(isinstance(col, str) for col in metric_fields):
                raise InvalidMetricColumnError("All metric_fields must be strings")

    # Time Operations
    # --------------

    def update_start(self, new_start: Union[IntervalBoundary, BoundaryValue]) -> 'Interval':
        """
        Creates a new interval with updated start time while maintaining
        the user's field mapping structure
        """
        new_boundaries = IntervalBoundaries.create(start=new_start, end=self.end)
        new_data = self.boundary_accessor.set_boundaries(self.data, new_boundaries)

        return Interval(
            new_data,
            self.boundary_accessor,
            self.series_fields,
            self.metric_fields
        )

    def update_end(self, new_end: Union[IntervalBoundary, BoundaryValue]) -> 'Interval':
        """
        Creates a new interval with updated end time while maintaining
        the user's field mapping structure
        """
        new_boundaries = IntervalBoundaries.create(start=self.start, end=new_end)
        new_data = self.boundary_accessor.set_boundaries(self.data, new_boundaries)

        return Interval(
            new_data,
            self.boundary_accessor,
            self.series_fields,
            self.metric_fields
        )

    # Interval Relationships
    # ---------------------

    def contains(self, other: 'Interval') -> bool:
        """Checks if this interval fully contains another interval"""
        return self._start <= other._start <= self._end

    def overlaps_with(self, other: 'Interval') -> bool:
        """Checks if this interval overlaps with another interval"""
        return self._start < other._end and self._end > other._start

    # Metric Operations
    # ----------------

    def merge_metrics(self, other: 'Interval', merge_config: MetricMergeConfig = None) -> Series:
        """Combines metrics between intervals according to merging strategy"""
        merger = DefaultMetricMerger(merge_config)
        return merger.merge(self, other)

    def validate_metrics_alignment(self, other: 'Interval') -> ValidationResult:
        """Validates that metric columns align between intervals"""
        return self._validate_column_alignment(
            self.metric_fields,
            other.metric_fields,
            "metric_fields"
        )

    def validate_series_alignment(self, other: 'Interval') -> ValidationResult:
        """Validates that series identifiers align between intervals"""
        return self._validate_column_alignment(
            self.series_fields,
            other.series_fields,
            "series_fields"
        )

    @staticmethod
    def _validate_column_alignment(
            self_columns: Sequence[str],
            other_columns: Sequence[str],
            column_type: str
    ) -> ValidationResult:
        """Generic validator for column alignment between intervals"""
        error_message = f"{column_type} don't match: {self_columns} vs {other_columns}"
        if set(self_columns) != set(other_columns):
            if column_type == "metric_fields":
                raise InvalidMetricColumnError(
                    error_message
                )
            elif column_type == "series_fields":
                raise InvalidSeriesColumnError(
                    error_message
                )
            else:
                raise ValueError(f"Unknown column type: {column_type}")
        return ValidationResult(is_valid=True)
