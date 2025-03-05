from typing import Optional, Sequence, Union

from pandas import Series

from tempo.intervals.core.boundaries import _BoundaryAccessor, BoundaryValue, IntervalBoundaries
from tempo.intervals.core.exceptions import InvalidDataTypeError, EmptyIntervalError, InvalidMetricColumnError
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

        if start_field is None or end_field is None:
            raise ValueError("start_field and end_field cannot be None")

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
        # self._validate_initialization(data, start_ts, end_ts, series_ids, metric_columns)

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
            series_ids: Optional[Sequence[str]],
            metric_columns: Optional[Sequence[str]]
    ) -> None:
        """Validates all components during initialization"""
        self._validate_data(data)
        self._validate_series_ids(series_ids)
        self._validate_metric_columns(metric_columns)

    @property
    def start(self) -> BoundaryValue:
        """Returns the start timestamp of the interval"""
        return self._boundaries.internal_start

    @property
    def user_start(self):
        return self._boundaries.start

    @property
    def start_field(self):
        return self.boundary_accessor.start_field

    @property
    def end(self) -> BoundaryValue:
        """Returns the end timestamp of the interval"""
        return self._boundaries.internal_end

    @property
    def user_end(self):
        return self._boundaries.end

    @property
    def end_field(self) -> str:

        return self.boundary_accessor.end_field

    @property
    def boundaries(self) -> tuple[BoundaryValue, BoundaryValue]:
        """Returns the start and end timestamps as a Series"""
        return self._boundaries.internal_start, self._boundaries.internal_end

    @staticmethod
    def _validate_data(data: Series) -> None:
        """Validates the basic data structure"""
        if not isinstance(data, Series):
            raise InvalidDataTypeError("Data must be a pandas Series")
        if data.empty:
            raise EmptyIntervalError("Data cannot be empty")

    @staticmethod
    def _validate_series_ids(series_ids: Optional[Sequence[str]]) -> None:
        """Validates series identifier columns"""
        if series_ids is not None:
            if not isinstance(series_ids, Sequence) or isinstance(series_ids, str):
                raise InvalidMetricColumnError("series_ids must be a sequence")
            if not all(isinstance(col, str) for col in series_ids):
                raise InvalidMetricColumnError("All series_ids must be strings")

    @staticmethod
    def _validate_metric_columns(metric_columns: Optional[Sequence[str]]) -> None:
        """Validates metric column names"""
        if metric_columns is not None:
            if not isinstance(metric_columns, Sequence) or isinstance(metric_columns, str):
                raise InvalidMetricColumnError("metric_columns must be a sequence")
            if not all(isinstance(col, str) for col in metric_columns):
                raise InvalidMetricColumnError("All metric_columns must be strings")

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
        return self.start <= other.start <= self.end

    def overlaps_with(self, other: 'Interval') -> bool:
        """Checks if this interval overlaps with another interval"""
        return self.start < other.end and self.end > other.start

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
            "Metric columns"
        )

    def validate_series_alignment(self, other: 'Interval') -> ValidationResult:
        """Validates that series identifiers align between intervals"""
        return self._validate_column_alignment(
            self.series_fields,
            other.series_fields,
            "Series IDs"
        )

    @staticmethod
    def _validate_column_alignment(
            self_columns: Sequence[str],
            other_columns: Sequence[str],
            column_type: str
    ) -> ValidationResult:
        """Generic validator for column alignment between intervals"""
        if len(self_columns) != len(other_columns):
            raise InvalidMetricColumnError(
                f"{column_type} count mismatch: {len(self_columns)} vs {len(other_columns)}"
            )
        if set(self_columns) != set(other_columns):
            raise InvalidMetricColumnError(
                f"{column_type} don't match: {self_columns} vs {other_columns}"
            )
        return ValidationResult(is_valid=True)
