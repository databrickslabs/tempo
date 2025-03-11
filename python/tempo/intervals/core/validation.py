from dataclasses import dataclass
from typing import Optional, Sequence

from pandas import Series

from tempo.intervals.core.exceptions import EmptyIntervalError, InvalidDataTypeError, InvalidMetricColumnError


@dataclass
class ValidationResult:
    is_valid: bool
    message: Optional[str] = None


class IntervalValidator:
    """Validates interval data and properties"""

    @staticmethod
    def validate_data(data: Series) -> ValidationResult:

        if not isinstance(data, Series):
            raise InvalidDataTypeError("Expected data to be a Pandas Series")

        if data.empty:
            raise EmptyIntervalError("Data must not be empty")

        return ValidationResult(is_valid=True)

    @staticmethod
    def _validate_columns(columns: Optional[Sequence[str]], column_type: str) -> ValidationResult:
        if columns is None:
            return ValidationResult(is_valid=True)

        if not (isinstance(columns, Sequence) and not isinstance(columns, str)):
            raise InvalidMetricColumnError(f"{column_type} must be a sequence")

        if not all(isinstance(col, str) for col in columns):
            raise InvalidMetricColumnError(f"All {column_type} must be of type str")

        if len(set(columns)) != len(columns):
            raise InvalidMetricColumnError(f"Duplicate {column_type} found")

        return ValidationResult(is_valid=True)

    @staticmethod
    def validate_series_id_columns(series_ids: Optional[Sequence[str]]) -> ValidationResult:
        return IntervalValidator._validate_columns(series_ids, "series ID columns")

    @staticmethod
    def validate_metric_columns(metric_columns: Optional[Sequence[str]]) -> ValidationResult:
        return IntervalValidator._validate_columns(metric_columns, "metric columns")
