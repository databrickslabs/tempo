class IntervalValidationError(Exception):
    """Base exception for interval validation errors"""
    pass


class EmptyIntervalError(IntervalValidationError):
    """Raised when interval data is empty"""
    pass


class InvalidDataTypeError(IntervalValidationError):
    """Raised when data is not of expected type"""
    pass


class InvalidTimestampError(IntervalValidationError):
    """Raised when timestamps are invalid"""
    pass


class InvalidSeriesColumnError(IntervalValidationError):
    """Raised when metric columns are invalid"""
    pass


class InvalidMetricColumnError(IntervalValidationError):
    """Raised when metric columns are invalid"""
    pass


class ErrorMessages:
    """Centralized error message definitions for consistent error handling"""
    INVALID_SERIES_IDS = "series_ids must be an Iterable or comma separated string of column names, got {}"
    NO_RESOLVER = "No resolver registered for overlap type: {}"
    RESOLUTION_FAILED = "Resolution failed for {}: {}"
    RESOLVER_REGISTERED = "Resolver already registered for {}"
    INTERVAL_INDICES = "Expected indices of interval elements to be equivalent"
    METRIC_COLUMNS_LENGTH = "Metric columns must have the same length"
