from abc import ABC, abstractmethod
from typing import Union, Callable, TypeVar

from numpy import integer, floating
from pandas import notna, isna, Series

from tempo.intervals.core.types import MetricValue

# Type variables for better typing
T = TypeVar("T")
ScalarFunc = Callable[[MetricValue, MetricValue], MetricValue]
SeriesScalarFunc = Callable[[Series, MetricValue, bool], Series]
SeriesSeriesFunc = Callable[[Series, Series], Series]


class MetricMergeStrategy(ABC):
    """Abstract base class for implementing metric merge strategies"""

    @abstractmethod
    def merge(
        self,
        value1: Union[MetricValue, Series],
        value2: Union[MetricValue, Series],
    ) -> Union[MetricValue, Series]:
        """Merge two metric values according to the strategy"""
        pass

    def validate(
        self,
        value1: Union[MetricValue, Series],
        value2: Union[MetricValue, Series],
    ) -> None:
        """Validate that values are of the appropriate type for the strategy"""
        # For Series inputs
        if isinstance(value1, Series):
            for val in value1:
                if notna(val) and not isinstance(val, (int, float, integer, floating)):
                    raise ValueError(
                        f"{self.__class__.__name__} requires numeric values"
                    )

        elif notna(value1) and not isinstance(value1, (int, float, integer, floating)):
            raise ValueError(f"{self.__class__.__name__} requires numeric values")

        if isinstance(value2, Series):
            for val in value2:
                if notna(val) and not isinstance(val, (int, float, integer, floating)):
                    raise ValueError(
                        f"{self.__class__.__name__} requires numeric values"
                    )

        elif notna(value2) and not isinstance(value2, (int, float, integer, floating)):
            raise ValueError(f"{self.__class__.__name__} requires numeric values")

    def _handle_scalar_case(
        self, value1: MetricValue, value2: MetricValue, scalar_strategy_func: ScalarFunc
    ) -> MetricValue:
        """
        Generic handler for scalar-scalar merge cases.

        Args:
            value1: First scalar value
            value2: Second scalar value
            scalar_strategy_func: Function that implements the specific strategy logic for scalars

        Returns:
            Merged scalar value according to the strategy
        """
        return scalar_strategy_func(value1, value2)

    def _handle_series_scalar_case(
        self,
        series_value: Series,
        scalar_value: MetricValue,
        series_is_first: bool,
        series_scalar_strategy_func: SeriesScalarFunc,
    ) -> Series:
        """
        Generic handler for series-scalar merge cases.

        Args:
            series_value: The Series value
            scalar_value: The scalar value
            series_is_first: True if series is value1, False if series is value2
            series_scalar_strategy_func: Function that implements the specific strategy logic

        Returns:
            Merged Series according to the strategy
        """
        return series_scalar_strategy_func(series_value, scalar_value, series_is_first)

    def _handle_series_series_case(
        self,
        series1: Series,
        series2: Series,
        series_series_strategy_func: SeriesSeriesFunc,
    ) -> Series:
        """
        Generic handler for series-series merge cases.

        Args:
            series1: First Series
            series2: Second Series
            series_series_strategy_func: Function that implements the specific strategy logic

        Returns:
            Merged Series according to the strategy
        """
        return series_series_strategy_func(series1, series2)


class KeepFirstStrategy(MetricMergeStrategy):
    """Keep the first non-null value"""

    def merge(
        self, value1: Union[MetricValue, Series], value2: Union[MetricValue, Series]
    ) -> Union[MetricValue, Series]:
        # 1. Handle scalar + scalar case
        if not isinstance(value1, Series) and not isinstance(value2, Series):
            return self._handle_scalar_case(value1, value2, self._keep_first_scalar)

        # 2. Handle Series + scalar case
        if isinstance(value1, Series) and not isinstance(value2, Series):
            return self._handle_series_scalar_case(
                value1, value2, True, self._keep_first_series_scalar
            )

        # 3. Handle scalar + Series case
        if not isinstance(value1, Series) and isinstance(value2, Series):
            return self._handle_series_scalar_case(
                value2, value1, False, self._keep_first_series_scalar
            )

        # 4. Handle Series + Series case (with explicit casting for type checker)
        if isinstance(value1, Series) and isinstance(value2, Series):
            return self._handle_series_series_case(
                value1, value2, self._keep_first_series_series
            )

        # This should never happen due to the conditions above, but makes the type checker happy
        raise ValueError("Unexpected input types")

    def _keep_first_scalar(
        self, value1: MetricValue, value2: MetricValue
    ) -> MetricValue:
        """Keep the first non-null scalar value"""
        return value1 if notna(value1) else value2

    def _keep_first_series_scalar(
        self, series: Series, scalar: MetricValue, series_is_first: bool
    ) -> Series:
        """Merge a Series and a scalar, keeping the first non-null value"""
        if series_is_first:
            # Series is first, scalar is second
            merged_result = series.copy()
            for i in range(len(merged_result)):
                if isna(merged_result.iloc[i]):
                    merged_result.iloc[i] = scalar
            return merged_result
        else:
            # Scalar is first, series is second
            if notna(scalar):
                # If scalar is not null, create a Series with that value
                return Series([scalar] * len(series))
            else:
                # If scalar is null, use the second series
                return series.copy()

    def _keep_first_series_series(self, series1: Series, series2: Series) -> Series:
        """Merge two Series, keeping the first non-null value at each position"""
        merged_result = series1.copy()
        for i in range(len(merged_result)):
            if isna(merged_result.iloc[i]):
                # Only replace if we're within range of series2
                if i < len(series2):
                    merged_result.iloc[i] = series2.iloc[i]
        return merged_result


class KeepLastStrategy(MetricMergeStrategy):
    """Keep the last non-null value"""

    def merge(
        self, value1: Union[MetricValue, Series], value2: Union[MetricValue, Series]
    ) -> Union[MetricValue, Series]:
        # 1. Handle scalar + scalar case
        if not isinstance(value1, Series) and not isinstance(value2, Series):
            return self._handle_scalar_case(value1, value2, self._keep_last_scalar)

        # 2. Handle Series + scalar case
        if isinstance(value1, Series) and not isinstance(value2, Series):
            return self._handle_series_scalar_case(
                value1, value2, True, self._keep_last_series_scalar
            )

        # 3. Handle scalar + Series case
        if not isinstance(value1, Series) and isinstance(value2, Series):
            return self._handle_series_scalar_case(
                value2, value1, False, self._keep_last_series_scalar
            )

        # 4. Handle Series + Series case (with explicit casting for type checker)
        if isinstance(value1, Series) and isinstance(value2, Series):
            return self._handle_series_series_case(
                value1, value2, self._keep_last_series_series
            )

        # This should never happen due to the conditions above, but makes the type checker happy
        raise ValueError("Unexpected input types")

    def _keep_last_scalar(
        self, value1: MetricValue, value2: MetricValue
    ) -> MetricValue:
        """Keep the last non-null scalar value"""
        return value2 if notna(value2) else value1

    def _keep_last_series_scalar(
        self, series: Series, scalar: MetricValue, series_is_first: bool
    ) -> Series:
        """Merge a Series and a scalar, keeping the last non-null value"""
        if series_is_first:
            # Series is first, scalar is second
            if notna(scalar):
                # If scalar is not null, use it for all positions
                return Series([scalar] * len(series))
            else:
                # If scalar is null, use the first series
                return series.copy()
        else:
            # Scalar is first, series is second
            merged_result = series.copy()
            for i in range(len(merged_result)):
                if isna(merged_result.iloc[i]):
                    merged_result.iloc[i] = scalar
            return merged_result

    def _keep_last_series_series(self, series1: Series, series2: Series) -> Series:
        """Merge two Series, keeping the last non-null value at each position"""
        merged_result = series2.copy()
        for i in range(len(merged_result)):
            if isna(merged_result.iloc[i]):
                # Only replace if we're within range of series1
                if i < len(series1):
                    merged_result.iloc[i] = series1.iloc[i]
        return merged_result


class SumStrategy(MetricMergeStrategy):
    """Sum the values, treating nulls as 0"""

    def merge(
        self,
        value1: Union[MetricValue, Series],
        value2: Union[MetricValue, Series],
    ) -> Union[MetricValue, Series]:
        # Validate inputs
        self.validate(value1, value2)

        # 1. Handle scalar + scalar case
        if not isinstance(value1, Series) and not isinstance(value2, Series):
            return self._handle_scalar_case(value1, value2, self._sum_scalars)

        # 2. Handle Series + scalar case
        if isinstance(value1, Series) and not isinstance(value2, Series):
            return self._handle_series_scalar_case(
                value1, value2, True, self._sum_series_scalar
            )

        # 3. Handle scalar + Series case
        if not isinstance(value1, Series) and isinstance(value2, Series):
            return self._handle_series_scalar_case(
                value2, value1, False, self._sum_series_scalar
            )

        # 4. Handle Series + Series case (with explicit casting for type checker)
        if isinstance(value1, Series) and isinstance(value2, Series):
            return self._handle_series_series_case(
                value1, value2, self._sum_series_series
            )

        # This should never happen due to the conditions above, but makes the type checker happy
        raise ValueError("Unexpected input types")

    def _sum_scalars(self, value1: MetricValue, value2: MetricValue) -> MetricValue:
        """Sum two scalar values, treating NaN as 0"""
        # Treat NaN as 0
        first_value = 0 if isna(value1) else value1
        second_value = 0 if isna(value2) else value2
        return first_value + second_value

    def _sum_series_scalar(
        self, series: Series, scalar: MetricValue, series_is_first: bool
    ) -> Series:
        """Sum a Series and a scalar value, treating NaN as 0"""
        # Create a copy of the Series and fill NA values with 0
        merged_result = series.copy().fillna(0)

        # Add the scalar value (treating NaN as 0)
        scalar_value = 0 if isna(scalar) else scalar
        for i in range(len(merged_result)):
            merged_result.iloc[i] += scalar_value

        return merged_result

    def _sum_series_series(self, series1: Series, series2: Series) -> Series:
        """Sum two Series, treating NaN as 0"""
        # Create result with maximum length
        max_len = max(len(series1), len(series2))
        merged_result = Series(index=range(max_len))

        # Calculate sum for each position
        for i in range(max_len):
            # Get value from first_series (default to 0 if out of range or null)
            first_value = 0
            if i < len(series1):
                if notna(series1.iloc[i]):
                    first_value = series1.iloc[i]

            # Get value from second_series (default to 0 if out of range or null)
            second_value = 0
            if i < len(series2):
                if notna(series2.iloc[i]):
                    second_value = series2.iloc[i]

            # Sum the values
            merged_result.iloc[i] = first_value + second_value

        return merged_result


class MaxStrategy(MetricMergeStrategy):
    """Take the maximum value"""

    def merge(
        self, value1: Union[MetricValue, Series], value2: Union[MetricValue, Series]
    ) -> Union[MetricValue, Series]:
        # Validate inputs
        self.validate(value1, value2)

        # 1. Handle scalar + scalar case
        if not isinstance(value1, Series) and not isinstance(value2, Series):
            return self._handle_scalar_case(value1, value2, self._max_scalars)

        # 2. Handle Series + scalar case
        if isinstance(value1, Series) and not isinstance(value2, Series):
            return self._handle_series_scalar_case(
                value1, value2, True, self._max_series_scalar
            )

        # 3. Handle scalar + Series case
        if not isinstance(value1, Series) and isinstance(value2, Series):
            return self._handle_series_scalar_case(
                value2, value1, False, self._max_series_scalar
            )

        # 4. Handle Series + Series case (with explicit casting for type checker)
        if isinstance(value1, Series) and isinstance(value2, Series):
            return self._handle_series_series_case(
                value1, value2, self._max_series_series
            )

        # This should never happen due to the conditions above, but makes the type checker happy
        raise ValueError("Unexpected input types")

    def _max_scalars(self, value1: MetricValue, value2: MetricValue) -> MetricValue:
        """Find the maximum of two scalar values, handling NaN"""
        # Use pandas Series.max() for proper NaN handling
        s = Series([value1, value2])
        return s.max(skipna=True)

    def _max_series_scalar(
        self, series: Series, scalar: MetricValue, series_is_first: bool
    ) -> Series:
        """Find the maximum between Series values and a scalar"""
        # Create result Series
        merged_result = Series(index=range(len(series)))

        # Compare each element with the scalar
        for i in range(len(merged_result)):
            current_value = series.iloc[i]
            if isna(current_value):
                merged_result.iloc[i] = scalar
            elif isna(scalar):
                merged_result.iloc[i] = current_value
            else:
                merged_result.iloc[i] = max(current_value, scalar)

        return merged_result

    def _max_series_series(self, series1: Series, series2: Series) -> Series:
        """Find the maximum values between two Series"""
        # Create a result Series with appropriate length
        max_len = max(len(series1), len(series2))
        merged_result = Series(index=range(max_len))

        # Calculate max for each position
        for i in range(max_len):
            # Get value from first_series (default to -inf if out of range)
            first_value = float("-inf")
            if i < len(series1):
                if notna(series1.iloc[i]):
                    first_value = series1.iloc[i]
                else:
                    first_value = float("-inf")  # treat NaN as -inf for max comparison

            # Get value from second_series (default to -inf if out of range)
            second_value = float("-inf")
            if i < len(series2):
                if notna(series2.iloc[i]):
                    second_value = series2.iloc[i]
                else:
                    second_value = float("-inf")  # treat NaN as -inf for max comparison

            # If both are -inf (representing NaN or out of range), result is NaN
            if first_value == float("-inf") and second_value == float("-inf"):
                merged_result.iloc[i] = float("nan")
            # Otherwise, take the max
            else:
                # If one is -inf, the other will be selected
                merged_result.iloc[i] = max(first_value, second_value)

        return merged_result


class MinStrategy(MetricMergeStrategy):
    """Take the minimum value"""

    def merge(
        self, value1: Union[MetricValue, Series], value2: Union[MetricValue, Series]
    ) -> Union[MetricValue, Series]:
        # Validate inputs
        self.validate(value1, value2)

        # 1. Handle scalar + scalar case
        if not isinstance(value1, Series) and not isinstance(value2, Series):
            return self._handle_scalar_case(value1, value2, self._min_scalars)

        # 2. Handle Series + scalar case
        if isinstance(value1, Series) and not isinstance(value2, Series):
            return self._handle_series_scalar_case(
                value1, value2, True, self._min_series_scalar
            )

        # 3. Handle scalar + Series case
        if not isinstance(value1, Series) and isinstance(value2, Series):
            return self._handle_series_scalar_case(
                value2, value1, False, self._min_series_scalar
            )

        # 4. Handle Series + Series case (with explicit casting for type checker)
        if isinstance(value1, Series) and isinstance(value2, Series):
            return self._handle_series_series_case(
                value1, value2, self._min_series_series
            )

        # This should never happen due to the conditions above, but makes the type checker happy
        raise ValueError("Unexpected input types")

    def _min_scalars(self, value1: MetricValue, value2: MetricValue) -> MetricValue:
        """Find the minimum of two scalar values, handling NaN"""
        # Use pandas Series.min() for proper NaN handling
        s = Series([value1, value2])
        return s.min(skipna=True)

    def _min_series_scalar(
        self, series: Series, scalar: MetricValue, series_is_first: bool
    ) -> Series:
        """Find the minimum between Series values and a scalar"""
        # Create result Series
        merged_result = Series(index=range(len(series)))

        # Compare each element with the scalar
        for i in range(len(merged_result)):
            current_value = series.iloc[i]
            if isna(current_value):
                merged_result.iloc[i] = scalar
            elif isna(scalar):
                merged_result.iloc[i] = current_value
            else:
                merged_result.iloc[i] = min(current_value, scalar)

        return merged_result

    def _min_series_series(self, series1: Series, series2: Series) -> Series:
        """Find the minimum values between two Series"""
        # Create a result Series with appropriate length
        max_len = max(len(series1), len(series2))
        merged_result = Series(index=range(max_len))

        # Calculate min for each position
        for i in range(max_len):
            # Get value from first_series (default to inf if out of range)
            first_value = float("inf")
            if i < len(series1):
                if notna(series1.iloc[i]):
                    first_value = series1.iloc[i]
                else:
                    first_value = float("inf")  # treat NaN as inf for min comparison

            # Get value from second_series (default to inf if out of range)
            second_value = float("inf")
            if i < len(series2):
                if notna(series2.iloc[i]):
                    second_value = series2.iloc[i]
                else:
                    second_value = float("inf")  # treat NaN as inf for min comparison

            # If both are inf (representing NaN or out of range), result is NaN
            if first_value == float("inf") and second_value == float("inf"):
                merged_result.iloc[i] = float("nan")
            # Otherwise, take the min
            else:
                # If one is inf, the other will be selected
                merged_result.iloc[i] = min(first_value, second_value)

        return merged_result


class AverageStrategy(MetricMergeStrategy):
    """Take the average of non-null values"""

    def merge(
        self, value1: Union[MetricValue, Series], value2: Union[MetricValue, Series]
    ) -> Union[MetricValue, Series]:
        # Validate inputs
        self.validate(value1, value2)

        # 1. Handle scalar + scalar case
        if not isinstance(value1, Series) and not isinstance(value2, Series):
            return self._handle_scalar_case(value1, value2, self._avg_scalars)

        # 2. Handle Series + scalar case
        if isinstance(value1, Series) and not isinstance(value2, Series):
            return self._handle_series_scalar_case(
                value1, value2, True, self._avg_series_scalar
            )

        # 3. Handle scalar + Series case
        if not isinstance(value1, Series) and isinstance(value2, Series):
            return self._handle_series_scalar_case(
                value2, value1, False, self._avg_series_scalar
            )

        # 4. Handle Series + Series case (with explicit casting for type checker)
        if isinstance(value1, Series) and isinstance(value2, Series):
            return self._handle_series_series_case(
                value1, value2, self._avg_series_series
            )

        # This should never happen due to the conditions above, but makes the type checker happy
        raise ValueError("Unexpected input types")

    def _avg_scalars(self, value1: MetricValue, value2: MetricValue) -> MetricValue:
        """Calculate the average of two scalar values, handling NaN"""
        # Use pandas Series.mean() for proper NaN handling
        s = Series([value1, value2])
        return s.mean()

    def _avg_series_scalar(
        self, series: Series, scalar: MetricValue, series_is_first: bool
    ) -> Series:
        """Calculate the average between Series values and a scalar"""
        # Create result Series
        merged_result = Series(index=range(len(series)))

        for i in range(len(merged_result)):
            series_value = series.iloc[i]

            # If both are NaN, result is NaN
            if isna(series_value) and isna(scalar):
                merged_result.iloc[i] = float("nan")
            # If one is NaN, use the other value (not an average)
            elif isna(series_value):
                merged_result.iloc[i] = scalar
            elif isna(scalar):
                merged_result.iloc[i] = series_value
            # Otherwise, calculate the average
            else:
                merged_result.iloc[i] = (series_value + scalar) / 2

        return merged_result

    def _avg_series_series(self, series1: Series, series2: Series) -> Series:
        """Calculate the average between two Series values"""
        # Create a result Series with appropriate length
        max_len = max(len(series1), len(series2))
        merged_result = Series(index=range(max_len))

        # Calculate average for each position
        for i in range(max_len):
            # Get values (handle out of bounds)
            first_value = None
            if i < len(series1):
                first_value = series1.iloc[i]

            second_value = None
            if i < len(series2):
                second_value = series2.iloc[i]

            # If both are None or NaN, result is NaN
            if (first_value is None or isna(first_value)) and (
                second_value is None or isna(second_value)
            ):
                merged_result.iloc[i] = float("nan")
            # If first_value is None or NaN, use second_value
            elif first_value is None or isna(first_value):
                merged_result.iloc[i] = second_value
            # If second_value is None or NaN, use first_value
            elif second_value is None or isna(second_value):
                merged_result.iloc[i] = first_value
            # Otherwise, calculate the average
            else:
                merged_result.iloc[i] = (first_value + second_value) / 2

        return merged_result
