from abc import ABC, abstractmethod
from typing import Union, Tuple

from pandas import notna, isna, Series

from tempo.intervals.core.types import MetricValue


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
    ) -> Union[MetricValue, Series]:
        """Optional validation of values before merging"""
        pass

    def _convert_to_series(
            self,
            value1: Union[MetricValue, Series],
            value2: Union[MetricValue, Series]
    ) -> Tuple[Series, Series, bool]:
        """
        Convert scalar values to Series if needed.

        Returns:
            Tuple containing:
            - Series version of value1
            - Series version of value2
            - Boolean indicating if either input was a Series
        """
        was_series = isinstance(value1, Series) or isinstance(value2, Series)

        if not isinstance(value1, Series):
            v1 = Series([value1])
        else:
            v1 = value1.copy()

        if not isinstance(value2, Series):
            v2 = Series([value2])
        else:
            v2 = value2.copy()

        return v1, v2, was_series


class KeepFirstStrategy(MetricMergeStrategy):
    """Keep the first non-null value"""

    def merge(self, value1: Union[MetricValue, Series], value2: Union[MetricValue, Series]) -> Union[
        MetricValue, Series]:
        # Handle scalar inputs
        if not isinstance(value1, Series) and not isinstance(value2, Series):
            return value1 if notna(value1) else value2

        # Convert to Series if needed
        v1, v2, was_series = self._convert_to_series(value1, value2)

        # Create a new result Series
        result = v1.copy()

        # For each position in the result
        for i in range(len(result)):
            # If value1 is null at this position, take value2's value
            if isna(result.iloc[i]):
                result.iloc[i] = v2.iloc[i]

        # Return the correct type
        return result if was_series else result.iloc[0]


class KeepLastStrategy(MetricMergeStrategy):
    """Keep the last non-null value"""

    def merge(self, value1: Union[MetricValue, Series], value2: Union[MetricValue, Series]) -> Union[
        MetricValue, Series]:
        # Handle scalar inputs
        if not isinstance(value1, Series) and not isinstance(value2, Series):
            return value2 if notna(value2) else value1

        # Convert to Series if needed
        v1, v2, was_series = self._convert_to_series(value1, value2)

        # Create a new result Series - start with value2 as the preference
        result = v2.copy()

        # For each position in the result
        for i in range(len(result)):
            # If value2 is null at this position, take value1's value
            if isna(result.iloc[i]):
                result.iloc[i] = v1.iloc[i]

        # Return the correct type
        return result if was_series else result.iloc[0]


class SumStrategy(MetricMergeStrategy):
    """Sum the values, treating nulls as 0"""

    def merge(self, value1: Union[MetricValue, Series], value2: Union[MetricValue, Series]) -> Union[
        MetricValue, Series]:
        # Handle scalar case first
        if not isinstance(value1, Series) and not isinstance(value2, Series):
            # Validate scalar inputs
            self.validate(value1, value2)

            # Treat NaN as 0
            v1 = 0 if isna(value1) else value1
            v2 = 0 if isna(value2) else value2
            return v1 + v2

        # Handle Series-based inputs
        v1, v2, was_series = self._convert_to_series(value1, value2)

        # Validate Series
        self.validate(v1, v2)

        # Handle Series + scalar case specially
        if isinstance(value1, Series) and not isinstance(value2, Series):
            # When adding a scalar to a Series, just add the scalar value to each element
            result = v1.fillna(0).copy()
            scalar_value = 0 if isna(value2) else value2
            for i in range(len(result)):
                result.iloc[i] += scalar_value
            return result

        # Handle scalar + Series case
        elif not isinstance(value1, Series) and isinstance(value2, Series):
            # When adding a scalar to a Series, just add the scalar value to each element
            result = v2.fillna(0).copy()
            scalar_value = 0 if isna(value1) else value1
            for i in range(len(result)):
                result.iloc[i] += scalar_value
            return result

        # Handle Series + Series case
        else:
            # Ensure same length Series
            if len(v1) != len(v2):
                # If we have Series of different lengths, this is unexpected
                # but we'll try to handle it reasonably
                max_len = max(len(v1), len(v2))
                if len(v1) < max_len:
                    # Extend v1 with NaN values if needed
                    v1 = v1.reindex(range(max_len), fill_value=float('nan'))
                if len(v2) < max_len:
                    # Extend v2 with NaN values if needed
                    v2 = v2.reindex(range(max_len), fill_value=float('nan'))

            # Sum the Series, treating NaN as 0
            result = v1.fillna(0).add(v2.fillna(0))

            # Return appropriate type
            return result if was_series else result.iloc[0]

    def validate(self, value1: Union[MetricValue, Series], value2: Union[MetricValue, Series]) -> None:
        # For Series inputs
        if isinstance(value1, Series):
            for val in value1:
                if notna(val) and not isinstance(val, (int, float)):
                    raise ValueError("SumStrategy requires numeric values")

        elif notna(value1) and not isinstance(value1, (int, float)):
            raise ValueError("SumStrategy requires numeric values")

        if isinstance(value2, Series):
            for val in value2:
                if notna(val) and not isinstance(val, (int, float)):
                    raise ValueError("SumStrategy requires numeric values")

        elif notna(value2) and not isinstance(value2, (int, float)):
            raise ValueError("SumStrategy requires numeric values")


class MaxStrategy(MetricMergeStrategy):
    """Take the maximum value"""

    def merge(self, value1: Union[MetricValue, Series], value2: Union[MetricValue, Series]) -> Union[
        MetricValue, Series]:
        # Handle scalar case
        if not isinstance(value1, Series) and not isinstance(value2, Series):
            # Use pandas Series.max() for proper NaN handling
            s = Series([value1, value2])
            return s.max(skipna=True)

        # Handle Series + scalar case specifically
        if isinstance(value1, Series) and not isinstance(value2, Series):
            # Create result Series
            result = Series(index=range(len(value1)))
            scalar_val = value2

            # Compare each element with the scalar
            for i in range(len(result)):
                val = value1.iloc[i]
                if isna(val):
                    result.iloc[i] = scalar_val
                elif isna(scalar_val):
                    result.iloc[i] = val
                else:
                    result.iloc[i] = max(val, scalar_val)

            return result

        # Handle scalar + Series case
        if not isinstance(value1, Series) and isinstance(value2, Series):
            # Create result Series
            result = Series(index=range(len(value2)))
            scalar_val = value1

            # Compare each element with the scalar
            for i in range(len(result)):
                val = value2.iloc[i]
                if isna(val):
                    result.iloc[i] = scalar_val
                elif isna(scalar_val):
                    result.iloc[i] = val
                else:
                    result.iloc[i] = max(scalar_val, val)

            return result

        # Otherwise handle Series + Series
        # Convert to Series if needed
        v1, v2, was_series = self._convert_to_series(value1, value2)

        # Create a result Series with appropriate length
        result = Series(index=range(max(len(v1), len(v2))))

        # Calculate max for each position
        for i in range(len(result)):
            # Get value from v1 (handle index out of range)
            if i < len(v1):
                val1 = v1.iloc[i]
            else:
                val1 = float('-inf')  # Use -inf as placeholder for missing values

            # Get value from v2 (handle index out of range)
            if i < len(v2):
                val2 = v2.iloc[i]
            else:
                val2 = float('-inf')  # Use -inf as placeholder for missing values

            # For max(), we treat NaN as missing value (not included in comparison)
            if isna(val1) and isna(val2):
                result.iloc[i] = float('nan')
            elif isna(val1):
                result.iloc[i] = val2
            elif isna(val2):
                result.iloc[i] = val1
            else:
                result.iloc[i] = max(val1, val2)

        # Return the correct type
        return result if was_series else result.iloc[0]


class MinStrategy(MetricMergeStrategy):
    """Take the minimum value"""

    def merge(self, value1: Union[MetricValue, Series], value2: Union[MetricValue, Series]) -> Union[
        MetricValue, Series]:
        # Handle scalar case
        if not isinstance(value1, Series) and not isinstance(value2, Series):
            # Use pandas Series.min() for proper NaN handling
            s = Series([value1, value2])
            return s.min(skipna=True)

        # Handle Series + scalar case specifically
        if isinstance(value1, Series) and not isinstance(value2, Series):
            # Create result Series
            result = Series(index=range(len(value1)))
            scalar_val = value2

            # Compare each element with the scalar
            for i in range(len(result)):
                val = value1.iloc[i]
                if isna(val):
                    result.iloc[i] = scalar_val
                elif isna(scalar_val):
                    result.iloc[i] = val
                else:
                    result.iloc[i] = min(val, scalar_val)

            return result

        # Handle scalar + Series case
        if not isinstance(value1, Series) and isinstance(value2, Series):
            # Create result Series
            result = Series(index=range(len(value2)))
            scalar_val = value1

            # Compare each element with the scalar
            for i in range(len(result)):
                val = value2.iloc[i]
                if isna(val):
                    result.iloc[i] = scalar_val
                elif isna(scalar_val):
                    result.iloc[i] = val
                else:
                    result.iloc[i] = min(scalar_val, val)

            return result

        # Otherwise handle Series + Series
        # Convert to Series if needed
        v1, v2, was_series = self._convert_to_series(value1, value2)

        # Create a result Series with appropriate length
        result = Series(index=range(max(len(v1), len(v2))))

        # Calculate min for each position
        for i in range(len(result)):
            # Get value from v1 (handle index out of range)
            if i < len(v1):
                val1 = v1.iloc[i]
            else:
                val1 = float('inf')  # Use inf as placeholder for missing values

            # Get value from v2 (handle index out of range)
            if i < len(v2):
                val2 = v2.iloc[i]
            else:
                val2 = float('inf')  # Use inf as placeholder for missing values

            # For min(), we treat NaN as missing value (not included in comparison)
            if isna(val1) and isna(val2):
                result.iloc[i] = float('nan')
            elif isna(val1):
                result.iloc[i] = val2
            elif isna(val2):
                result.iloc[i] = val1
            else:
                result.iloc[i] = min(val1, val2)

        # Return the correct type
        return result if was_series else result.iloc[0]


class AverageStrategy(MetricMergeStrategy):
    """Take the average of non-null values"""

    def merge(self, value1: Union[MetricValue, Series], value2: Union[MetricValue, Series]) -> Union[
        MetricValue, Series]:
        # Handle scalar case
        if not isinstance(value1, Series) and not isinstance(value2, Series):
            # Validate scalar inputs
            self.validate(value1, value2)

            # Use pandas Series.mean() for proper NaN handling
            s = Series([value1, value2])
            return s.mean()

        # Handle Series + scalar case
        if isinstance(value1, Series) and not isinstance(value2, Series):
            # Validate inputs
            self.validate(value1, value2)

            # Convert scalar to Series for averaging
            scalar_val = value2
            result = Series(index=range(len(value1)))

            for i in range(len(result)):
                val = value1.iloc[i]

                # If both are NaN, result is NaN
                if isna(val) and isna(scalar_val):
                    result.iloc[i] = float('nan')
                # If one is NaN, use the other value (not an average)
                elif isna(val):
                    result.iloc[i] = scalar_val
                elif isna(scalar_val):
                    result.iloc[i] = val
                # Otherwise, calculate the average
                else:
                    result.iloc[i] = (val + scalar_val) / 2

            return result

        # Handle scalar + Series case
        if not isinstance(value1, Series) and isinstance(value2, Series):
            # Validate inputs
            self.validate(value1, value2)

            # Convert scalar to Series for averaging
            scalar_val = value1
            result = Series(index=range(len(value2)))

            for i in range(len(result)):
                val = value2.iloc[i]

                # If both are NaN, result is NaN
                if isna(val) and isna(scalar_val):
                    result.iloc[i] = float('nan')
                # If one is NaN, use the other value (not an average)
                elif isna(val):
                    result.iloc[i] = scalar_val
                elif isna(scalar_val):
                    result.iloc[i] = val
                # Otherwise, calculate the average
                else:
                    result.iloc[i] = (scalar_val + val) / 2

            return result

        # Handle Series + Series case
        # Convert to Series if needed
        v1, v2, was_series = self._convert_to_series(value1, value2)

        # Validate Series
        self.validate(v1, v2)

        # Create a result Series with appropriate length
        result = Series(index=range(max(len(v1), len(v2))))

        # Calculate average for each position
        for i in range(len(result)):
            # Get values (handle out of bounds)
            val1 = v1.iloc[i] if i < len(v1) else float('nan')
            val2 = v2.iloc[i] if i < len(v2) else float('nan')

            # If both are NaN, result is NaN
            if isna(val1) and isna(val2):
                result.iloc[i] = float('nan')
            # If one is NaN, use the other value (not an average)
            elif isna(val1):
                result.iloc[i] = val2
            elif isna(val2):
                result.iloc[i] = val1
            # Otherwise, calculate the average
            else:
                result.iloc[i] = (val1 + val2) / 2

        # Return the correct type
        return result if was_series else result.iloc[0]

    def validate(self, value1: Union[MetricValue, Series], value2: Union[MetricValue, Series]) -> None:
        # For Series inputs
        if isinstance(value1, Series):
            for val in value1:
                if notna(val) and not isinstance(val, (int, float)):
                    raise ValueError("AverageStrategy requires numeric values")

        elif notna(value1) and not isinstance(value1, (int, float)):
            raise ValueError("AverageStrategy requires numeric values")

        if isinstance(value2, Series):
            for val in value2:
                if notna(val) and not isinstance(val, (int, float)):
                    raise ValueError("AverageStrategy requires numeric values")

        elif notna(value2) and not isinstance(value2, (int, float)):
            raise ValueError("AverageStrategy requires numeric values")