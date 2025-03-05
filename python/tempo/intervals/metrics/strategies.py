from abc import ABC, abstractmethod
from typing import Union

from pandas import notna, Series

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


class KeepFirstStrategy(MetricMergeStrategy):
    """Keep the first non-null value"""

    def merge(self, value1: MetricValue, value2: MetricValue) -> MetricValue:
        return value1 if notna(value1) else value2


class KeepLastStrategy(MetricMergeStrategy):
    """Keep the last non-null value"""

    def merge(self, value1: MetricValue, value2: MetricValue) -> MetricValue:
        return value2 if notna(value2) else value1


class SumStrategy(MetricMergeStrategy):
    """Sum the values, treating nulls as 0"""

    def merge(self, value1: MetricValue, value2: MetricValue) -> MetricValue:
        return Series([value1, value2]).fillna(0).sum()

    def validate(self, value1: MetricValue, value2: MetricValue) -> None:
        if notna(value1) and not isinstance(value1, (int, float)):
            raise ValueError("SumStrategy requires numeric values")
        if notna(value2) and not isinstance(value2, (int, float)):
            raise ValueError("SumStrategy requires numeric values")


class MaxStrategy(MetricMergeStrategy):
    """Take the maximum value"""

    def merge(self, value1: MetricValue, value2: MetricValue) -> MetricValue:
        return Series([value1, value2]).max()


class MinStrategy(MetricMergeStrategy):
    """Take the minimum value"""

    def merge(self, value1: MetricValue, value2: MetricValue) -> MetricValue:
        return Series([value1, value2]).min()


class AverageStrategy(MetricMergeStrategy):
    """Take the average of non-null values"""

    def merge(self, value1: MetricValue, value2: MetricValue) -> MetricValue:
        return Series([value1, value2]).mean()

    def validate(self, value1: MetricValue, value2: MetricValue) -> None:
        if notna(value1) and not isinstance(value1, (int, float)):
            raise ValueError("AverageStrategy requires numeric values")
        if notna(value2) and not isinstance(value2, (int, float)):
            raise ValueError("AverageStrategy requires numeric values")
