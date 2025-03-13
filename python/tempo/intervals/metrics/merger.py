from abc import ABC
from typing import TYPE_CHECKING, Optional, Union

from pandas import Series

from tempo.intervals.core.exceptions import ErrorMessages
from tempo.intervals.core.types import MetricValue
from tempo.intervals.metrics.operations import MetricMergeConfig
from tempo.intervals.metrics.strategies import MetricMergeStrategy

if TYPE_CHECKING:
    from tempo.intervals.core.interval import Interval


class MetricMerger(ABC):
    """Abstract base class defining metric merging strategy"""

    def __init__(self, merge_config: Optional[MetricMergeConfig] = None):
        self.merge_config = merge_config or MetricMergeConfig()

    def merge(self, interval: "Interval", other: "Interval") -> Series:
        """Merge metrics from two intervals according to strategy"""
        self._validate_metric_columns(interval, other)

        # Create a copy of interval's data
        merged_data = interval.data.copy()

        # Apply merge strategy for each metric column
        for metric_col in interval.metric_fields:
            strategy = self.merge_config.get_strategy(metric_col)
            merged_data[metric_col] = self._apply_merge_strategy(
                interval.data[metric_col], other.data[metric_col], strategy
            )

        return merged_data

    @staticmethod
    def _apply_merge_strategy(
            value1: Series,
            value2: Series,
            strategy: MetricMergeStrategy,
    ) -> Union[MetricValue, Series]:
        """Apply the specified merge strategy to two values"""
        try:
            strategy.validate(value1, value2)
            return strategy.merge(value1, value2)
        except Exception as e:
            raise ValueError(f"Strategy {strategy.__class__.__name__} failed: {str(e)}")

    @staticmethod
    def _validate_metric_columns(interval: "Interval", other: "Interval") -> None:
        """Validate that metric columns are aligned between intervals"""
        if len(interval.metric_fields) != len(other.metric_fields):
            raise ValueError(ErrorMessages.METRIC_COLUMNS_LENGTH)


class DefaultMetricMerger(MetricMerger):
    """Default implementation that uses configured merge strategies"""

    pass
