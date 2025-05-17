from abc import abstractmethod, ABC
from typing import Dict, TYPE_CHECKING, Optional

from pandas import Series

from tempo.intervals.metrics.strategies import MetricMergeStrategy, KeepLastStrategy

if TYPE_CHECKING:
    from tempo.intervals.core.interval import Interval


class MetricNormalizer(ABC):
    @abstractmethod
    def normalize(self, interval: "Interval") -> Series:
        pass


class MetricMergeConfig:
    """Configuration for metric merging behavior"""

    def __init__(
        self,
        default_strategy: Optional[MetricMergeStrategy] = None,
        column_strategies: Optional[Dict[str, MetricMergeStrategy]] = None,
    ):
        self.default_strategy = default_strategy or KeepLastStrategy()
        self.column_strategies = column_strategies or {}
        self._validate_strategies()

    def _validate_strategies(self) -> None:
        """Validate that all strategies are proper MetricMergeStrategy instances"""
        if not isinstance(self.default_strategy, MetricMergeStrategy):
            raise ValueError(
                "default_strategy must be an instance of MetricMergeStrategy"
            )

        for col, strategy in self.column_strategies.items():
            if not isinstance(strategy, MetricMergeStrategy):
                raise ValueError(
                    f"Strategy for column {col} must be an instance of MetricMergeStrategy"
                )

    def get_strategy(self, column: str) -> MetricMergeStrategy:
        """Get the merge strategy for a specific column"""
        return self.column_strategies.get(column, self.default_strategy)

    def set_strategy(self, column: str, strategy: MetricMergeStrategy) -> None:
        """Set the merge strategy for a specific column"""
        if not isinstance(strategy, MetricMergeStrategy):
            raise ValueError(
                "The provided strategy must be an instance of MetricMergeStrategy"
            )
        self.column_strategies[column] = strategy
