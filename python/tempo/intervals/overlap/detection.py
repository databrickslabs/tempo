from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tempo.intervals.core.interval import Interval
else:
    # For runtime, use Any as a placeholder for Interval
    Interval = Any


class OverlapChecker(ABC):
    """Abstract base class for overlap checking strategies"""

    @abstractmethod
    def check(self, interval: "Interval", other: "Interval") -> bool:
        pass


class MetricsEquivalentChecker(OverlapChecker):
    """Checks if intervals have equivalent metrics and have any overlap or containment"""

    def check(self, interval: "Interval", other: "Interval") -> bool:
        # First check if metrics are equal using pandas equals() for proper NULL handling
        interval_metrics = interval.data[interval.metric_fields]
        other_metrics = other.data[interval.metric_fields]
        metrics_equal = interval_metrics.equals(other_metrics)

        if not metrics_equal:
            return False

        # Then check if intervals overlap or one contains the other
        has_overlap = (
            # Overlap cases
                (interval._start.internal_value < other._end.internal_value and
                 interval._end.internal_value > other._start.internal_value) or
                # Containment cases
                (interval._start.internal_value <= other._start.internal_value and
                 interval._end.internal_value >= other._end.internal_value) or
                (other._start.internal_value <= interval._start.internal_value and
                 other._end.internal_value >= interval._end.internal_value)
        )

        return metrics_equal and has_overlap


class BeforeChecker(OverlapChecker):
    """Checks if interval is completely before other"""

    def check(self, interval: "Interval", other: "Interval") -> bool:
        return interval._end <= other._start


class MeetsChecker(OverlapChecker):
    """Checks if interval._ends exactly where other._starts"""

    def check(self, interval: "Interval", other: "Interval") -> bool:
        return interval._end == other._start


class OverlapsChecker(OverlapChecker):
    """Checks if interval overlaps the start of other"""

    def check(self, interval: "Interval", other: "Interval") -> bool:
        return interval._start < other._start < interval._end < other._end


class StartsChecker(OverlapChecker):
    """Checks if interval._starts together with other but ends earlier"""

    def check(self, interval: "Interval", other: "Interval") -> bool:
        return (interval._start == other._start and
                interval._end < other._end)


class DuringChecker(OverlapChecker):
    """Checks if interval is completely contained within other"""

    def check(self, interval: "Interval", other: "Interval") -> bool:
        return (interval._start > other._start and
                interval._end < other._end)


class FinishesChecker(OverlapChecker):
    """Checks if interval._ends together with other but starts later"""

    def check(self, interval: "Interval", other: "Interval") -> bool:
        return (interval._start > other._start and
                interval._end == other._end)


class EqualsChecker(OverlapChecker):
    """Checks if interval is identical to other"""

    def check(self, interval: "Interval", other: "Interval") -> bool:
        return (interval._start == other._start and
                interval._end == other._end)


class ContainsChecker(OverlapChecker):
    """Checks if interval completely contains other"""

    def check(self, interval: "Interval", other: "Interval") -> bool:
        return (interval._start < other._start and
                interval._end > other._end)


class StartedByChecker(OverlapChecker):
    """Checks if other._starts together with interval but ends earlier"""

    def check(self, interval: "Interval", other: "Interval") -> bool:
        return (interval._start == other._start and
                interval._end > other._end)


class FinishedByChecker(OverlapChecker):
    """Checks if other._ends together with interval but starts later"""

    def check(self, interval: "Interval", other: "Interval") -> bool:
        return (interval._end == other._end and
                interval._start < other._start)


class OverlappedByChecker(OverlapChecker):
    """Checks if other overlaps the start of interval"""

    def check(self, interval: "Interval", other: "Interval") -> bool:
        return other._start < interval._start < other._end < interval._end


class MetByChecker(OverlapChecker):
    """Checks if other._ends exactly where interval._starts"""

    def check(self, interval: "Interval", other: "Interval") -> bool:
        return other._end == interval._start


class AfterChecker(OverlapChecker):
    """Checks if interval is completely after other"""

    def check(self, interval: "Interval", other: "Interval") -> bool:
        return interval._start >= other._end
