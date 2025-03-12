from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from pandas import Series

from tempo.intervals.core.interval import Interval


@dataclass
class ResolutionResult:
    """Represents the result of interval resolution"""
    _resolved_intervals: List[Series] = field(repr=False)  # Private field for storage
    metadata: Optional[Dict] = None
    warnings: List[str] = field(default_factory=list)

    def __init__(self, resolved_intervals: List[Series], metadata=None, warnings=None):
        # Make defensive copies of mutable inputs
        self._resolved_intervals = resolved_intervals.copy()
        self.metadata = metadata.copy() if metadata is not None else None
        self.warnings = warnings.copy() if warnings is not None else []

    @property
    def resolved_intervals(self) -> List[Series]:
        """Return a copy of the resolved intervals to prevent modification"""
        return self._resolved_intervals.copy()


class OverlapResolver(ABC):
    @abstractmethod
    def resolve(self, interval: "Interval", other: "Interval") -> list[Series]:
        pass


class MetricsEquivalentResolver(OverlapResolver):
    """
    Resolver for intervals with equivalent metrics that overlap.
    Returns a single interval spanning the total time range.
    """

    def resolve(self, interval: "Interval", other: "Interval") -> list[Series]:
        # Create a new interval spanning the entire range
        earliest = interval if interval._start <= other._start else other
        latest = interval if interval._end >= other._end else other

        # Use interval's data as base since metrics are equivalent
        result = interval.data.copy()
        result[interval.start_field] = earliest.start
        result[interval.end_field] = latest.end

        return [result]


class BeforeResolver(OverlapResolver):
    """Resolver for intervals that are completely before one another"""

    def resolve(self, interval: "Interval", other: "Interval") -> list[Series]:
        # No resolution needed - intervals are already disjoint
        return [interval.data, other.data]


class MeetsResolver(OverlapResolver):
    """Resolver for intervals that meet exactly at a boundary"""

    def resolve(self, interval: "Interval", other: "Interval") -> list[Series]:
        # No resolution needed - intervals are already disjoint
        return [interval.data, other.data]


class OverlapsResolver(OverlapResolver):
    """Resolver for intervals where one overlaps the start of the other"""

    def resolve(self, interval: "Interval", other: "Interval") -> list[Series]:
        # Split into three intervals: before overlap, overlap, after overlap
        first_part = interval.update_end(other._start).data

        # Overlapping part
        interval_part = interval.update_start(other._start)
        other_part = other.update_end(interval._end)
        merged_overlap = interval_part.merge_metrics(other_part)

        # Last part
        last_part = other.update_start(interval._end).data

        return [first_part, merged_overlap, last_part]


class StartsResolver(OverlapResolver):
    """Resolver for intervals that start together but one ends earlier"""

    def resolve(self, interval: "Interval", other: "Interval") -> list[Series]:
        # Shared start portion
        other_updated = other.update_end(interval._end)
        shared_part = interval.merge_metrics(other_updated)

        # Remaining portion of longer interval
        remaining_part = other.update_start(interval._end).data

        return [shared_part, remaining_part]


class DuringResolver(OverlapResolver):
    """Resolver for intervals where one is completely contained within the other"""

    def resolve(self, interval: "Interval", other: "Interval") -> list[Series]:
        # First part of containing interval
        first_part = other.update_end(interval._start).data

        # Contained interval (merge metrics)
        other_middle = other.update_end(interval._end)
        merged_middle = interval.merge_metrics(other_middle)

        # Last part of containing interval
        last_part = other.update_start(interval._end).data

        return [first_part, merged_middle, last_part]


class FinishesResolver(OverlapResolver):
    """Resolver for intervals that end together but started at different times"""

    def resolve(self, interval: "Interval", other: "Interval") -> list[Series]:
        # Non-overlapping start portion
        first_part = other.update_end(interval._start).data

        # Shared end portion (merge metrics)
        other_updated = other.update_start(interval._start)
        merged_end = interval.merge_metrics(other_updated)

        return [first_part, merged_end]


class EqualsResolver(OverlapResolver):
    """Resolver for intervals that are exactly equal"""

    def resolve(self, interval: "Interval", other: "Interval") -> list[Series]:
        # Merge metrics for the identical intervals
        merged = interval.merge_metrics(other)
        return [merged]


class ContainsResolver(OverlapResolver):
    """Resolver for intervals where one completely contains the other"""

    def resolve(self, interval: "Interval", other: "Interval") -> list[Series]:
        # Same logic as DuringResolver but with intervals swapped
        # First part of containing interval
        first_part = interval.update_end(other._start).data

        # Contained interval (merge metrics)
        interval_middle = interval.update_end(other._end)
        merged_middle = other.merge_metrics(interval_middle)

        # Last part of containing interval
        last_part = interval.update_start(other._end).data

        return [first_part, merged_middle, last_part]


class StartedByResolver(OverlapResolver):
    """Resolver for intervals where both start together but one continues longer"""

    def resolve(self, interval: "Interval", other: "Interval") -> list[Series]:
        # Same logic as StartsResolver but with intervals swapped
        # Shared start portion
        interval_start = interval.update_end(other._end)
        shared_part = other.merge_metrics(interval_start)

        # Remaining portion of longer interval
        remaining_part = interval.update_start(other._end).data

        return [shared_part, remaining_part]


class FinishedByResolver(OverlapResolver):
    """Resolver for intervals where other ends together with interval but starts later"""

    def resolve(self, interval: "Interval", other: "Interval") -> list[Series]:
        # First part before other starts
        first_part = interval.update_end(other._start).data

        # Shared end portion (merge metrics)
        updated_interval = interval.update_start(other._start)
        merged_result = updated_interval.merge_metrics(other)

        return [first_part, merged_result]


class OverlappedByResolver(OverlapResolver):
    """Resolver for intervals where one is overlapped by the start of another"""

    def resolve(self, interval: "Interval", other: "Interval") -> list[Series]:
        # Same logic as OverlapsResolver but with intervals swapped
        # First part (before overlap)
        first_part = other.update_end(interval._start).data

        # Overlapping part (merge metrics)
        other_part = other.update_start(interval._start)
        interval_part = interval.update_end(other._end)
        merged_overlap = other_part.merge_metrics(interval_part)

        # Last part (after overlap)
        last_part = interval.update_start(other._end).data

        return [first_part, merged_overlap, last_part]


class MetByResolver(OverlapResolver):
    """Resolver for intervals where one ends exactly where the other._starts"""

    def resolve(self, interval: "Interval", other: "Interval") -> list[Series]:
        # No resolution needed - intervals are already disjoint
        return [other.data, interval.data]


class AfterResolver(OverlapResolver):
    """Resolver for intervals that are completely after one another"""

    def resolve(self, interval: "Interval", other: "Interval") -> list[Series]:
        # No resolution needed - intervals are already disjoint
        return [other.data, interval.data]