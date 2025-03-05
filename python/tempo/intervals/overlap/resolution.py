from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from pandas import Series

from tempo.intervals.core.interval import Interval


@dataclass
class ResolutionResult:
    """Represents the result of interval resolution"""
    resolved_intervals: List[Series]
    metadata: Optional[Dict] = None
    warnings: List[str] = field(default_factory=list)


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
        earliest = interval if interval.start <= other.start else other
        latest = interval if interval.end >= other.end else other

        # Use interval's data as base since metrics are equivalent
        result = interval.data.copy()
        result[interval.start_field] = earliest.user_start
        result[interval.end_field] = latest.user_end

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
        return [
            # First part (before overlap)
            interval.update_end(other.start).data,

            # Overlapping part (merge metrics)
            interval.update_start(other.start).merge_metrics(
                other.update_end(interval.end)
            ),

            # Last part (after overlap)
            other.update_start(interval.end).data
        ]


class StartsResolver(OverlapResolver):
    """Resolver for intervals that start together but one ends earlier"""

    def resolve(self, interval: "Interval", other: "Interval") -> list[Series]:
        return [
            # Shared start portion
            interval.merge_metrics(other.update_end(interval.end)),

            # Remaining portion of longer interval
            other.update_start(interval.end).data
        ]


class DuringResolver(OverlapResolver):
    """Resolver for intervals where one is completely contained within the other"""

    def resolve(self, interval: "Interval", other: "Interval") -> list[Series]:
        return [
            # First part of containing interval
            other.update_end(interval.start).data,

            # Contained interval (merge metrics)
            interval.merge_metrics(other.update_end(interval.end)),

            # Last part of containing interval
            other.update_start(interval.end).data
        ]


class FinishesResolver(OverlapResolver):
    """Resolver for intervals that end together but started at different times"""

    def resolve(self, interval: "Interval", other: "Interval") -> list[Series]:
        return [
            # Non-overlapping start portion
            other.update_end(interval.start).data,

            # Shared end portion (merge metrics)
            interval.merge_metrics(other.update_start(interval.start))
        ]


class EqualsResolver(OverlapResolver):
    """Resolver for intervals that are exactly equal"""

    def resolve(self, interval: "Interval", other: "Interval") -> list[Series]:
        # Merge metrics for the identical intervals
        return [interval.merge_metrics(other)]


class ContainsResolver(OverlapResolver):
    """Resolver for intervals where one completely contains the other"""

    def resolve(self, interval: "Interval", other: "Interval") -> list[Series]:
        # Same logic as DuringResolver but with intervals swapped
        return [
            # First part of containing interval
            interval.update_end(other.start).data,

            # Contained interval (merge metrics)
            other.merge_metrics(interval.update_end(other.end)),

            # Last part of containing interval
            interval.update_start(other.end).data
        ]


class StartedByResolver(OverlapResolver):
    """Resolver for intervals where both start together but one continues longer"""

    def resolve(self, interval: "Interval", other: "Interval") -> list[Series]:
        # Same logic as StartsResolver but with intervals swapped
        return [
            # Shared start portion
            other.merge_metrics(interval.update_end(other.end)),

            # Remaining portion of longer interval
            interval.update_start(other.end).data
        ]


class FinishedByResolver(OverlapResolver):
    """Resolver for intervals where other ends together with interval but starts later"""

    def resolve(self, interval: "Interval", other: "Interval") -> list[Series]:
        return [
            # First part before other starts
            interval.update_end(other.start).data,

            # Shared end portion (merge metrics)
            interval.update_start(other.start).merge_metrics(other)
        ]


class OverlappedByResolver(OverlapResolver):
    """Resolver for intervals where one is overlapped by the start of another"""

    def resolve(self, interval: "Interval", other: "Interval") -> list[Series]:
        # Same logic as OverlapsResolver but with intervals swapped
        return [
            # First part (before overlap)
            other.update_end(interval.start).data,

            # Overlapping part (merge metrics)
            other.update_end(other.end).merge_metrics(
                interval.update_end(other.end)
            ),

            # Last part (after overlap)
            interval.update_start(other.end).data
        ]


class MetByResolver(OverlapResolver):
    """Resolver for intervals where one ends exactly where the other starts"""

    def resolve(self, interval: "Interval", other: "Interval") -> list[Series]:
        # No resolution needed - intervals are already disjoint
        return [other.data, interval.data]


class AfterResolver(OverlapResolver):
    """Resolver for intervals that are completely after one another"""

    def resolve(self, interval: "Interval", other: "Interval") -> list[Series]:
        # No resolution needed - intervals are already disjoint
        return [other.data, interval.data]
