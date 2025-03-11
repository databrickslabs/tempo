from collections import OrderedDict
from typing import Optional, TYPE_CHECKING, Any

from pandas import Series

from tempo.intervals.core.exceptions import ErrorMessages

if TYPE_CHECKING:
    from tempo.intervals.core.interval import Interval
else:
    # For runtime, use Any as a placeholder for Interval
    Interval = Any

from tempo.intervals.overlap.detection import MetricsEquivalentChecker, EqualsChecker, DuringChecker, ContainsChecker, \
    StartsChecker, StartedByChecker, FinishesChecker, FinishedByChecker, MeetsChecker, MetByChecker, OverlapsChecker, \
    OverlappedByChecker, BeforeChecker, AfterChecker
from tempo.intervals.overlap.resolution import OverlapResolver, MetricsEquivalentResolver, EqualsResolver, \
    DuringResolver, ContainsResolver, StartsResolver, StartedByResolver, FinishesResolver, FinishedByResolver, \
    MeetsResolver, MetByResolver, OverlapsResolver, OverlappedByResolver, BeforeResolver, AfterResolver
from tempo.intervals.overlap.types import OverlapType


class IntervalTransformer:
    def __init__(
            self,
            interval: "Interval",
            other: "Interval",
    ) -> None:
        """
        Initialize OverlapResolver with two intervals, ensuring the one with the earlier start comes first.

        Args:
            interval (Interval): The first interval.
            other (Interval): The second interval.
        """

        # Ensure intervals are ordered by start time
        if interval._start <= other._start:
            self.interval = interval
            self.other = other
        else:
            self.interval = other
            self.other = interval

        self.validate_intervals(self.interval, self.other)

    @staticmethod
    def validate_intervals(interval: "Interval", other: "Interval"):
        """Validate that intervals can be compared"""
        if set(interval.data.index) != set(other.data.index):
            raise ValueError(ErrorMessages.INTERVAL_INDICES.format(interval.data.index, other.data.index))

    def detect_relationship(self) -> Optional[OverlapType]:
        """
        Detect the Allen's interval relationship between the intervals.
        Returns None if no relationship is found.

        The order of checks is important as some relationships are more specific
        than others. For example, EQUALS is more specific than STARTS.
        """
        # Check for metric equivalence first
        if MetricsEquivalentChecker().check(self.interval, self.other):
            return OverlapType.METRICS_EQUIVALENT

        checkers = OrderedDict([
            # 1. Most specific - exact equality
            (OverlapType.EQUALS, EqualsChecker()),  # intervals are identical

            # 2. Containment relationships (one interval completely contains another)
            (OverlapType.DURING, DuringChecker()),  # interval is inside other
            (OverlapType.CONTAINS, ContainsChecker()),  # interval contains other

            # 3. Boundary sharing relationships (intervals share a boundary but have different spans)
            (OverlapType.STARTS, StartsChecker()),  # intervals start together, interval._ends first
            (OverlapType.STARTED_BY, StartedByChecker()),  # intervals start together, other._ends first
            (OverlapType.FINISHES, FinishesChecker()),  # intervals end together, interval._starts later
            (OverlapType.FINISHED_BY, FinishedByChecker()),  # intervals end together, other._starts later

            # 4. Boundary touching relationships (intervals touch but don't overlap)
            (OverlapType.MEETS, MeetsChecker()),  # interval._ends where other._starts
            (OverlapType.MET_BY, MetByChecker()),  # other._ends where interval._starts

            # 5. Partial overlap relationships (intervals overlap but don't share boundaries)
            (OverlapType.OVERLAPS, OverlapsChecker()),  # interval._starts first, overlaps start of other
            (OverlapType.OVERLAPPED_BY, OverlappedByChecker()),  # other._starts first, overlaps start of interval

            # 6. Disjoint relationships (no overlap)
            (OverlapType.BEFORE, BeforeChecker()),  # interval completely before other
            (OverlapType.AFTER, AfterChecker())  # interval completely after other
        ])

        for relationship_type, checker in checkers.items():
            if checker.check(self.interval, self.other):
                return relationship_type

        return None

    def resolve_overlap(self) -> list[Series]:
        """Resolve overlapping intervals into disjoint intervals."""
        relationship = self.detect_relationship()
        if relationship is None:
            raise NotImplementedError("Unable to determine interval relationship")

        resolver = self._get_resolver(relationship)
        return resolver.resolve(self.interval, self.other)

    @staticmethod
    def _get_resolver(relationship: OverlapType) -> OverlapResolver:
        """Get the appropriate resolver for the relationship type."""
        resolvers = OrderedDict([
            # Special case: Metric equivalence overrides Allen relationships
            (OverlapType.METRICS_EQUIVALENT, MetricsEquivalentResolver()),
            # Merges overlapping intervals with equal metrics
            # Returns: [merged_spanning_interval]

            # 1. Most specific - exact equality
            (OverlapType.EQUALS, EqualsResolver()),  # Merges metrics for identical intervals
            # Returns: [merged_interval]

            # 2. Containment relationships (one interval completely contains another)
            (OverlapType.DURING, DuringResolver()),  # Interval is inside other
            # Returns: [before_contained, contained_with_merged_metrics, after_contained]

            (OverlapType.CONTAINS, ContainsResolver()),  # Interval contains other
            # Returns: [before_other, other_with_merged_metrics, after_other]

            # 3. Boundary sharing relationships (intervals share a boundary but have different spans)
            (OverlapType.STARTS, StartsResolver()),  # Intervals start together, interval._ends first
            # Returns: [shared_start_with_merged_metrics, remaining_other]

            (OverlapType.STARTED_BY, StartedByResolver()),  # Intervals start together, other._ends first
            # Returns: [shared_start_with_merged_metrics, remaining_interval]

            (OverlapType.FINISHES, FinishesResolver()),  # Intervals end together, interval._starts later
            # Returns: [other_before_interval, shared_end_with_merged_metrics]

            (OverlapType.FINISHED_BY, FinishedByResolver()),  # Intervals end together, other._starts later
            # Returns: [interval_before_other, shared_end_with_merged_metrics]

            # 4. Boundary touching relationships (intervals touch but don't overlap)
            (OverlapType.MEETS, MeetsResolver()),  # interval._ends where other._starts
            # Returns: [interval, other] (no merging needed)

            (OverlapType.MET_BY, MetByResolver()),  # other._ends where interval._starts
            # Returns: [other, interval] (no merging needed)

            # 5. Partial overlap relationships (intervals overlap but don't share boundaries)
            (OverlapType.OVERLAPS, OverlapsResolver()),  # interval._starts first, overlaps start of other
            # Returns: [before_overlap, overlapping_with_merged_metrics, after_overlap]

            (OverlapType.OVERLAPPED_BY, OverlappedByResolver()),  # other._starts first, overlaps start of interval
            # Returns: [before_overlap, overlapping_with_merged_metrics, after_overlap]

            # 6. Disjoint relationships (no overlap)
            (OverlapType.BEFORE, BeforeResolver()),  # Interval completely before other
            # Returns: [interval, other] (no merging needed)

            (OverlapType.AFTER, AfterResolver())  # Interval completely after other
            # Returns: [other, interval] (no merging needed)
        ])

        resolver = resolvers.get(relationship)
        if resolver is None:
            raise ValueError(f"No resolver found for relationship type: {relationship}")

        return resolver
