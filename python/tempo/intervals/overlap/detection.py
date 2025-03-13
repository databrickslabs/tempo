from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

if TYPE_CHECKING:
    from tempo.intervals.core.interval import Interval
else:
    # For runtime, use Any as a placeholder for Interval
    Interval = Any


class OverlapChecker(ABC):
    """Abstract base class for overlap checking strategies"""

    def check(self, interval: "Interval", other: "Interval") -> bool:
        """
        Base implementation of check that handles null safety
        before delegating to the specific implementation
        """
        # Check for None intervals
        if interval is None or other is None:
            return False

        # Delegate to the specific implementation
        return self._check_impl(interval, other)

    @abstractmethod
    def _check_impl(self, interval: "Interval", other: "Interval") -> bool:
        """Implementation of the specific overlap checking strategy"""
        pass

    def _check_boundary_values(
            self,
            interval: "Interval",
            other: "Interval",
            required_boundaries: Optional[Tuple[str, ...]] = None,
    ) -> bool:
        """
        Helper method to check if interval boundaries are valid.
        Returns False if any required boundary is None.

        Parameters:
        interval: The first interval to check
        other: The second interval to check
        required_boundaries: Tuple of boundaries to check, e.g. ('_start', '_end')
                             If None, checks all boundaries
        """
        if required_boundaries is None:
            required_boundaries = ("_start", "_end", "_start", "_end")

        # Check interval boundaries
        for boundary in required_boundaries[:2]:
            if getattr(interval, boundary, None) is None:
                return False

        # Check other boundaries
        for boundary in required_boundaries[2:]:
            if getattr(other, boundary, None) is None:
                return False

        return True

    def _safe_compare(self, comparison_fn: Callable[[], bool]) -> bool:
        """
        Safely executes a comparison function, handling type errors.
        Returns False if the comparison raises a TypeError.

        Parameters:
        comparison_fn: A lambda or function that performs the comparison

        Returns:
        bool: Result of the comparison, or False if the comparison fails
        """
        try:
            result = comparison_fn()
            return bool(result)  # Convert any pandas/numpy types to Python bool
        except TypeError:
            # Handle case where comparison fails due to type mismatch
            return False


class MetricsEquivalentChecker(OverlapChecker):
    """Checks if intervals have equivalent metrics and have any overlap or containment"""

    def _check_impl(self, interval: "Interval", other: "Interval") -> bool:
        # First check if both intervals have the same metric fields
        interval_metric_fields = set(interval.metric_fields)
        other_metric_fields = set(other.metric_fields)

        if interval_metric_fields != other_metric_fields:
            return False

        # If there are no metrics to compare, skip metrics comparison
        if not interval_metric_fields:
            metrics_equal = True
        else:
            # Check if metrics are equal using pandas equals() for proper NULL handling
            # Convert sequence of strings to List[str] for type safety
            interval_metrics = interval.data[list(interval.metric_fields)]
            other_metrics = other.data[list(other.metric_fields)]
            metrics_equal = interval_metrics.equals(other_metrics)

        if not metrics_equal:
            return False

        # Check all boundary values
        if not self._check_boundary_values(interval, other):
            return False

        # Extract internal values with null checks
        interval_start = getattr(interval._start, "internal_value", None)
        interval_end = getattr(interval._end, "internal_value", None)
        other_start = getattr(other._start, "internal_value", None)
        other_end = getattr(other._end, "internal_value", None)

        # Skip overlap check if any value is None
        if any(
                value is None
                for value in [interval_start, interval_end, other_start, other_end]
        ):
            return False

        # Then check if intervals overlap or one contains the other
        # Check overlap condition - interval overlaps with other
        overlap_check = self._check_overlap(
            interval_start, interval_end, other_start, other_end
        )
        if overlap_check:
            return True

        # If not overlapping, check if one contains the other
        containment_check = self._check_containment(
            interval_start, interval_end, other_start, other_end
        )
        return containment_check

    def _check_overlap(
            self, interval_start: Any, interval_end: Any, other_start: Any, other_end: Any
    ) -> bool:
        """Check if intervals overlap."""
        try:
            # Check if interval starts before other ends
            start_before_other_end = False
            try:
                start_before_other_end = interval_start < other_end
            except TypeError:
                return False

            if not start_before_other_end:
                return False

            # Check if interval ends after other starts
            end_after_other_start = False
            try:
                end_after_other_start = interval_end > other_start
            except TypeError:
                return False

            return end_after_other_start
        except Exception:
            return False

    def _check_containment(
            self, interval_start: Any, interval_end: Any, other_start: Any, other_end: Any
    ) -> bool:
        """Check if one interval contains the other."""
        try:
            # Check if interval contains other
            interval_contains_other = False
            try:
                start_before_or_equal = interval_start <= other_start
                end_after_or_equal = interval_end >= other_end
                interval_contains_other = start_before_or_equal and end_after_or_equal
            except TypeError:
                pass

            if interval_contains_other:
                return True

            # Check if other contains interval
            other_contains_interval = False
            try:
                start_before_or_equal = other_start <= interval_start
                end_after_or_equal = other_end >= interval_end
                other_contains_interval = start_before_or_equal and end_after_or_equal
            except TypeError:
                pass

            return other_contains_interval
        except Exception:
            return False


class BeforeChecker(OverlapChecker):
    """Checks if interval is completely before other"""

    def _check_impl(self, interval: "Interval", other: "Interval") -> bool:
        # Check required boundaries
        if not self._check_boundary_values(interval, other, ("_end", "_start")):
            return False

        # For type safety, ensure we're never comparing None values
        return self._safe_compare(lambda: interval._end < other._start)


class MeetsChecker(OverlapChecker):
    """Checks if interval._ends exactly where other._starts"""

    def _check_impl(self, interval: "Interval", other: "Interval") -> bool:
        # Check required boundaries
        if not self._check_boundary_values(interval, other, ("_end", "_start")):
            return False

        # For type safety
        return self._safe_compare(lambda: interval._end == other._start)


class OverlapsChecker(OverlapChecker):
    """Checks if interval overlaps the start of other"""

    def _check_impl(self, interval: "Interval", other: "Interval") -> bool:
        # Check all boundary values
        if not self._check_boundary_values(interval, other):
            return False

        # For type safety, execute each comparison separately
        try:
            cond1 = self._safe_compare(lambda: interval._start < other._start)
            cond2 = self._safe_compare(lambda: interval._end > other._start)
            cond3 = self._safe_compare(lambda: interval._end < other._end)
            return cond1 and cond2 and cond3
        except TypeError:
            return False


class StartsChecker(OverlapChecker):
    """Checks if interval._starts together with other but ends earlier"""

    def _check_impl(self, interval: "Interval", other: "Interval") -> bool:
        # Check all boundary values
        if not self._check_boundary_values(interval, other):
            return False

        # For type safety
        cond1 = self._safe_compare(lambda: interval._start == other._start)
        cond2 = self._safe_compare(lambda: interval._end < other._end)
        return cond1 and cond2


class DuringChecker(OverlapChecker):
    """Checks if interval is completely contained within other"""

    def _check_impl(self, interval: "Interval", other: "Interval") -> bool:
        # Check all boundary values
        if not self._check_boundary_values(interval, other):
            return False

        # For type safety
        cond1 = self._safe_compare(lambda: interval._start > other._start)
        cond2 = self._safe_compare(lambda: interval._end < other._end)
        return cond1 and cond2


class FinishesChecker(OverlapChecker):
    """Checks if interval._ends together with other but starts later"""

    def _check_impl(self, interval: "Interval", other: "Interval") -> bool:
        # Check all boundary values
        if not self._check_boundary_values(interval, other):
            return False

        # For type safety
        cond1 = self._safe_compare(lambda: interval._start > other._start)
        cond2 = self._safe_compare(lambda: interval._end == other._end)
        return cond1 and cond2


class EqualsChecker(OverlapChecker):
    """Checks if interval is identical to other"""

    def _check_impl(self, interval: "Interval", other: "Interval") -> bool:
        # Check all boundary values
        if not self._check_boundary_values(interval, other):
            return False

        # For type safety
        cond1 = self._safe_compare(lambda: interval._start == other._start)
        cond2 = self._safe_compare(lambda: interval._end == other._end)
        return cond1 and cond2


class ContainsChecker(OverlapChecker):
    """Checks if interval completely contains other"""

    def _check_impl(self, interval: "Interval", other: "Interval") -> bool:
        # Check all boundary values
        if not self._check_boundary_values(interval, other):
            return False

        # For type safety
        cond1 = self._safe_compare(lambda: interval._start < other._start)
        cond2 = self._safe_compare(lambda: interval._end > other._end)
        return cond1 and cond2


class StartedByChecker(OverlapChecker):
    """Checks if other._starts together with interval but ends earlier"""

    def _check_impl(self, interval: "Interval", other: "Interval") -> bool:
        # Check all boundary values
        if not self._check_boundary_values(interval, other):
            return False

        # For type safety
        cond1 = self._safe_compare(lambda: interval._start == other._start)
        cond2 = self._safe_compare(lambda: interval._end > other._end)
        return cond1 and cond2


class FinishedByChecker(OverlapChecker):
    """Checks if other._ends together with interval but starts later"""

    def _check_impl(self, interval: "Interval", other: "Interval") -> bool:
        # Check all boundary values
        if not self._check_boundary_values(interval, other):
            return False

        # For type safety
        cond1 = self._safe_compare(lambda: interval._end == other._end)
        cond2 = self._safe_compare(lambda: interval._start < other._start)
        return cond1 and cond2


class OverlappedByChecker(OverlapChecker):
    """Checks if other overlaps the start of interval"""

    def _check_impl(self, interval: "Interval", other: "Interval") -> bool:
        # Check all boundary values
        if not self._check_boundary_values(interval, other):
            return False

        # For type safety
        try:
            cond1 = self._safe_compare(lambda: other._start < interval._start)
            cond2 = self._safe_compare(lambda: other._end > interval._start)
            cond3 = self._safe_compare(lambda: other._end < interval._end)
            return cond1 and cond2 and cond3
        except TypeError:
            return False


class MetByChecker(OverlapChecker):
    """Checks if other._ends exactly where interval._starts"""

    def _check_impl(self, interval: "Interval", other: "Interval") -> bool:
        # Check required boundaries
        if not self._check_boundary_values(interval, other, ("_start", "_end")):
            return False

        # For type safety
        return self._safe_compare(lambda: other._end == interval._start)


class AfterChecker(OverlapChecker):
    """Checks if interval is completely after other"""

    def _check_impl(self, interval: "Interval", other: "Interval") -> bool:
        # Check required boundaries
        if not self._check_boundary_values(interval, other, ("_start", "_end")):
            return False

        # For type safety
        return self._safe_compare(lambda: interval._start > other._end)
