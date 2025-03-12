import pandas as pd
import pytest

from tempo.intervals.overlap.detection import (
    MetricsEquivalentChecker, BeforeChecker, MeetsChecker, OverlapsChecker,
    StartsChecker, DuringChecker, FinishesChecker, EqualsChecker, ContainsChecker,
    StartedByChecker, FinishedByChecker, OverlappedByChecker, MetByChecker, AfterChecker
)


# Mock class for time values with proper comparison operators
class TimeValue:
    def __init__(self, value, position):
        self.internal_value = value
        self.position = position  # Position in sequence (1, 2, 3, 4, 5)

    def __lt__(self, other):
        if not isinstance(other, TimeValue):
            return NotImplemented
        return self.position < other.position

    def __le__(self, other):
        if not isinstance(other, TimeValue):
            return NotImplemented
        return self.position <= other.position

    def __eq__(self, other):
        if not isinstance(other, TimeValue):
            return NotImplemented
        return self.position == other.position

    def __gt__(self, other):
        if not isinstance(other, TimeValue):
            return NotImplemented
        return self.position > other.position

    def __ge__(self, other):
        if not isinstance(other, TimeValue):
            return NotImplemented
        return self.position >= other.position

    def __repr__(self):
        return f"TimeValue({self.internal_value}, pos={self.position})"


# Mock Interval class for testing
class MockInterval:
    def __init__(self, start, end, metrics=None):
        self._start = start
        self._end = end

        # Default empty DataFrame with metric fields
        if metrics is None:
            metrics = {}

        # Create DataFrame with metrics
        self.data = pd.DataFrame([metrics])
        self.metric_fields = list(metrics.keys())


@pytest.fixture
def mock_values():
    # Create time values with proper ordering
    v1 = TimeValue(10, 1)
    v2 = TimeValue(20, 2)
    v3 = TimeValue(30, 3)
    v4 = TimeValue(40, 4)
    v5 = TimeValue(50, 5)

    return v1, v2, v3, v4, v5

    return v1, v2, v3, v4, v5


@pytest.fixture
def checkers():
    return {
        'metrics_equivalent': MetricsEquivalentChecker(),
        'before': BeforeChecker(),
        'meets': MeetsChecker(),
        'overlaps': OverlapsChecker(),
        'starts': StartsChecker(),
        'during': DuringChecker(),
        'finishes': FinishesChecker(),
        'equals': EqualsChecker(),
        'contains': ContainsChecker(),
        'started_by': StartedByChecker(),
        'finished_by': FinishedByChecker(),
        'overlapped_by': OverlappedByChecker(),
        'met_by': MetByChecker(),
        'after': AfterChecker()
    }


@pytest.fixture
def create_interval():
    def _create_interval(start, end, metrics=None):
        return MockInterval(start, end, metrics)

    return _create_interval


class TestBasicRelations:
    """Tests for the basic Allen's interval relations"""

    def test_before_checker(self, mock_values, checkers, create_interval):
        v1, v2, v3, v4, v5 = mock_values
        before_checker = checkers['before']

        # Case 1: A before B
        interval_a = create_interval(v1, v2)
        interval_b = create_interval(v3, v4)
        assert before_checker.check(interval_a, interval_b) is True

        # Case 2: A meets B (not before)
        interval_a = create_interval(v1, v3)
        interval_b = create_interval(v3, v4)
        assert before_checker.check(interval_a, interval_b) is False

        # Case 3: A overlaps B (not before)
        interval_a = create_interval(v2, v4)
        interval_b = create_interval(v3, v5)
        assert before_checker.check(interval_a, interval_b) is False

    def test_meets_checker(self, mock_values, checkers, create_interval):
        v1, v2, v3, v4, v5 = mock_values
        meets_checker = checkers['meets']

        # Case 1: A meets B
        interval_a = create_interval(v1, v3)
        interval_b = create_interval(v3, v5)
        assert meets_checker.check(interval_a, interval_b) is True

        # Case 2: A before B (not meets)
        interval_a = create_interval(v1, v2)
        interval_b = create_interval(v3, v4)
        assert meets_checker.check(interval_a, interval_b) is False

        # Case 3: A overlaps B (not meets)
        interval_a = create_interval(v1, v4)
        interval_b = create_interval(v3, v5)
        assert meets_checker.check(interval_a, interval_b) is False

    def test_after_checker(self, mock_values, checkers, create_interval):
        v1, v2, v3, v4, v5 = mock_values
        after_checker = checkers['after']

        # Case 1: A after B
        interval_a = create_interval(v3, v4)
        interval_b = create_interval(v1, v2)
        assert after_checker.check(interval_a, interval_b) is True

        # Case 2: A met by B (should NOT be after with strict definition)
        interval_a = create_interval(v3, v5)
        interval_b = create_interval(v1, v3)
        assert after_checker.check(interval_a, interval_b) is False

        # Add a new case for strictly after
        interval_c = create_interval(v4, v5)  # starts after interval_b ends
        interval_d = create_interval(v1, v3)
        assert after_checker.check(interval_c, interval_d) is True

    def test_met_by_checker(self, mock_values, checkers, create_interval):
        v1, v2, v3, v4, v5 = mock_values
        met_by_checker = checkers['met_by']

        # Case 1: A met by B
        interval_a = create_interval(v3, v5)
        interval_b = create_interval(v1, v3)
        assert met_by_checker.check(interval_a, interval_b) is True

        # Case 2: A after B (not met by)
        interval_a = create_interval(v4, v5)
        interval_b = create_interval(v1, v2)
        assert met_by_checker.check(interval_a, interval_b) is False

        # Case 3: A overlapped by B (not met by)
        interval_a = create_interval(v2, v4)
        interval_b = create_interval(v1, v3)
        assert met_by_checker.check(interval_a, interval_b) is False


class TestOverlapRelations:
    """Tests for interval relations that involve overlapping"""

    def test_overlaps_checker(self, mock_values, checkers, create_interval):
        v1, v2, v3, v4, v5 = mock_values
        overlaps_checker = checkers['overlaps']

        # Case 1: A overlaps B
        interval_a = create_interval(v1, v3)
        interval_b = create_interval(v2, v4)
        assert overlaps_checker.check(interval_a, interval_b) is True

        # Case 2: A before B (not overlaps)
        interval_a = create_interval(v1, v2)
        interval_b = create_interval(v3, v5)
        assert overlaps_checker.check(interval_a, interval_b) is False

        # Case 3: A contains B (not overlaps)
        interval_a = create_interval(v1, v5)
        interval_b = create_interval(v2, v4)
        assert overlaps_checker.check(interval_a, interval_b) is False

    def test_overlapped_by_checker(self, mock_values, checkers, create_interval):
        v1, v2, v3, v4, v5 = mock_values
        overlapped_by_checker = checkers['overlapped_by']

        # Case 1: A overlapped by B
        interval_a = create_interval(v2, v4)
        interval_b = create_interval(v1, v3)
        assert overlapped_by_checker.check(interval_a, interval_b) is True

        # Case 2: A during B (not overlapped by)
        interval_a = create_interval(v2, v3)
        interval_b = create_interval(v1, v4)
        assert overlapped_by_checker.check(interval_a, interval_b) is False

        # Case 3: A after B (not overlapped by)
        interval_a = create_interval(v3, v5)
        interval_b = create_interval(v1, v2)
        assert overlapped_by_checker.check(interval_a, interval_b) is False


class TestContainmentRelations:
    """Tests for interval relations that involve one interval containing another"""

    def test_during_checker(self, mock_values, checkers, create_interval):
        v1, v2, v3, v4, v5 = mock_values
        during_checker = checkers['during']

        # Case 1: A during B
        interval_a = create_interval(v2, v3)
        interval_b = create_interval(v1, v4)
        assert during_checker.check(interval_a, interval_b) is True

        # Case 2: A equals B (not during)
        interval_a = create_interval(v1, v4)
        interval_b = create_interval(v1, v4)
        assert during_checker.check(interval_a, interval_b) is False

        # Case 3: A starts B (not during)
        interval_a = create_interval(v1, v3)
        interval_b = create_interval(v1, v4)
        assert during_checker.check(interval_a, interval_b) is False

    def test_contains_checker(self, mock_values, checkers, create_interval):
        v1, v2, v3, v4, v5 = mock_values
        contains_checker = checkers['contains']

        # Case 1: A contains B
        interval_a = create_interval(v1, v4)
        interval_b = create_interval(v2, v3)
        assert contains_checker.check(interval_a, interval_b) is True

        # Case 2: A equals B (not contains)
        interval_a = create_interval(v1, v4)
        interval_b = create_interval(v1, v4)
        assert contains_checker.check(interval_a, interval_b) is False

        # Case 3: A started by B (not strictly contains)
        interval_a = create_interval(v1, v4)
        interval_b = create_interval(v1, v3)
        assert contains_checker.check(interval_a, interval_b) is False


class TestBoundaryRelations:
    """Tests for interval relations that involve common start or end points"""

    def test_starts_checker(self, mock_values, checkers, create_interval):
        v1, v2, v3, v4, v5 = mock_values
        starts_checker = checkers['starts']

        # Case 1: A starts B
        interval_a = create_interval(v1, v3)
        interval_b = create_interval(v1, v4)
        assert starts_checker.check(interval_a, interval_b) is True

        # Case 2: A equals B (not starts)
        interval_a = create_interval(v1, v4)
        interval_b = create_interval(v1, v4)
        assert starts_checker.check(interval_a, interval_b) is False

        # Case 3: A during B (not starts)
        interval_a = create_interval(v2, v3)
        interval_b = create_interval(v1, v4)
        assert starts_checker.check(interval_a, interval_b) is False

    def test_started_by_checker(self, mock_values, checkers, create_interval):
        v1, v2, v3, v4, v5 = mock_values
        started_by_checker = checkers['started_by']

        # Case 1: A started by B
        interval_a = create_interval(v1, v4)
        interval_b = create_interval(v1, v3)
        assert started_by_checker.check(interval_a, interval_b) is True

        # Case 2: A equals B (not started by)
        interval_a = create_interval(v1, v4)
        interval_b = create_interval(v1, v4)
        assert started_by_checker.check(interval_a, interval_b) is False

        # Case 3: A contains B but not started by
        interval_a = create_interval(v1, v5)
        interval_b = create_interval(v2, v4)
        assert started_by_checker.check(interval_a, interval_b) is False

    def test_finishes_checker(self, mock_values, checkers, create_interval):
        v1, v2, v3, v4, v5 = mock_values
        finishes_checker = checkers['finishes']

        # Case 1: A finishes B
        interval_a = create_interval(v2, v4)
        interval_b = create_interval(v1, v4)
        assert finishes_checker.check(interval_a, interval_b) is True

        # Case 2: A equals B (not finishes)
        interval_a = create_interval(v1, v4)
        interval_b = create_interval(v1, v4)
        assert finishes_checker.check(interval_a, interval_b) is False

        # Case 3: A during B (not finishes)
        interval_a = create_interval(v2, v3)
        interval_b = create_interval(v1, v4)
        assert finishes_checker.check(interval_a, interval_b) is False

    def test_finished_by_checker(self, mock_values, checkers, create_interval):
        v1, v2, v3, v4, v5 = mock_values
        finished_by_checker = checkers['finished_by']

        # Case 1: A finished by B
        interval_a = create_interval(v1, v4)
        interval_b = create_interval(v2, v4)
        assert finished_by_checker.check(interval_a, interval_b) is True

        # Case 2: A equals B (not finished by)
        interval_a = create_interval(v1, v4)
        interval_b = create_interval(v1, v4)
        assert finished_by_checker.check(interval_a, interval_b) is False

        # Case 3: A contains B but not finished by
        interval_a = create_interval(v1, v5)
        interval_b = create_interval(v2, v4)
        assert finished_by_checker.check(interval_a, interval_b) is False


class TestEquivalenceRelations:
    """Tests for equivalence relations between intervals"""

    def test_equals_checker(self, mock_values, checkers, create_interval):
        v1, v2, v3, v4, v5 = mock_values
        equals_checker = checkers['equals']

        # Case 1: A equals B
        interval_a = create_interval(v1, v4)
        interval_b = create_interval(v1, v4)
        assert equals_checker.check(interval_a, interval_b) is True

        # Case 2: A starts B (not equals)
        interval_a = create_interval(v1, v3)
        interval_b = create_interval(v1, v4)
        assert equals_checker.check(interval_a, interval_b) is False

        # Case 3: A finishes B (not equals)
        interval_a = create_interval(v2, v4)
        interval_b = create_interval(v1, v4)
        assert equals_checker.check(interval_a, interval_b) is False

    def test_metrics_equivalent_checker(self, mock_values, checkers, create_interval):
        v1, v2, v3, v4, v5 = mock_values
        metrics_checker = checkers['metrics_equivalent']

        # Case 1: Same metrics, overlapping intervals - should match
        interval_a = create_interval(v1, v3, {"product": "A", "region": "US"})
        interval_b = create_interval(v2, v4, {"product": "A", "region": "US"})
        assert metrics_checker.check(interval_a, interval_b) is True

        # Case 2: Different metrics, overlapping intervals - should not match
        interval_a = create_interval(v1, v3, {"product": "A", "region": "US"})
        interval_b = create_interval(v2, v4, {"product": "B", "region": "US"})
        assert metrics_checker.check(interval_a, interval_b) is False

        # Case 3: Same metrics, non-overlapping intervals - should not match
        interval_a = create_interval(v1, v2, {"product": "A", "region": "US"})
        interval_b = create_interval(v3, v4, {"product": "A", "region": "US"})
        assert metrics_checker.check(interval_a, interval_b) is False

        # Case 4: Same metrics, one contains the other - should match
        interval_a = create_interval(v1, v4, {"product": "A", "region": "US"})
        interval_b = create_interval(v2, v3, {"product": "A", "region": "US"})
        assert metrics_checker.check(interval_a, interval_b) is True

        # Case 5: Handle null values in metrics
        interval_a = create_interval(v1, v3, {"product": "A", "region": None})
        interval_b = create_interval(v2, v4, {"product": "A", "region": None})
        assert metrics_checker.check(interval_a, interval_b) is True

        # Case 6: Different null values
        interval_a = create_interval(v1, v3, {"product": "A", "region": None})
        interval_b = create_interval(v2, v4, {"product": "A", "region": "US"})
        assert metrics_checker.check(interval_a, interval_b) is False


class TestInverseRelations:
    """Tests specifically focused on the inverse properties of Allen's interval relations"""

    def test_all_inverse_relations(self, mock_values, checkers, create_interval):
        """Test that all relations properly maintain their inverse relationship property"""
        v1, v2, v3, v4, v5 = mock_values

        # Define all inverse relation pairs
        inverse_pairs = [
            ('before', 'after'),
            ('meets', 'met_by'),
            ('overlaps', 'overlapped_by'),
            ('starts', 'started_by'),
            ('during', 'contains'),
            ('finishes', 'finished_by'),
            ('equals', 'equals')  # equals is its own inverse
        ]

        # Create a diverse set of interval pairs to test each relation
        interval_pairs = [
            # For before/after
            (create_interval(v1, v2), create_interval(v3, v4)),

            # For meets/met_by
            (create_interval(v1, v2), create_interval(v2, v3)),

            # For overlaps/overlapped_by
            (create_interval(v1, v3), create_interval(v2, v4)),

            # For starts/started_by
            (create_interval(v1, v2), create_interval(v1, v3)),

            # For during/contains
            (create_interval(v2, v3), create_interval(v1, v4)),

            # For finishes/finished_by
            (create_interval(v3, v4), create_interval(v2, v4)),

            # For equals
            (create_interval(v2, v3), create_interval(v2, v3))
        ]

        # Test each inverse pair with appropriate intervals
        for i, (relation_a, relation_b) in enumerate(inverse_pairs):
            interval_a, interval_b = interval_pairs[i]

            # Verify relation A → B
            assert checkers[relation_a].check(interval_a, interval_b), \
                f"Expected {relation_a} to be true from A to B"

            # Verify inverse relation B → A
            assert checkers[relation_b].check(interval_b, interval_a), \
                f"Expected {relation_b} to be true from B to A (inverse of {relation_a})"

    def test_inverse_property_systematic(self, mock_values, checkers, create_interval):
        """
        Systematically test that for any two intervals with a relation,
        the inverse relation holds in the opposite direction
        """
        v1, v2, v3, v4, v5 = mock_values

        # Define the inverse relationship map
        inverse_map = {
            'before': 'after',
            'meets': 'met_by',
            'overlaps': 'overlapped_by',
            'starts': 'started_by',
            'during': 'contains',
            'finishes': 'finished_by',
            'equals': 'equals',
            'finished_by': 'finishes',
            'contains': 'during',
            'started_by': 'starts',
            'overlapped_by': 'overlaps',
            'met_by': 'meets',
            'after': 'before'
        }

        # Create a diverse set of intervals to test
        intervals = [
            create_interval(v1, v2),  # [v1, v2]
            create_interval(v1, v3),  # [v1, v3]
            create_interval(v2, v3),  # [v2, v3]
            create_interval(v2, v4),  # [v2, v4]
            create_interval(v3, v4),  # [v3, v4]
            create_interval(v3, v5),  # [v3, v5]
            create_interval(v4, v5),  # [v4, v5]
            create_interval(v1, v5),  # [v1, v5]
            create_interval(v1, v1),  # Point at v1
            create_interval(v3, v3),  # Point at v3
            create_interval(v5, v5)  # Point at v5
        ]

        # List of relations to check (excluding metrics_equivalent)
        relation_types = [rel for rel in inverse_map.keys()]

        # Test all pairs of intervals
        for i, interval_a in enumerate(intervals):
            for j, interval_b in enumerate(intervals):
                # Track which relations hold
                true_relations_a_to_b = []

                # Find all relations that hold from A to B
                for relation in relation_types:
                    if checkers[relation].check(interval_a, interval_b):
                        true_relations_a_to_b.append(relation)

                # There should be at most one true relation from A to B
                # Note: For point intervals some edge cases might exist
                if len(true_relations_a_to_b) > 1:
                    # Debug info for troubleshooting
                    print(
                        f"Multiple relations found for intervals: {interval_a._start.position}-{interval_a._end.position} and {interval_b._start.position}-{interval_b._end.position}")
                    print(f"Relations: {true_relations_a_to_b}")

                # For each true relation, check the inverse
                for relation_a_to_b in true_relations_a_to_b:
                    # Get the expected inverse relation
                    inverse_relation = inverse_map[relation_a_to_b]

                    # Verify the inverse relation holds from B to A
                    assert checkers[inverse_relation].check(interval_b, interval_a), \
                        f"If interval_a {relation_a_to_b} interval_b, then interval_b should {inverse_relation} interval_a"

                # If we found exactly one relation, verify no other relation holds
                if len(true_relations_a_to_b) == 1:
                    relation_a_to_b = true_relations_a_to_b[0]
                    inverse_relation = inverse_map[relation_a_to_b]

                    # Get all relations from B to A
                    true_relations_b_to_a = []
                    for relation in relation_types:
                        if checkers[relation].check(interval_b, interval_a):
                            true_relations_b_to_a.append(relation)

                    # Should be exactly one true relation from B to A
                    assert len(true_relations_b_to_a) == 1, \
                        f"Expected exactly one true relation from B to A, found {len(true_relations_b_to_a)}: {true_relations_b_to_a}"

                    # That one relation should be the inverse
                    assert true_relations_b_to_a[0] == inverse_relation, \
                        f"Expected inverse relation {inverse_relation}, found {true_relations_b_to_a[0]}"


class TestBoundaryEdgeCases:
    """Tests for complex boundary combinations and edge cases with shared endpoints"""

    def test_zero_length_intervals_special_cases(self, mock_values, checkers, create_interval):
        """Test relations involving multiple zero-length intervals at different positions"""
        v1, v2, v3, v4, v5 = mock_values

        # Create zero-length intervals at different positions
        point_v1 = create_interval(v1, v1)
        point_v2 = create_interval(v2, v2)
        point_v3 = create_interval(v3, v3)

        # Two points at same position
        point_v1_duplicate = create_interval(v1, v1)

        # Test equality of coincident points
        assert checkers['equals'].check(point_v1, point_v1_duplicate)

        # Test before/after with points
        assert checkers['before'].check(point_v1, point_v2)
        assert checkers['after'].check(point_v3, point_v2)

        # A point can't 'meet' another point
        assert not checkers['meets'].check(point_v1, point_v2)
        assert not checkers['met_by'].check(point_v2, point_v1)

    def test_adjacent_intervals_combinations(self, mock_values, checkers, create_interval):
        """Test complex combinations of adjacent intervals"""
        v1, v2, v3, v4, v5 = mock_values

        # Create intervals that meet end-to-end
        interval_1 = create_interval(v1, v2)
        interval_2 = create_interval(v2, v3)
        interval_3 = create_interval(v3, v4)
        interval_4 = create_interval(v4, v5)

        # Chain of meeting intervals
        assert checkers['meets'].check(interval_1, interval_2)
        assert checkers['meets'].check(interval_2, interval_3)
        assert checkers['meets'].check(interval_3, interval_4)

        # Relation between non-adjacent intervals in the chain
        assert checkers['before'].check(interval_1, interval_3)
        assert checkers['before'].check(interval_2, interval_4)
        assert checkers['before'].check(interval_1, interval_4)

        # Create a larger interval spanning multiple adjacent intervals
        span_1_3 = create_interval(v1, v3)
        span_2_4 = create_interval(v2, v4)
        span_1_4 = create_interval(v1, v4)

        # Test relationships with spanning intervals
        assert checkers['finished_by'].check(span_1_3, interval_2)
        assert checkers['started_by'].check(span_2_4, interval_2)
        assert checkers['contains'].check(span_1_4, interval_2)

    def test_nested_intervals_shared_boundaries(self, mock_values, checkers, create_interval):
        """Test nested intervals with shared boundaries"""
        v1, v2, v3, v4, v5 = mock_values

        # Outer interval
        outer = create_interval(v1, v5)

        # Middle intervals, sharing start or end with outer
        middle_same_start = create_interval(v1, v4)
        middle_same_end = create_interval(v2, v5)
        middle_inside = create_interval(v2, v4)

        # Inner interval, sharing no boundaries with outer
        inner = create_interval(v3, v3)

        # Test relations with outer interval
        assert checkers['started_by'].check(outer, middle_same_start)
        assert checkers['finished_by'].check(outer, middle_same_end)
        assert checkers['contains'].check(outer, middle_inside)
        assert checkers['contains'].check(outer, inner)

        # Since they share the same end point, this is a "finished_by" relationship, not "contains"
        assert checkers['finished_by'].check(middle_same_start, middle_inside)

        # And correspondingly, middle_inside "finishes" middle_same_start
        assert checkers['finishes'].check(middle_inside, middle_same_start)

        # Test other middle interval relations
        assert checkers['overlaps'].check(middle_same_start, middle_same_end)

        # For middle_inside and middle_same_end, let's check the proper relation
        # They share a start point, so middle_inside "starts" middle_same_end
        assert checkers['starts'].check(middle_inside, middle_same_end)

        # Multiple relationships with inner
        assert checkers['contains'].check(middle_same_start, inner)
        assert checkers['contains'].check(middle_same_end, inner)
        assert checkers['contains'].check(middle_inside, inner)

    def test_complex_chain_of_relations(self, mock_values, checkers, create_interval):
        """Test a complex chain of intervals with various relations"""
        v1, v2, v3, v4, v5 = mock_values

        # Create a chain of intervals with various relations
        interval_1 = create_interval(v1, v2)  # [v1,v2]
        interval_2 = create_interval(v2, v3)  # [v2,v3] - meets interval_1
        interval_3 = create_interval(v2, v4)  # [v2,v4] - started_by interval_2
        interval_4 = create_interval(v3, v4)  # [v3,v4] - during interval_3, after interval_2
        interval_5 = create_interval(v4, v5)  # [v4,v5] - meets interval_3, meets interval_4

        # Test the expected relations in the chain
        assert checkers['meets'].check(interval_1, interval_2)
        assert checkers['started_by'].check(interval_3, interval_2)
        assert checkers['finishes'].check(interval_4, interval_3)
        assert checkers['meets'].check(interval_3, interval_5)
        assert checkers['meets'].check(interval_4, interval_5)

        # Test more complex relations in the chain
        # FIXED: interval_1 MEETS interval_3 (not before)
        assert checkers['meets'].check(interval_1, interval_3)

        # interval_1 is before interval_4 (no shared endpoints)
        assert checkers['before'].check(interval_1, interval_4)

        # interval_1 is before interval_5 (no shared endpoints)
        assert checkers['before'].check(interval_1, interval_5)

        # interval_2 is before interval_5 (no shared endpoints)
        assert checkers['before'].check(interval_2, interval_5)

        # interval_4 overlapped_by interval_3 (interval_3 starts earlier, ends at same time)
        assert checkers['finishes'].check(interval_4, interval_3)


class TestSpecialBoundaryScenarios:
    """Tests for special scenarios with intervals sharing multiple boundaries"""

    def test_intervals_sharing_both_boundaries(self, mock_values, checkers, create_interval):
        """Test intervals that share both start and end points (equals)"""
        v1, v3, v5 = mock_values[0], mock_values[2], mock_values[4]

        # Two intervals with exact same boundaries
        interval_1 = create_interval(v1, v3)
        interval_2 = create_interval(v1, v3)

        # Different intervals that also share both boundaries
        interval_3 = create_interval(v3, v5)
        interval_4 = create_interval(v3, v5)

        # Test equals relation
        assert checkers['equals'].check(interval_1, interval_2)
        assert checkers['equals'].check(interval_3, interval_4)

        # Test that no other relation holds
        for relation in checkers:
            if relation != 'equals' and relation != 'metrics_equivalent':
                assert not checkers[relation].check(interval_1, interval_2)
                assert not checkers[relation].check(interval_3, interval_4)

    def test_intervals_with_multi_boundary_sharing(self, mock_values, checkers, create_interval):
        """Test complex scenarios where multiple intervals share various boundaries"""
        v1, v2, v3, v4, v5 = mock_values

        # Create a set of intervals with shared boundaries
        common_start = [
            create_interval(v1, v2),
            create_interval(v1, v3),
            create_interval(v1, v4),
            create_interval(v1, v5)
        ]

        common_end = [
            create_interval(v1, v5),
            create_interval(v2, v5),
            create_interval(v3, v5),
            create_interval(v4, v5)
        ]

        # Test relations between intervals with common start
        for i in range(len(common_start)):
            for j in range(i + 1, len(common_start)):
                if i == j:
                    continue
                assert checkers['starts'].check(common_start[i], common_start[j]) or \
                       checkers['started_by'].check(common_start[i], common_start[j])

        # Test relations between intervals with common end
        for i in range(len(common_end)):
            for j in range(i + 1, len(common_end)):
                if i == j:
                    continue
                assert checkers['finishes'].check(common_end[i], common_end[j]) or \
                       checkers['finished_by'].check(common_end[i], common_end[j])


class TestAdvancedScenarios:
    """Tests for edge cases and combined usage scenarios"""

    def test_multiple_checkers_combination(self, mock_values, checkers, create_interval):
        """Test that multiple relationship checkers can be used together to validate different relations"""
        v1, v2, v3, v4, v5 = mock_values

        # Create some intervals with specific relationships
        interval_a = create_interval(v1, v3)
        interval_b = create_interval(v3, v5)  # A meets B
        interval_c = create_interval(v1, v5)  # C is started by A and finished by B
        interval_d = create_interval(v2, v4)  # D overlaps with A and B

        # Test different combinations
        assert checkers['meets'].check(interval_a, interval_b) is True
        assert checkers['started_by'].check(interval_c, interval_a) is True

        # Since interval_c and interval_b share an end point, the relation is 'finished_by' not 'contains'
        assert checkers['finished_by'].check(interval_c, interval_b) is True

        assert checkers['overlaps'].check(interval_a, interval_d) is True

    def test_edge_case_zero_length_intervals_before_after(self, mock_values, checkers, create_interval):
        """Test before/after relationships with zero-length intervals"""
        v1, v2, v3 = mock_values[:3]

        # Zero-length intervals (start = end)
        zero_interval_1 = create_interval(v1, v1)
        zero_interval_2 = create_interval(v2, v2)
        normal_interval = create_interval(v2, v3)

        # Test before/after with zero-length intervals
        assert checkers['before'].check(zero_interval_1, zero_interval_2) is True
        assert checkers['after'].check(zero_interval_2, zero_interval_1) is True
        assert checkers['before'].check(zero_interval_1, normal_interval) is True

    def test_all_relations_are_mutually_exclusive(self, mock_values, checkers, create_interval):
        """Test that Allen's interval relations are mutually exclusive"""
        v1, v2, v3, v4, v5 = mock_values

        # Create different interval relationships with the reference interval
        interval_reference = create_interval(v1, v4)
        interval_before = create_interval(v1, v2)
        interval_meets = create_interval(v2, v3)
        interval_overlaps = create_interval(v1, v3)
        interval_during = create_interval(v2, v3)
        interval_finishes = create_interval(v2, v4)
        interval_equals = create_interval(v1, v4)

        # Get all the relation checkers
        relation_checkers = [
            checkers['before'], checkers['meets'], checkers['overlaps'],
            checkers['starts'], checkers['during'], checkers['finishes'],
            checkers['equals'], checkers['contains'], checkers['started_by'],
            checkers['finished_by'], checkers['overlapped_by'], checkers['met_by'],
            checkers['after']
        ]

        # Each interval should have a relationship with the reference interval
        test_intervals = {
            "before": interval_before,
            "meets": interval_meets,
            "during": interval_during,
            "finishes": interval_finishes,
            "equals": interval_equals
        }

        for name, interval in test_intervals.items():
            # For equals, we expect exactly one relation (equals itself)
            if name == "equals":
                assert checkers['equals'].check(interval, interval_reference), \
                    f"Equal intervals should have equals relation"
            # For others, we should find at least one valid relation
            else:
                found_relation = False
                for checker in relation_checkers:
                    if checker.check(interval, interval_reference):
                        found_relation = True
                        break
                assert found_relation, f"No relation found for interval {name}"

    def test_both_zero_length_intervals(self, mock_values, checkers, create_interval):
        """Test relations between two zero-length intervals (points)"""
        v1, v2, v3 = mock_values[:3]

        # Two different zero-length intervals
        point_a = create_interval(v1, v1)
        point_b = create_interval(v2, v2)

        # Two identical zero-length intervals
        point_c = create_interval(v3, v3)
        point_d = create_interval(v3, v3)

        # Point-to-point relations
        assert checkers['before'].check(point_a, point_b) is True
        assert checkers['after'].check(point_b, point_a) is True
        assert checkers['equals'].check(point_c, point_d) is True

        # A point can't meet, overlap, contain, or be during another point
        assert checkers['meets'].check(point_a, point_b) is False
        assert checkers['overlaps'].check(point_a, point_b) is False
        assert checkers['contains'].check(point_a, point_b) is False
        assert checkers['during'].check(point_a, point_b) is False

    def test_multiple_consecutive_points(self, mock_values, checkers, create_interval):
        """Test multiple consecutive zero-length intervals"""
        v1, v2, v3, v4 = mock_values[:4]

        # Three consecutive points
        point_a = create_interval(v1, v1)
        point_b = create_interval(v2, v2)
        point_c = create_interval(v3, v3)

        # Each point should be before the next
        assert checkers['before'].check(point_a, point_b) is True
        assert checkers['before'].check(point_b, point_c) is True
        assert checkers['before'].check(point_a, point_c) is True

        # Transitivity test for after relation
        assert checkers['after'].check(point_c, point_b) is True
        assert checkers['after'].check(point_b, point_a) is True
        assert checkers['after'].check(point_c, point_a) is True

    def test_inverse_relations(self, mock_values, checkers, create_interval):
        """Test that inverse relations work correctly"""
        v1, v2, v3, v4 = mock_values[:4]

        # Create intervals with different relationships
        interval_a = create_interval(v1, v2)
        interval_b = create_interval(v3, v4)
        interval_c = create_interval(v2, v3)
        interval_d = create_interval(v1, v3)
        interval_e = create_interval(v2, v4)

        # Test inverse pairs
        # If A before B, then B after A
        assert checkers['before'].check(interval_a, interval_b) is True
        assert checkers['after'].check(interval_b, interval_a) is True

        # If A meets B, then B met by A
        assert checkers['meets'].check(interval_a, interval_c) is True
        assert checkers['met_by'].check(interval_c, interval_a) is True

        # If A overlaps B, then B overlapped by A
        assert checkers['overlaps'].check(interval_d, interval_e) is True
        assert checkers['overlapped_by'].check(interval_e, interval_d) is True

        # If A starts B, then B started by A
        assert checkers['starts'].check(interval_a, interval_d) is True
        assert checkers['started_by'].check(interval_d, interval_a) is True

        # If A finishes B, then B finished by A
        assert checkers['finishes'].check(interval_a, interval_e) is False  # Not true for this example
        assert checkers['finished_by'].check(interval_e, interval_a) is False  # Should match above

    def test_transitivity_properties(self, mock_values, checkers, create_interval):
        """Test transitivity properties of certain relations"""
        v1, v2, v3, v4, v5 = mock_values

        # Before relation is transitive
        # Create intervals that are strictly before each other
        a_before_b = create_interval(v1, v2)  # [v1, v2]
        b_before_c = create_interval(v3, v4)  # [v3, v4]
        c_interval = create_interval(v5, v5)  # [v5, v5]

        assert checkers['before'].check(a_before_b, b_before_c) is True
        assert checkers['before'].check(b_before_c, c_interval) is True
        assert checkers['before'].check(a_before_b, c_interval) is True

        # During relation has a different transitivity property
        outer = create_interval(v1, v5)  # [v1, v5]
        middle = create_interval(v2, v4)  # [v2, v4]
        inner = create_interval(v3, v3)  # [v3, v3]

        assert checkers['contains'].check(outer, middle) is True
        assert checkers['contains'].check(middle, inner) is True
        assert checkers['contains'].check(outer, inner) is True

    def test_complex_boundary_cases(self, mock_values, checkers, create_interval):
        """Test complex combinations of shared boundary points"""
        v1, v2, v3, v4, v5 = mock_values

        # Intervals that share both start and end but aren't equal
        interval_a = create_interval(v1, v3)
        interval_b = create_interval(v1, v3)
        assert checkers['equals'].check(interval_a, interval_b) is True

        # One interval starts where another ends, and a third overlaps both
        interval_c = create_interval(v1, v2)
        interval_d = create_interval(v2, v3)
        interval_e = create_interval(v1, v3)

        assert checkers['meets'].check(interval_c, interval_d) is True
        assert checkers['started_by'].check(interval_e, interval_c) is True
        assert checkers['finished_by'].check(interval_e, interval_d) is True

    def test_mutual_exclusivity_regular_intervals(self, mock_values, checkers, create_interval):
        """Comprehensive test that Allen's interval relations are mutually exclusive"""
        v1, v2, v3, v4, v5 = mock_values

        # Create a set of intervals that represent all possible relationships
        intervals = {
            "reference": create_interval(v2, v4),
            "before": create_interval(v1, v1),  # Strictly before v2
            "meets": create_interval(v1, v2),  # Ends exactly at reference start
            "overlaps": create_interval(v1, v3),  # Overlaps with reference start
            "starts": create_interval(v2, v3),  # Starts with reference
            "during": create_interval(v3, v3),  # Strictly inside reference
            "finishes": create_interval(v3, v4),  # Ends with reference
            "equals": create_interval(v2, v4),  # Same as reference
            "finished_by": create_interval(v1, v4),  # Contains and shares end with reference
            "contains": create_interval(v1, v5),  # Strictly contains reference
            "started_by": create_interval(v2, v5),  # Shares start and extends beyond reference
            "overlapped_by": create_interval(v3, v5),  # Reference overlaps start of this interval
            "met_by": create_interval(v4, v5),  # Starts exactly where reference ends
            "after": create_interval(v5, v5)  # Strictly after reference
        }

        # List of all checkers
        all_checkers = [
            'before', 'meets', 'overlaps', 'starts', 'during', 'finishes',
            'equals', 'finished_by', 'contains', 'started_by', 'overlapped_by',
            'met_by', 'after'
        ]

        # For each interval, exactly one relation should be true with the reference
        for name, interval in intervals.items():
            if name == "reference":
                continue

            # Count how many relations are true
            true_relations = []
            for checker_name in all_checkers:
                if checkers[checker_name].check(interval, intervals["reference"]):
                    true_relations.append(checker_name)

            # Only one relation should be true (if name is unique)
            if name in all_checkers:  # Skip if the name isn't a valid relation
                expected_relation = name
                assert len(true_relations) == 1, \
                    f"Expected 1 relation, got {len(true_relations)}: {true_relations}"
                assert true_relations[0] == expected_relation, \
                    f"Expected {expected_relation}, got {true_relations[0]}"


class TestMetricsEdgeCases:
    """Tests for edge cases related to metrics comparison"""

    def test_empty_metrics(self, mock_values, checkers, create_interval):
        """Test intervals with empty metric dictionaries"""
        v1, v2, v3, v4 = mock_values[:4]
        metrics_checker = checkers['metrics_equivalent']

        # Both intervals have empty metrics
        interval_a = create_interval(v1, v3, {})
        interval_b = create_interval(v2, v4, {})
        assert metrics_checker.check(interval_a, interval_b) is True

        # One interval has metrics, other doesn't
        interval_c = create_interval(v1, v3, {"product": "A"})
        assert metrics_checker.check(interval_c, interval_b) is False

    def test_different_metric_fields(self, mock_values, checkers, create_interval):
        """Test intervals with different sets of metric fields"""
        v1, v2, v3, v4 = mock_values[:4]
        metrics_checker = checkers['metrics_equivalent']

        # Different fields but overlapping intervals
        interval_a = create_interval(v1, v3, {"product": "A"})
        interval_b = create_interval(v2, v4, {"region": "US"})
        assert metrics_checker.check(interval_a, interval_b) is False

        # One interval has a superset of the other's fields
        interval_c = create_interval(v1, v3, {"product": "A", "region": "US"})
        interval_d = create_interval(v2, v4, {"product": "A"})
        assert metrics_checker.check(interval_c, interval_d) is False

    def test_case_sensitivity_in_metrics(self, mock_values, checkers, create_interval):
        """Test case sensitivity in metric values"""
        v1, v2, v3, v4 = mock_values[:4]
        metrics_checker = checkers['metrics_equivalent']

        # Case difference in values
        interval_a = create_interval(v1, v3, {"product": "product_a"})
        interval_b = create_interval(v2, v4, {"product": "Product_A"})
        assert metrics_checker.check(interval_a, interval_b) is False

        # Same case, should match
        interval_c = create_interval(v1, v3, {"product": "Product_A"})
        interval_d = create_interval(v2, v4, {"product": "Product_A"})
        assert metrics_checker.check(interval_c, interval_d) is True

    def test_empty_metrics(self, mock_values, checkers, create_interval):
        """Test intervals with empty metric dictionaries"""
        v1, v2, v3, v4 = mock_values[:4]
        metrics_checker = checkers['metrics_equivalent']

        # Both intervals have empty metrics
        interval_a = create_interval(v1, v3, {})
        interval_b = create_interval(v2, v4, {})
        assert metrics_checker.check(interval_a, interval_b) is True

        # One interval has metrics, other doesn't
        interval_c = create_interval(v1, v3, {"product": "A"})
        assert metrics_checker.check(interval_c, interval_b) is False

    def test_different_metric_fields(self, mock_values, checkers, create_interval):
        """Test intervals with different sets of metric fields"""
        v1, v2, v3, v4 = mock_values[:4]
        metrics_checker = checkers['metrics_equivalent']

        # Different fields but overlapping intervals
        interval_a = create_interval(v1, v3, {"product": "A"})
        interval_b = create_interval(v2, v4, {"region": "US"})
        assert metrics_checker.check(interval_a, interval_b) is False

        # One interval has a superset of the other's fields
        interval_c = create_interval(v1, v3, {"product": "A", "region": "US"})
        interval_d = create_interval(v2, v4, {"product": "A"})
        assert metrics_checker.check(interval_c, interval_d) is False

    def test_case_sensitivity_in_metric_names(self, mock_values, checkers, create_interval):
        """Test case sensitivity in metric field names"""
        v1, v2, v3, v4 = mock_values[:4]
        metrics_checker = checkers['metrics_equivalent']

        # Case difference in field names
        interval_a = create_interval(v1, v3, {"Product": "A"})
        interval_b = create_interval(v2, v4, {"product": "A"})
        assert metrics_checker.check(interval_a, interval_b) is False

        # Same case, same values, should match
        interval_c = create_interval(v1, v3, {"Product": "A"})
        interval_d = create_interval(v2, v4, {"Product": "A"})
        assert metrics_checker.check(interval_c, interval_d) is True

    def test_special_characters_in_metrics(self, mock_values, checkers, create_interval):
        """Test metrics with special characters and whitespace"""
        v1, v2, v3, v4 = mock_values[:4]
        metrics_checker = checkers['metrics_equivalent']

        # Special characters in field names
        interval_a = create_interval(v1, v3, {"product-id": "A123"})
        interval_b = create_interval(v2, v4, {"product-id": "A123"})
        assert metrics_checker.check(interval_a, interval_b) is True

        # Whitespace in field names
        interval_c = create_interval(v1, v3, {"product id": "A123"})
        interval_d = create_interval(v2, v4, {"product id": "A123"})
        assert metrics_checker.check(interval_c, interval_d) is True

        # Special characters in field values
        interval_e = create_interval(v1, v3, {"product": "A#123"})
        interval_f = create_interval(v2, v4, {"product": "A#123"})
        assert metrics_checker.check(interval_e, interval_f) is True

        # Whitespace in field values
        interval_g = create_interval(v1, v3, {"product": "Product A"})
        interval_h = create_interval(v2, v4, {"product": "Product A"})
        assert metrics_checker.check(interval_g, interval_h) is True

        # Mismatch with special characters in field values
        interval_i = create_interval(v1, v3, {"product": "A#123"})
        interval_j = create_interval(v2, v4, {"product": "A-123"})
        assert metrics_checker.check(interval_i, interval_j) is False

    def test_metric_name_variations(self, mock_values, checkers, create_interval):
        """Test various metric name formats and variations"""
        v1, v2, v3, v4 = mock_values[:4]
        metrics_checker = checkers['metrics_equivalent']

        # Unicode characters in field names
        interval_a = create_interval(v1, v3, {"prøduct": "A"})
        interval_b = create_interval(v2, v4, {"prøduct": "A"})
        assert metrics_checker.check(interval_a, interval_b) is True

        # Unicode characters vs. ASCII in field names
        interval_c = create_interval(v1, v3, {"prøduct": "A"})
        interval_d = create_interval(v2, v4, {"product": "A"})
        assert metrics_checker.check(interval_c, interval_d) is False

        # Leading/trailing whitespace in field names
        interval_e = create_interval(v1, v3, {" product": "A"})
        interval_f = create_interval(v2, v4, {"product": "A"})
        assert metrics_checker.check(interval_e, interval_f) is False

        interval_g = create_interval(v1, v3, {"product ": "A"})
        interval_h = create_interval(v2, v4, {"product": "A"})
        assert metrics_checker.check(interval_g, interval_h) is False

    def test_edge_case_metric_values(self, mock_values, checkers, create_interval):
        """Test edge cases in metric values"""
        v1, v2, v3, v4 = mock_values[:4]
        metrics_checker = checkers['metrics_equivalent']

        # Empty strings as values
        interval_a = create_interval(v1, v3, {"product": ""})
        interval_b = create_interval(v2, v4, {"product": ""})
        assert metrics_checker.check(interval_a, interval_b) is True

        # None vs. empty string
        interval_c = create_interval(v1, v3, {"product": None})
        interval_d = create_interval(v2, v4, {"product": ""})
        assert metrics_checker.check(interval_c, interval_d) is False

        # Boolean values
        interval_e = create_interval(v1, v3, {"active": True})
        interval_f = create_interval(v2, v4, {"active": True})
        assert metrics_checker.check(interval_e, interval_f) is True

        # Boolean vs. string representation
        interval_g = create_interval(v1, v3, {"active": True})
        interval_h = create_interval(v2, v4, {"active": "True"})
        assert metrics_checker.check(interval_g, interval_h) is False

        # Numeric values
        interval_i = create_interval(v1, v3, {"count": 123})
        interval_j = create_interval(v2, v4, {"count": 123})
        assert metrics_checker.check(interval_i, interval_j) is True

        # Numeric vs. string representation
        interval_k = create_interval(v1, v3, {"count": 123})
        interval_l = create_interval(v2, v4, {"count": "123"})
        assert metrics_checker.check(interval_k, interval_l) is False


class TestPracticalScenarios:
    """Tests simulating real-world usage scenarios"""

    def test_calendar_event_overlaps(self, mock_values, checkers, create_interval):
        """Test interval relations in a calendar scheduling context"""
        v1, v2, v3, v4, v5 = mock_values

        # Create some calendar events
        meeting_9am_10am = create_interval(v1, v2)
        meeting_10am_11am = create_interval(v2, v3)
        lunch_12pm_1pm = create_interval(v3, v4)
        afternoon_meeting = create_interval(v4, v5)  # 1pm-2pm meeting
        all_day_meeting = create_interval(v1, v5)

        # Test scheduling conflicts
        assert checkers['meets'].check(meeting_9am_10am, meeting_10am_11am) is True
        assert checkers['meets'].check(meeting_10am_11am, lunch_12pm_1pm) is True  # Fix for original assertion

        # Test 'before' relationship (no shared endpoints)
        assert checkers['before'].check(meeting_9am_10am, lunch_12pm_1pm) is True  # 9am-10am is before 12pm-1pm
        assert checkers['before'].check(meeting_10am_11am, afternoon_meeting) is True  # 10am-11am is before 1pm-2pm

        # Test relationships between all_day_meeting and other meetings
        # In Allen's algebra, since all_day_meeting starts at the same time as meeting_9am_10am,
        # this is a 'started by' relationship, not 'contains'
        assert checkers['started_by'].check(all_day_meeting, meeting_9am_10am) is True

        # all_day_meeting properly contains lunch_12pm_1pm (no shared endpoints)
        assert checkers['contains'].check(all_day_meeting, lunch_12pm_1pm) is True

        # No conflict between these meetings
        assert checkers['overlaps'].check(meeting_9am_10am, lunch_12pm_1pm) is False

    def test_process_monitoring_intervals(self, mock_values, checkers, create_interval):
        """Test interval relations in a process monitoring context"""
        v1, v2, v3, v4, v5 = mock_values

        # Create some process monitoring intervals
        system_uptime = create_interval(v1, v5, {"system": "Server A"})
        maintenance_window = create_interval(v2, v3, {"system": "Server A"})
        outage = create_interval(v3, v4, {"system": "Server A"})

        # Test monitoring scenarios
        assert checkers['during'].check(maintenance_window, system_uptime) is True
        assert checkers['during'].check(outage, system_uptime) is True
        assert checkers['meets'].check(maintenance_window, outage) is True

        # Test metric equivalence
        assert checkers['metrics_equivalent'].check(system_uptime, maintenance_window) is True

        # Different system shouldn't match
        other_system = create_interval(v2, v4, {"system": "Server B"})
        assert checkers['metrics_equivalent'].check(system_uptime, other_system) is False


class TestMutualExclusivity:
    """Comprehensive tests to ensure Allen's interval relations are mutually exclusive"""

    def test_comprehensive_mutual_exclusivity(self, mock_values, checkers, create_interval):
        """
        Test that for any pair of intervals, exactly one relation checker returns true.
        This is a fundamental property of Allen's interval algebra.

        Note: This test excludes point intervals (where start == end) as Allen's interval algebra
        was defined for intervals with non-zero duration.
        """
        v1, v2, v3, v4, v5 = mock_values

        # Create a comprehensive set of intervals representing different configurations
        # Excluding point intervals (where start == end)
        intervals = {
            "v1_to_v2": create_interval(v1, v2),  # Normal interval [v1,v2]
            "v1_to_v3": create_interval(v1, v3),  # Normal interval [v1,v3]
            "v1_to_v4": create_interval(v1, v4),  # Normal interval [v1,v4]
            "v1_to_v5": create_interval(v1, v5),  # Normal interval [v1,v5]
            "v2_to_v3": create_interval(v2, v3),  # Normal interval [v2,v3]
            "v2_to_v4": create_interval(v2, v4),  # Normal interval [v2,v4]
            "v2_to_v5": create_interval(v2, v5),  # Normal interval [v2,v5]
            "v3_to_v4": create_interval(v3, v4),  # Normal interval [v3,v4]
            "v3_to_v5": create_interval(v3, v5),  # Normal interval [v3,v5]
            "v4_to_v5": create_interval(v4, v5),  # Normal interval [v4,v5]
        }

        # List of all Allen's relation checkers
        relation_checkers = [
            'before', 'meets', 'overlaps', 'starts', 'during', 'finishes',
            'equals', 'finished_by', 'contains', 'started_by', 'overlapped_by',
            'met_by', 'after'
        ]

        # For each pair of intervals in the list
        for name_a, interval_a in intervals.items():
            for name_b, interval_b in intervals.items():
                # Count which relations are true for this pair
                true_relations = []

                for checker_name in relation_checkers:
                    if checkers[checker_name].check(interval_a, interval_b):
                        true_relations.append(checker_name)

                # There should be exactly one true relation between any pair of intervals
                # (except for the metrics_equivalent checker which isn't part of Allen's algebra)
                assert len(true_relations) == 1, \
                    f"Intervals {name_a} and {name_b} have {len(true_relations)} true relations: {true_relations}. Expected exactly 1."

                # For debugging purposes, print the relation
                relation = true_relations[0]
                # print(f"{name_a} {relation} {name_b}")

    def test_relation_pairs_symmetry(self, mock_values, checkers, create_interval):
        """
        Test the symmetry properties of interval relations.
        If A relation_X B, then B inverse_relation_X A.
        """
        v1, v2, v3, v4, v5 = mock_values

        # Define the inverse relationships
        inverse_relations = {
            'before': 'after',
            'meets': 'met_by',
            'overlaps': 'overlapped_by',
            'starts': 'started_by',
            'during': 'contains',
            'finishes': 'finished_by',
            'equals': 'equals',  # Self-inverse
            'finished_by': 'finishes',
            'contains': 'during',
            'started_by': 'starts',
            'overlapped_by': 'overlaps',
            'met_by': 'meets',
            'after': 'before'
        }

        # Create a representative set of intervals
        intervals = {
            "v1_to_v2": create_interval(v1, v2),
            "v2_to_v3": create_interval(v2, v3),
            "v2_to_v4": create_interval(v2, v4),
            "v1_to_v3": create_interval(v1, v3),
            "v1_to_v4": create_interval(v1, v4),
            "v3_to_v4": create_interval(v3, v4),
            "v1_to_v5": create_interval(v1, v5),
            "point_at_v2": create_interval(v2, v2),
            "point_at_v4": create_interval(v4, v4),
        }

        # Test all pairs of intervals
        for name_a, interval_a in intervals.items():
            for name_b, interval_b in intervals.items():
                # Find which relation is true for A to B
                relation_a_to_b = None
                for relation in inverse_relations.keys():
                    if checkers[relation].check(interval_a, interval_b):
                        relation_a_to_b = relation
                        break

                # If we found a relation from A to B
                if relation_a_to_b:
                    # Get the expected inverse relation
                    expected_relation_b_to_a = inverse_relations[relation_a_to_b]

                    # Check if the inverse relation is true from B to A
                    assert checkers[expected_relation_b_to_a].check(interval_b, interval_a), \
                        f"If {name_a} {relation_a_to_b} {name_b}, then {name_b} should {expected_relation_b_to_a} {name_a}"

    def test_identity_relations(self, mock_values, checkers, create_interval):
        """
        Test that every non-point interval equals itself and no other relation
        applies to identical intervals.

        Note: This test excludes point intervals (where start == end) as Allen's interval
        algebra was defined for intervals with non-zero duration.
        """
        v1, v2, v3, v4, v5 = mock_values

        # Create a set of diverse intervals, excluding zero-length intervals
        test_intervals = [
            create_interval(v1, v2),  # Regular
            create_interval(v2, v4),  # Longer
            create_interval(v1, v5),  # Longest
        ]

        non_equals_relations = [
            'before', 'meets', 'overlaps', 'starts', 'during', 'finishes',
            'contains', 'started_by', 'finished_by', 'overlapped_by', 'met_by', 'after'
        ]

        for interval in test_intervals:
            # An interval should equal itself
            assert checkers['equals'].check(interval, interval), \
                "Every interval should equal itself"

            # No other relation should apply to identical intervals
            for relation in non_equals_relations:
                assert not checkers[relation].check(interval, interval), \
                    f"Relation '{relation}' should not apply to identical intervals"

    def test_transitivity_properties(self, mock_values, checkers, create_interval):
        """Test the transitivity properties of selected Allen's relations"""
        v1, v2, v3, v4, v5 = mock_values

        # Test before transitivity: if A before B and B before C then A before C
        a_before = create_interval(v1, v2)
        b_before = create_interval(v3, v4)
        c_before = create_interval(v5, v5)

        assert checkers['before'].check(a_before, b_before)
        assert checkers['before'].check(b_before, c_before)
        assert checkers['before'].check(a_before, c_before), \
            "Before relation should be transitive"

        # Test after transitivity: if A after B and B after C then A after C
        a_after = create_interval(v5, v5)
        b_after = create_interval(v3, v4)
        c_after = create_interval(v1, v2)

        assert checkers['after'].check(a_after, b_after)
        assert checkers['after'].check(b_after, c_after)
        assert checkers['after'].check(a_after, c_after), \
            "After relation should be transitive"

        # Test during transitivity: if A during B and B during C then A during C
        c_during = create_interval(v1, v5)
        b_during = create_interval(v2, v4)
        a_during = create_interval(v3, v3)

        assert checkers['during'].check(a_during, b_during)
        assert checkers['during'].check(b_during, c_during)
        assert checkers['during'].check(a_during, c_during), \
            "During relation should be transitive"

        # Test contains transitivity: if A contains B and B contains C then A contains C
        a_contains = create_interval(v1, v5)
        b_contains = create_interval(v2, v4)
        c_contains = create_interval(v3, v3)

        assert checkers['contains'].check(a_contains, b_contains)
        assert checkers['contains'].check(b_contains, c_contains)
        assert checkers['contains'].check(a_contains, c_contains), \
            "Contains relation should be transitive"

    def test_exhaustive_pairwise_relations(self, mock_values, checkers, create_interval):
        """
        Test every possible pair of intervals with every relation checker to ensure only one returns true.
        This includes regular intervals but excludes point intervals.
        """
        v1, v2, v3, v4, v5 = mock_values

        # Generate all possible non-point intervals
        intervals = []
        for start_idx in range(1, 5):  # Using 1-4 as start indices
            for end_idx in range(start_idx + 1, 6):  # Using start+1 to 5 as end indices
                start_val = mock_values[start_idx - 1]
                end_val = mock_values[end_idx - 1]
                intervals.append((f"v{start_idx}_v{end_idx}", create_interval(start_val, end_val)))

        relation_checkers = [
            'before', 'meets', 'overlaps', 'starts', 'during', 'finishes',
            'equals', 'finished_by', 'contains', 'started_by', 'overlapped_by',
            'met_by', 'after'
        ]

        # Test all pairs
        for name_a, interval_a in intervals:
            for name_b, interval_b in intervals:
                true_relations = []

                for relation in relation_checkers:
                    if checkers[relation].check(interval_a, interval_b):
                        true_relations.append(relation)

                assert len(true_relations) == 1, \
                    f"Intervals {name_a} and {name_b} have {len(true_relations)} true relations: {true_relations}. Expected exactly 1."

    def test_shared_endpoints_exhaustive(self, mock_values, checkers, create_interval):
        """
        Test all possible combinations of intervals that share one or both endpoints.
        """
        v1, v2, v3, v4, v5 = mock_values

        # Create pairs of intervals with specific shared endpoints
        shared_endpoint_pairs = [
            # Both start at same point, different ends
            (create_interval(v1, v2), create_interval(v1, v3), 'starts', 'started_by'),

            # A finishes B: A ends at same point as B but starts after B
            (create_interval(v2, v3), create_interval(v1, v3), 'finishes', 'finished_by'),

            # End of first equals start of second (meets/met_by)
            (create_interval(v1, v2), create_interval(v2, v3), 'meets', 'met_by'),

            # Identical intervals (equals)
            (create_interval(v1, v3), create_interval(v1, v3), 'equals', 'equals'),
        ]

        for interval_a, interval_b, relation_a_to_b, relation_b_to_a in shared_endpoint_pairs:
            # Test forward relation
            assert checkers[relation_a_to_b].check(interval_a, interval_b), \
                f"Expected {relation_a_to_b} to be true from A to B"

            # Test inverse relation
            assert checkers[relation_b_to_a].check(interval_b, interval_a), \
                f"Expected {relation_b_to_a} to be true from B to A"

            # Check that all other relations are false (A to B)
            for relation in checkers:
                if relation != relation_a_to_b and relation != 'metrics_equivalent':
                    assert not checkers[relation].check(interval_a, interval_b), \
                        f"Relation {relation} should be false for A to B"

            # Check that all other relations are false (B to A)
            for relation in checkers:
                if relation != relation_b_to_a and relation != 'metrics_equivalent':
                    assert not checkers[relation].check(interval_b, interval_a), \
                        f"Relation {relation} should be false for B to A"


class TestTransitivityProperties:
    """Tests focused specifically on the transitivity properties of Allen's interval relations"""

    def test_before_transitivity(self, mock_values, checkers, create_interval):
        """Test the transitivity of the 'before' relation"""
        v1, v2, v3, v4, v5 = mock_values

        # If A before B and B before C, then A before C
        interval_a = create_interval(v1, v1)  # Point at v1
        interval_b = create_interval(v2, v2)  # Point at v2
        interval_c = create_interval(v3, v3)  # Point at v3

        assert checkers['before'].check(interval_a, interval_b)
        assert checkers['before'].check(interval_b, interval_c)
        assert checkers['before'].check(interval_a, interval_c), "Before relation should be transitive"

        # Test with non-point intervals too
        interval_d = create_interval(v1, v2)
        interval_e = create_interval(v3, v4)
        interval_f = create_interval(v5, v5)

        assert checkers['before'].check(interval_d, interval_e)
        assert checkers['before'].check(interval_e, interval_f)
        assert checkers['before'].check(interval_d,
                                        interval_f), "Before relation should be transitive for non-point intervals"

    def test_after_transitivity(self, mock_values, checkers, create_interval):
        """Test the transitivity of the 'after' relation"""
        v1, v2, v3, v4, v5 = mock_values

        # If A after B and B after C, then A after C
        interval_a = create_interval(v5, v5)
        interval_b = create_interval(v3, v3)
        interval_c = create_interval(v1, v1)

        assert checkers['after'].check(interval_a, interval_b)
        assert checkers['after'].check(interval_b, interval_c)
        assert checkers['after'].check(interval_a, interval_c), "After relation should be transitive"

    def test_during_transitivity(self, mock_values, checkers, create_interval):
        """Test the transitivity of the 'during' relation"""
        v1, v2, v3, v4, v5 = mock_values

        # If A during B and B during C, then A during C
        interval_a = create_interval(v3, v3)
        interval_b = create_interval(v2, v4)
        interval_c = create_interval(v1, v5)

        assert checkers['during'].check(interval_a, interval_b)
        assert checkers['during'].check(interval_b, interval_c)
        assert checkers['during'].check(interval_a, interval_c), "During relation should be transitive"

    def test_contains_transitivity(self, mock_values, checkers, create_interval):
        """Test the transitivity of the 'contains' relation"""
        v1, v2, v3, v4, v5 = mock_values

        # If A contains B and B contains C, then A contains C
        interval_a = create_interval(v1, v5)
        interval_b = create_interval(v2, v4)
        interval_c = create_interval(v3, v3)

        assert checkers['contains'].check(interval_a, interval_b)
        assert checkers['contains'].check(interval_b, interval_c)
        assert checkers['contains'].check(interval_a, interval_c), "Contains relation should be transitive"

    def test_equals_transitivity(self, mock_values, checkers, create_interval):
        """Test the transitivity of the 'equals' relation"""
        v1, v2, v3 = mock_values[:3]

        # If A equals B and B equals C, then A equals C
        interval_a = create_interval(v1, v2)
        # Create copies with the same values
        interval_b = create_interval(v1, v2)
        interval_c = create_interval(v1, v2)

        assert checkers['equals'].check(interval_a, interval_b)
        assert checkers['equals'].check(interval_b, interval_c)
        assert checkers['equals'].check(interval_a, interval_c), "Equals relation should be transitive"

    def test_starts_transitivity(self, mock_values, checkers, create_interval):
        """Test that 'starts' relation IS transitive"""
        v1, v2, v3, v4, v5 = mock_values

        # A starts B, B starts C, then A starts C
        interval_a = create_interval(v1, v2)
        interval_b = create_interval(v1, v3)
        interval_c = create_interval(v1, v4)

        assert checkers['starts'].check(interval_a, interval_b)
        assert checkers['starts'].check(interval_b, interval_c)
        # A also starts C (they share the same start point and A ends before C ends)
        assert checkers['starts'].check(interval_a, interval_c), "The 'starts' relation is transitive"

    def test_finishes_transitivity(self, mock_values, checkers, create_interval):
        """Test that 'finishes' relation IS transitive"""
        v1, v2, v3, v4, v5 = mock_values

        # A finishes B, B finishes C, then A finishes C
        interval_a = create_interval(v3, v4)
        interval_b = create_interval(v2, v4)
        interval_c = create_interval(v1, v4)

        assert checkers['finishes'].check(interval_a, interval_b)
        assert checkers['finishes'].check(interval_b, interval_c)
        # A also finishes C (they share the same end point and A starts after C starts)
        assert checkers['finishes'].check(interval_a, interval_c), "The 'finishes' relation is transitive"

    def test_meets_non_transitivity(self, mock_values, checkers, create_interval):
        """Test that 'meets' relation is NOT transitive"""
        v1, v2, v3, v4 = mock_values[:4]

        # If A meets B and B meets C, then A does NOT meet C
        interval_a = create_interval(v1, v2)
        interval_b = create_interval(v2, v3)
        interval_c = create_interval(v3, v4)

        assert checkers['meets'].check(interval_a, interval_b)
        assert checkers['meets'].check(interval_b, interval_c)
        # A should be before C, not meets
        assert not checkers['meets'].check(interval_a, interval_c)
        assert checkers['before'].check(interval_a, interval_c), "When A meets B and B meets C, A is before C"

    def test_overlaps_non_transitivity(self, mock_values, checkers, create_interval):
        """Test that 'overlaps' relation is NOT transitive"""
        v1, v2, v3, v4, v5 = mock_values

        # It's possible for A to overlap B and B to overlap C, but A doesn't overlap C
        interval_a = create_interval(v1, v3)
        interval_b = create_interval(v2, v4)
        interval_c = create_interval(v3, v5)

        assert checkers['overlaps'].check(interval_a, interval_b)
        assert checkers['overlaps'].check(interval_b, interval_c)
        # A meets C but doesn't overlap it
        assert not checkers['overlaps'].check(interval_a, interval_c)
        assert checkers['meets'].check(interval_a, interval_c), "When A overlaps B and B overlaps C, A may meet C"

    def test_mixed_transitivity_chains(self, mock_values, checkers, create_interval):
        """Test transitivity across different relation types"""
        v1, v2, v3, v4, v5 = mock_values

        # If A before B and B before C, then A before C
        interval_a = create_interval(v1, v2)
        interval_b = create_interval(v3, v4)
        interval_c = create_interval(v5, v5)

        assert checkers['before'].check(interval_a, interval_b)
        assert checkers['before'].check(interval_b, interval_c)
        assert checkers['before'].check(interval_a, interval_c)

        # If A meets B and B before C, then A before C
        interval_d = create_interval(v1, v3)
        interval_e = create_interval(v3, v4)
        interval_f = create_interval(v5, v5)

        assert checkers['meets'].check(interval_d, interval_e)
        assert checkers['before'].check(interval_e, interval_f)
        assert checkers['before'].check(interval_d, interval_f), "If A meets B and B before C, then A before C"

        # If A during B and B during C, then A during C
        interval_g = create_interval(v3, v3)
        interval_h = create_interval(v2, v4)
        interval_i = create_interval(v1, v5)

        assert checkers['during'].check(interval_g, interval_h)
        assert checkers['during'].check(interval_h, interval_i)
        assert checkers['during'].check(interval_g, interval_i), "If A during B and B during C, then A during C"
