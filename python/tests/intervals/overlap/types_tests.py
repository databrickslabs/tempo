from tempo.intervals.overlap.types import OverlapType


class TestOverlapTypeContract:
    """Tests verifying the contract of the OverlapType enum."""

    def test_completeness(self):
        # Verify all expected types exist
        expected_types = [
            "METRICS_EQUIVALENT", "BEFORE", "MEETS", "OVERLAPS", "STARTS",
            "DURING", "FINISHES", "EQUALS", "CONTAINS", "STARTED_BY",
            "FINISHED_BY", "OVERLAPPED_BY", "MET_BY", "AFTER"
        ]
        actual_types = [member.name for member in OverlapType]
        assert set(expected_types) == set(actual_types)

        # Verify all values are unique
        values = [member.value for member in OverlapType]
        assert len(values) == len(set(values))
