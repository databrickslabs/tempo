from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Dict


class OverlapType(Enum):
    """
    Comprehensive classification of possible interval relationships.
    Based on Allen's interval algebra.
    """

    METRICS_EQUIVALENT = (
        auto()
    )  # Overlapping intervals with same metrics; this is a special case
    BEFORE = auto()  # X completely before Y
    MEETS = auto()  # X ends where Y starts
    OVERLAPS = auto()  # X overlaps start of Y
    STARTS = auto()  # X and Y start together
    DURING = auto()  # X completely inside Y
    FINISHES = auto()  # X and Y end together
    EQUALS = auto()  # X and Y are identical
    CONTAINS = auto()  # X completely contains Y
    STARTED_BY = auto()  # Y starts at X start
    FINISHED_BY = auto()  # Y ends at X end
    OVERLAPPED_BY = auto()  # Y overlaps start of X
    MET_BY = auto()  # Y ends where X starts
    AFTER = auto()  # X completely after Y


@dataclass
class OverlapResult:
    type: OverlapType
    details: Optional[Dict] = None
