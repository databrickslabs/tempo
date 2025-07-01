from functools import total_ordering
from typing import NamedTuple


@total_ordering
class TimeUnit(NamedTuple):
    name: str
    approx_seconds: float
    """
    Represents a unit of time, with a name,
    and the approximate number of seconds in that unit.
    """

    def __eq__(self, other):
        return self.approx_seconds == other.approx_seconds

    def __lt__(self, other):
        return self.approx_seconds < other.approx_seconds


class TimeUnitsType(NamedTuple):
    YEARS: TimeUnit
    MONTHS: TimeUnit
    WEEKS: TimeUnit
    DAYS: TimeUnit
    HOURS: TimeUnit
    MINUTES: TimeUnit
    SECONDS: TimeUnit
    MILLISECONDS: TimeUnit
    MICROSECONDS: TimeUnit
    NANOSECONDS: TimeUnit


StandardTimeUnits = TimeUnitsType(
    TimeUnit("year", 365 * 24 * 60 * 60),
    TimeUnit("month", 30 * 24 * 60 * 60),
    TimeUnit("week", 7 * 24 * 60 * 60),
    TimeUnit("day", 24 * 60 * 60),
    TimeUnit("hour", 60 * 60),
    TimeUnit("minute", 60),
    TimeUnit("second", 1),
    TimeUnit("millisecond", 1e-03),
    TimeUnit("microsecond", 1e-06),
    TimeUnit("nanosecond", 1e-09),
)
