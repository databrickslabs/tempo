"""
Utility functions and constants for resampling operations.
This module contains shared utilities that don't depend on TSDF class.
"""

from typing import (
    Callable,
    List,
    Tuple,
    TypedDict,
    Union,
    get_type_hints,
)

# define global frequency options
MUSEC = "microsec"
MS = "ms"
SEC = "sec"
MIN = "min"
HR = "hr"
DAY = "day"

# define global aggregate function options for downsampling
floor = "floor"
min = "min"
max = "max"
average = "mean"
ceiling = "ceil"


class FreqDict(TypedDict):
    musec: str
    microsec: str
    microsecond: str
    microseconds: str
    ms: str
    millisecond: str
    milliseconds: str
    sec: str
    second: str
    seconds: str
    min: str
    minute: str
    minutes: str
    hr: str
    hour: str
    hours: str
    day: str
    days: str


freq_dict: FreqDict = {
    "musec": "microseconds",
    "microsec": "microseconds",
    "microsecond": "microseconds",
    "microseconds": "microseconds",
    "ms": "milliseconds",
    "millisecond": "milliseconds",
    "milliseconds": "milliseconds",
    "sec": "seconds",
    "second": "seconds",
    "seconds": "seconds",
    "min": "minutes",
    "minute": "minutes",
    "minutes": "minutes",
    "hr": "hours",
    "hour": "hours",
    "hours": "hours",
    "day": "days",
    "days": "days",
}

ALLOWED_FREQ_KEYS: List[str] = list(get_type_hints(FreqDict).keys())


def is_valid_allowed_freq_keys(val: str, literal_constant: List[str]) -> bool:
    return val in literal_constant


allowableFreqs = [MUSEC, MS, SEC, MIN, HR, DAY]
allowableFuncs = [floor, min, max, average, ceiling]


def checkAllowableFreq(freq: str) -> Tuple[Union[int, str], str]:
    """
    Parses frequency and checks against allowable frequencies
    :param freq: frequncy at which to upsample/downsample, declared in resample function
    :return: list of parsed frequency value and time suffix
    """
    if not isinstance(freq, str):
        raise TypeError(f"Invalid type for `freq` argument: {freq}.")

    # Default value that will be overwritten if valid frequency is found
    allowable_freq: Tuple[Union[int, str], str] = (
        0,
        "will_always_fail_if_not_overwritten",
    )

    if is_valid_allowed_freq_keys(
            freq.lower(),
            ALLOWED_FREQ_KEYS,
    ):
        allowable_freq = 1, freq
        return allowable_freq

    try:
        periods = int(freq.lower().split(" ")[0].strip())
        units = freq.lower().split(" ")[1].strip()
    except (IndexError, ValueError):
        raise ValueError(
            "Allowable grouping frequencies are microsecond (musec), millisecond (ms), sec (second), min (minute), hr (hour), day. Reformat your frequency as <integer> <day/hour/minute/second>"
        )

    if is_valid_allowed_freq_keys(
            units.lower(),
            ALLOWED_FREQ_KEYS,
    ):
        if units.startswith(MUSEC):
            allowable_freq = periods, MUSEC
        elif units.startswith(MS) | units.startswith("millis"):
            allowable_freq = periods, MS
        elif units.startswith(SEC):
            allowable_freq = periods, SEC
        elif units.startswith(MIN):
            allowable_freq = periods, MIN
        elif units.startswith("hour") | units.startswith(HR):
            allowable_freq = periods, "hour"
        elif units.startswith(DAY):
            allowable_freq = periods, DAY
    else:
        raise ValueError(f"Invalid value for `freq` argument: {freq}.")

    return allowable_freq


def validateFuncExists(func: Union[Callable, str]) -> None:
    if func is None:
        raise TypeError(
            "Aggregate function missing. Provide one of the allowable functions: "
            + ", ".join(allowableFuncs)
        )
    elif func not in allowableFuncs:
        raise ValueError(
            "Aggregate function is not in the valid list. Provide one of the allowable functions: "
            + ", ".join(allowableFuncs)
        )
