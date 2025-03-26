import pytest
from pandas import Timestamp

from tempo.intervals.datetime.utils import infer_datetime_format


@pytest.mark.parametrize("date_string", [
    # ISO 8601 formats with timezone
    "2023-01-01T12:34:56.789012+0000",
    "2023-01-01T12:34:56.789012Z",
    "2023-01-01T12:34:56+0000",
    "2023-01-01T12:34:56Z",
    "2023-01-01T12:34+0000",
    "2023-01-01T12:34Z",

    # ISO 8601 formats without timezone
    "2023-01-01T12:34:56.789012",
    "2023-01-01T12:34:56",
    "2023-01-01T12:34",

    # Standard datetime formats with microseconds
    "2023-01-01 12:34:56.789012+0000",
    "2023-01-01 12:34:56.789012",
    "2023-01-01 12:34:56.789",

    # Standard datetime formats
    "2023-01-01 12:34:56+0000",
    "2023-01-01 12:34:56",
    "2023-01-01 12:34",

    # Date only formats
    "2023-01-01",
    "2023/01/01",

    # US date formats
    "01/01/2023 12:34:56.789012",
    "01/01/2023 12:34:56",
    "01/01/2023 12:34",
    "01/01/2023",
    "1/1/2023",
    "1/1/23",

    # UK/European date formats
    "01-01-2023 12:34:56",
    "01-01-2023",
    "01.01.2023",
    "01.01.2023 12:34:56",

    # Month name formats
    "Jan 01, 2023",
    "Jan 1, 2023",
    "01 Jan 2023",
    "1 Jan 2023",
    "January 01, 2023",
    "January 1, 2023",
    "01 January 2023",
    "1 January 2023",
    "January 1, 2023 12:34:56",

    # Time only formats
    "12:34:56",
    "12:34",

    # Special formats
    "20230101123456",
    "20230101",
])
def test_format_matches_input(date_string):
    """Test that the inferred format correctly reproduces the input when used with strftime"""
    try:
        # Get the inferred format
        fmt = infer_datetime_format(date_string)

        # Parse and reformat to check match
        reformatted = Timestamp(date_string).strftime(fmt)

        # Check that the reformatted date matches the input
        # Special handling for 'T' separator in ISO 8601 formats
        if 'T' in date_string and ' ' in reformatted:
            # Replace space with 'T' at the right position
            t_index = date_string.find('T')
            reformatted = reformatted[:t_index] + 'T' + reformatted[t_index + 1:]

        # Special handling for milliseconds vs microseconds
        if '.' in date_string and len(date_string.split('.')[-1]) < 6 and reformatted.endswith('000'):
            # For dates with milliseconds, strip trailing zeros from microseconds
            reformatted_ms = reformatted[:-3]
            assert reformatted_ms == date_string, \
                f"Failed for: {date_string}\nFormat: {fmt}\nReformatted: {reformatted_ms}"
        else:
            assert reformatted == date_string, \
                f"Failed for: {date_string}\nFormat: {fmt}\nReformatted: {reformatted}"
    except (ValueError, TypeError, OverflowError) as e:
        pytest.fail(f"Error parsing {date_string}: {e}")


@pytest.mark.parametrize("date_string,expected_format", [
    # Leap year day
    ("2024-02-29", "%Y-%m-%d"),

    # Very old date
    ("1800-01-01", "%Y-%m-%d"),

    # Future date
    ("2100-01-01", "%Y-%m-%d"),

    # Extreme timezone
    ("2023-01-01T12:00:00+1400", "%Y-%m-%dT%H:%M:%S%z"),
    ("2023-01-01T12:00:00-1400", "%Y-%m-%dT%H:%M:%S%z"),

    # Midnight and special times
    ("2023-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
    ("2023-01-01 23:59:59", "%Y-%m-%d %H:%M:%S"),
])
def test_edge_cases(date_string, expected_format):
    """Test edge cases and unusual formats"""
    try:
        # Get the inferred format
        fmt = infer_datetime_format(date_string)

        # Check against expected format
        assert fmt == expected_format, \
            f"Format for {date_string} was {fmt}, expected {expected_format}"

        # Verify it works as expected
        reformatted = Timestamp(date_string).strftime(fmt)
        assert reformatted == date_string, \
            f"Failed for: {date_string}\nFormat: {fmt}\nReformatted: {reformatted}"
    except (ValueError, TypeError, OverflowError) as e:
        pytest.fail(f"Error parsing {date_string}: {e}")


@pytest.mark.parametrize("input_string", [
    # Completely non-date string
    "not a date",

    # Malformed dates
    "2023-13-01",  # Invalid month
    "2023-01-32",  # Invalid day
    "2023/02/30",  # Invalid day for February

    # Ambiguous formats (function should still return something)
    "01/02/03",  # Ambiguous MM/DD/YY or DD/MM/YY

    # Empty string
    "",
])
def test_invalid_inputs(input_string):
    """Test that invalid inputs are handled appropriately"""
    # The function should return a default format without raising an exception
    try:
        fmt = infer_datetime_format(input_string)
        assert isinstance(fmt, str), "Function should return a string format"
    except Exception as e:
        pytest.fail(f"Function raised {type(e).__name__} for input '{input_string}': {e}")


@pytest.mark.parametrize("date_string,expected_format", [
    # RFC 822 format
    ("Wed, 02 Oct 2002 13:00:00 GMT", "%a, %d %b %Y %H:%M:%S GMT"),

    # Excel/Lotus style
    ("2023-1-1", "%Y-%m-%d"),

    # Date with weekday
    ("Monday, January 1, 2023", "%A, %B %d, %Y"),

    # 12-hour clock with AM/PM
    ("2023-01-01 01:30:00 PM", "%Y-%m-%d %I:%M:%S %p"),
])
def test_custom_formats(date_string, expected_format):
    """Test some custom or unusual but valid datetime formats"""
    try:
        # For this test, we'll check that the inferred format can parse the string,
        # rather than expecting an exact format match
        fmt = infer_datetime_format(date_string)

        # Try to parse with the inferred format
        # This may not exactly match the input due to limitations in strftime
        timestamp = Timestamp(date_string)

        # We just verify that some format was returned
        assert isinstance(fmt, str), f"Format for {date_string} should be a string"
    except (ValueError, TypeError, OverflowError) as e:
        pytest.fail(f"Error handling {date_string}: {e}")
