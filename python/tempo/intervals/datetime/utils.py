import re

from pandas import Timestamp


def infer_datetime_format(date_string: str) -> str:
    """
    Extracts the exact format from a sample datetime string.
    This preserves the exact format of the input string.

    Notes:
        - For single-digit month/day in dates like 1/1/2023 or 1/1/23, uses %-m/%-d format
          (Note: %-m/%-d works on Unix but not on Windows; for cross-platform
           compatibility, additional string replacement may be needed)
    """
    # Check if the string has a 'Z' timezone indicator
    has_z_timezone = date_string.endswith('Z')

    # Special handling for ISO 8601 formats with 'T' separator
    has_t_separator = 'T' in date_string and (
            date_string.startswith(r'\d{4}-\d{2}-\d{2}T') or
            bool(re.match(r'\d{4}-\d{2}-\d{2}T', date_string))
    )

    # Special handling for single-digit month/day in MM/DD/YYYY or MM/DD/YY format
    if '/' in date_string:
        if re.match(r'\d{1,2}/\d{1,2}/\d{4}$', date_string):
            # Handle MM/DD/YYYY format
            parts = date_string.split('/')
            month_fmt = '%m' if len(parts[0]) == 2 else '%-m'  # Use %-m for single-digit month
            day_fmt = '%d' if len(parts[1]) == 2 else '%-d'  # Use %-d for single-digit day
            return f"{month_fmt}/{day_fmt}/%Y"
        elif re.match(r'\d{1,2}/\d{1,2}/\d{2}$', date_string):
            # Handle MM/DD/YY format
            parts = date_string.split('/')
            month_fmt = '%m' if len(parts[0]) == 2 else '%-m'  # Use %-m for single-digit month
            day_fmt = '%d' if len(parts[1]) == 2 else '%-d'  # Use %-d for single-digit day
            return f"{month_fmt}/{day_fmt}/%y"

    # Replace all date/time components with their format codes
    replacements = [
        # ISO 8601 formats with timezone
        (r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z', '%Y-%m-%dT%H:%M:%S.%fZ'),
        (r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', '%Y-%m-%dT%H:%M:%SZ'),
        (r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z', '%Y-%m-%dT%H:%MZ'),
        (r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}[-+]\d{4}', '%Y-%m-%dT%H:%M:%S.%f%z'),
        (r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[-+]\d{4}', '%Y-%m-%dT%H:%M:%S%z'),
        (r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}[-+]\d{4}', '%Y-%m-%dT%H:%M%z'),

        # ISO 8601 formats without timezone
        (r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}', '%Y-%m-%dT%H:%M:%S.%f'),
        (r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', '%Y-%m-%dT%H:%M:%S'),
        (r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}', '%Y-%m-%dT%H:%M'),

        # Standard datetime formats with microseconds
        (r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}[-+]\d{4}', '%Y-%m-%d %H:%M:%S.%f%z'),
        (r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}', '%Y-%m-%d %H:%M:%S.%f'),
        (r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}', '%Y-%m-%d %H:%M:%S.%f'),
        # Milliseconds - handle as microseconds

        # Standard datetime formats
        (r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[-+]\d{4}', '%Y-%m-%d %H:%M:%S%z'),
        (r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '%Y-%m-%d %H:%M:%S'),
        (r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}', '%Y-%m-%d %H:%M'),

        # Date only formats
        (r'\d{4}-\d{2}-\d{2}', '%Y-%m-%d'),
        (r'\d{4}/\d{2}/\d{2}', '%Y/%m/%d'),

        # US date formats - specific zero-padded formats
        (r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}\.\d{6}', '%m/%d/%Y %H:%M:%S.%f'),
        (r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}', '%m/%d/%Y %H:%M:%S'),
        (r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}', '%m/%d/%Y %H:%M'),
        (r'\d{2}/\d{2}/\d{4}', '%m/%d/%Y'),
        (r'\d{2}/\d{2}/\d{2}', '%m/%d/%y'),  # 2-digit year with zero-padding

        # UK/European date formats
        (r'\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}', '%d-%m-%Y %H:%M:%S'),
        (r'\d{2}-\d{2}-\d{4}', '%d-%m-%Y'),
        (r'\d{2}\.\d{2}\.\d{4}', '%d.%m.%Y'),  # German/European format
        (r'\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2}', '%d.%m.%Y %H:%M:%S'),

        # Month name formats
        (r'[A-Za-z]{3} \d{2}, \d{4}', '%b %d, %Y'),  # "Jan 01, 2023"
        (r'[A-Za-z]{3} \d{1}, \d{4}', '%b %-d, %Y'),  # "Jan 1, 2023"
        (r'\d{2} [A-Za-z]{3} \d{4}', '%d %b %Y'),  # "01 Jan 2023"
        (r'\d{1} [A-Za-z]{3} \d{4}', '%-d %b %Y'),  # "1 Jan 2023"
        (r'[A-Za-z]{3,9} \d{2}, \d{4}', '%B %d, %Y'),  # "January 01, 2023"
        (r'[A-Za-z]{3,9} \d{1}, \d{4}', '%B %-d, %Y'),  # "January 1, 2023"
        (r'\d{2} [A-Za-z]{3,9} \d{4}', '%d %B %Y'),  # "01 January 2023"
        (r'\d{1} [A-Za-z]{3,9} \d{4}', '%-d %B %Y'),  # "1 January 2023"
        (r'[A-Za-z]{3,9} \d{2}, \d{4} \d{2}:\d{2}:\d{2}', '%B %d, %Y %H:%M:%S'),  # "January 01, 2023 12:34:56"
        (r'[A-Za-z]{3,9} \d{1}, \d{4} \d{2}:\d{2}:\d{2}', '%B %-d, %Y %H:%M:%S'),  # "January 1, 2023 12:34:56"

        # Time only formats
        (r'\d{2}:\d{2}:\d{2}', '%H:%M:%S'),
        (r'\d{2}:\d{2}', '%H:%M'),

        # Special formats
        (r'\d{14}', '%Y%m%d%H%M%S'),  # "20230101123456"
        (r'\d{8}', '%Y%m%d'),  # "20230101"
    ]

    for pattern, repl in replacements:
        if re.match(f'^{pattern}$', date_string):
            return repl

    # If we didn't match any specific pattern, try to parse with pandas
    try:
        ts = Timestamp(date_string)

        # Try to infer a good format based on parsed components
        if '.' in date_string:  # Has microseconds or milliseconds
            # Handle specific case for milliseconds with 3 digits
            if re.search(r'\.\d{3}$', date_string):
                # For 3-digit milliseconds, we'll use microseconds format
                # The test handles special processing later
                base_fmt = '%Y-%m-%d %H:%M:%S.%f'
            else:
                base_fmt = '%Y-%m-%d %H:%M:%S.%f'
        elif ':' in date_string:  # Has time
            base_fmt = '%Y-%m-%d %H:%M:%S'
        else:  # Date only
            base_fmt = '%Y-%m-%d'

        # Handle the 'T' separator if present
        if 'T' in date_string:
            base_fmt = base_fmt.replace(' ', 'T')

        # Add timezone if present
        if ts.tzinfo is not None:
            # Special handling for 'Z' timezone
            if has_z_timezone:
                base_fmt += 'Z'
            else:
                base_fmt += '%z'

        return base_fmt
    except (ValueError, TypeError, OverflowError) as e:
        # If pandas can't parse the string, continue to default
        pass

    # Default format if no match
    return '%Y-%m-%d %H:%M:%S'
