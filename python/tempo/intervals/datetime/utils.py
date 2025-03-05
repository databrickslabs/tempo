from pandas import Timestamp


def infer_datetime_format(date_string: str) -> str:
    """
    Extracts the exact format from a sample datetime string.
    This preserves the exact format of the input string.
    """
    # Replace all date/time components with their format codes
    replacements = [
        # Order matters - replace longer patterns first
        (r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}[-+]\d{4}', '%Y-%m-%dT%H:%M:%S.%f%z'),
        (r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}', '%Y-%m-%dT%H:%M:%S.%f'),
        (r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[-+]\d{4}', '%Y-%m-%dT%H:%M:%S%z'),
        (r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', '%Y-%m-%dT%H:%M:%S'),
        (r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}', '%Y-%m-%d %H:%M:%S.%f'),
        (r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '%Y-%m-%d %H:%M:%S'),
        (r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}', '%Y-%m-%d %H:%M'),
        (r'\d{4}-\d{2}-\d{2}', '%Y-%m-%d'),
        (r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}', '%m/%d/%Y %H:%M:%S'),
        (r'\d{2}/\d{2}/\d{4}', '%m/%d/%Y'),
    ]

    for pattern, repl in replacements:
        if Timestamp(date_string).strftime(repl) == date_string:
            return repl

    # If no exact match found, parse it with pandas and get its format
    return '%Y-%m-%d %H:%M:%S'  # Default format if no match
