from typing import Sequence, Callable

from pandas import DataFrame, Series
from pyspark.sql.types import (
    ByteType,
    ShortType,
    IntegerType,
    LongType,
    FloatType,
    DoubleType,
    DecimalType,
    StructField,
    BooleanType,
)

from tempo.intervals.core.interval import Interval
from tempo.intervals.core.utils import IntervalsUtils


def is_metric_col(col: StructField) -> bool:
    return isinstance(
        col.dataType,
        (
            ByteType,
            ShortType,
            IntegerType,
            LongType,
            FloatType,
            DoubleType,
            DecimalType,
        ),
    ) or isinstance(col.dataType, BooleanType)


def make_disjoint_wrap(
    start_field: str,
    end_field: str,
    series_fields: Sequence[str],
    metric_fields: Sequence[str],
) -> Callable[[DataFrame], DataFrame]:
    """Returns a Pandas UDF for resolving overlapping intervals into disjoint intervals."""

    def make_disjoint_inner(pdf: DataFrame) -> DataFrame:
        """
        Processes a grouped Pandas DataFrame to resolve overlapping intervals into disjoint intervals.

        Args:
            pdf (DataFrame): Pandas DataFrame grouped by Spark.

        Returns:
            DataFrame: Disjoint intervals as a Pandas DataFrame.
        """
        # Handle empty input DataFrame explicitly
        if pdf.empty:
            return pdf

        # Ensure intervals are sorted by start and end timestamps
        # Use inplace=True to avoid unnecessary copy
        pdf = pdf.sort_values(by=[start_field, end_field]).reset_index(drop=True)

        # Initialize empty disjoint intervals DataFrame
        disjoint_intervals = DataFrame(columns=pdf.columns)

        # For best performance, process in chunks using numpy arrays
        # Convert to numpy for faster access
        values = pdf.values
        columns = pdf.columns

        # Process rows in a more efficient manner
        for i in range(len(pdf)):
            # Create a Series from the row values for compatibility with Interval.create
            row = Series(values[i], index=columns)

            # Create interval and add as disjoint
            interval = Interval.create(
                row, start_field, end_field, series_fields, metric_fields
            )

            # Use cached utils object with updated disjoint_set
            local_utils = IntervalsUtils(disjoint_intervals)
            local_utils.disjoint_set = disjoint_intervals
            disjoint_intervals = local_utils.add_as_disjoint(interval)

        return disjoint_intervals

    return make_disjoint_inner
