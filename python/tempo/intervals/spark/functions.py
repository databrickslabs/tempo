from typing import Sequence, Callable

from pandas import DataFrame
from pyspark.sql.types import ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType, DecimalType, \
    StructField, BooleanType

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
        sorted_intervals = pdf.sort_values(by=[start_field, end_field]).reset_index(drop=True)

        # Initialize an empty DataFrame to store disjoint intervals
        disjoint_intervals = DataFrame(columns=pdf.columns)

        # Define a function to process each row and update disjoint intervals
        def process_row(row):
            nonlocal disjoint_intervals
            interval = Interval.create(row, start_field, end_field, series_fields, metric_fields)
            local_utils = IntervalsUtils(disjoint_intervals)
            local_utils.disjoint_set = disjoint_intervals
            disjoint_intervals = local_utils.add_as_disjoint(interval)

        # Apply the processing function to each row
        sorted_intervals.apply(process_row, axis=1)

        return disjoint_intervals

    return make_disjoint_inner
