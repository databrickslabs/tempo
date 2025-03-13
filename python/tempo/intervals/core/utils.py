from pandas import concat, DataFrame, NA, Series

from tempo.intervals.core.interval import Interval
from tempo.intervals.overlap.transformer import IntervalTransformer


class IntervalsUtils:
    def __init__(
            self,
            intervals: DataFrame,
    ):
        """
        Initialize IntervalsUtils with a DataFrame and interval properties.

        intervals_set (DataFrame): Input DataFrame containing interval data, with start and end timestamps.
        interval (Interval): An object representing the reference interval, containing the following:
            - data (pd.Series): A Series representing the interval's attributes.
            - start_ts (str): Column name for the start timestamp.
            - end_ts (str): Column name for the end timestamp.
        """
        self.intervals = intervals
        self._disjoint_set = DataFrame()

    @property
    def disjoint_set(self) -> DataFrame:
        return self._disjoint_set

    @disjoint_set.setter
    def disjoint_set(self, value: DataFrame) -> None:
        self._disjoint_set = value

    def _calculate_all_overlaps(self, interval: "Interval") -> DataFrame:
        max_start = "_MAX_START_TS"
        max_end = "_MAX_END_TS"

        # Check if input DataFrame or interval data is empty; return an empty DataFrame if true.
        if self.intervals.empty or interval.data.empty:
            # return in_pdf
            return DataFrame()

        intervals_copy = self.intervals.copy()

        # Calculate the latest possible start timestamp for overlap comparison.
        intervals_copy[max_start] = intervals_copy[interval.start_field].where(
            intervals_copy[interval.start_field] >= interval.data[interval.start_field],
            interval.data[interval.start_field],
        )

        # Calculate the earliest possible end timestamp for overlap comparison.
        intervals_copy[max_end] = intervals_copy[interval.end_field].where(
            intervals_copy[interval.end_field] <= interval.data[interval.end_field],
            interval.data[interval.end_field],
        )

        # https://www.baeldung.com/cs/finding-all-overlapping-intervals
        intervals_copy = intervals_copy[
            intervals_copy[max_start] < intervals_copy[max_end]
        ]

        # Remove intermediate columns used for interval overlap calculation.
        cols_to_drop = [max_start, max_end]
        intervals_copy = intervals_copy.drop(columns=cols_to_drop)

        return intervals_copy

    def find_overlaps(self, interval: "Interval") -> DataFrame:

        all_overlaps = self._calculate_all_overlaps(interval)

        # Remove rows that are identical to `interval.data`
        remove_with_row_mask = ~(
                all_overlaps.isna().eq(interval.data.isna())
                & all_overlaps.eq(interval.data).fillna(False)
        ).all(axis=1)

        deduplicated_overlaps = all_overlaps[remove_with_row_mask]

        return deduplicated_overlaps

    def add_as_disjoint(self, interval: "Interval") -> DataFrame:
        """
        returns a disjoint set consisting of the given interval, made disjoint with those already in `disjoint_set`
        """

        if self.disjoint_set is None or self.disjoint_set.empty:
            return DataFrame([interval.data])

        overlapping_subset_df = IntervalsUtils(self.disjoint_set).find_overlaps(
            interval
        )

        # if there are no overlaps, add the interval to disjoint_set
        if overlapping_subset_df.empty:
            element_wise_comparison = (
                    self.disjoint_set.copy().fillna(NA) == interval.data.fillna(NA).values
            )

            row_wise_comparison = element_wise_comparison.all(axis=1)
            # NB: because of the nested iterations, we need to check that the
            # record hasn't already been added to `global_disjoint_df` by another loop
            if row_wise_comparison.any():
                return self.disjoint_set
            else:
                return concat((self.disjoint_set, DataFrame([interval.data])))

        # identify all intervals which do not overlap with the given interval to
        # concatenate them to the disjoint set after resolving overlaps
        non_overlapping_subset_df = self.disjoint_set[
            ~self.disjoint_set.set_index(
                keys=[interval.start_field, interval.end_field]
            ).index.isin(
                overlapping_subset_df.set_index(
                    keys=[interval.start_field, interval.end_field]
                ).index
            )
        ]

        # Avoid a call to `resolve_all_overlaps` if there is only one to resolve
        multiple_to_resolve = len(overlapping_subset_df.index) > 1

        # If every record overlaps, no need to handle non-overlaps
        only_overlaps_present = len(self.disjoint_set.index) == len(
            overlapping_subset_df.index
        )

        # Resolve the interval against all the existing, overlapping intervals
        # `multiple_to_resolve` is used to avoid unnecessary calls to `resolve_all_overlaps`
        # `only_overlaps_present` is used to avoid unnecessary calls to `pd.concat`
        if not multiple_to_resolve and only_overlaps_present:
            resolver = IntervalTransformer(
                interval=Interval.create(
                    interval.data,
                    interval.start_field,
                    interval.end_field,
                    interval.series_fields,
                    interval.metric_fields,
                ),
                other=Interval.create(
                    overlapping_subset_df.iloc[0],
                    interval.start_field,
                    interval.end_field,
                    interval.series_fields,
                    interval.metric_fields,
                ),
            )
            return DataFrame(resolver.resolve_overlap())

        if multiple_to_resolve and only_overlaps_present:
            return IntervalsUtils(overlapping_subset_df).resolve_all_overlaps(interval)

        if not multiple_to_resolve and not only_overlaps_present:
            resolver = IntervalTransformer(
                interval=Interval.create(
                    interval.data,
                    interval.start_field,
                    interval.end_field,
                    interval.series_fields,
                    interval.metric_fields,
                ),
                other=Interval.create(
                    overlapping_subset_df.iloc[0],
                    interval.start_field,
                    interval.end_field,
                    interval.series_fields,
                    interval.metric_fields,
                ),
            )
            return concat(
                (
                    DataFrame(resolver.resolve_overlap()),
                    non_overlapping_subset_df,
                ),
            )

        if multiple_to_resolve and not only_overlaps_present:
            return concat(
                (
                    IntervalsUtils(overlapping_subset_df).resolve_all_overlaps(
                        interval
                    ),
                    non_overlapping_subset_df,
                ),
            )

        # if we get here, something went wrong
        raise NotImplementedError("Interval resolution not implemented")

    def resolve_all_overlaps(self, interval: "Interval") -> DataFrame:
        """
        Resolve the interval against all overlapping intervals in `intervals`,
        returning a set of disjoint intervals with the same spans
        """
        if self.intervals.empty:
            return DataFrame([interval.data])

        # First, check if there are any overlaps
        overlaps = self._calculate_all_overlaps(interval)

        # If no overlaps, just return the reference interval
        if overlaps.empty:
            return DataFrame([interval.data])

        # Process first row
        first_row = Interval.create(
            overlaps.iloc[0],  # Use overlaps, not self.intervals
            interval.start_field,
            interval.end_field,
            interval.series_fields,
            interval.metric_fields,
        )
        resolver = IntervalTransformer(interval, first_row)
        initial_intervals = resolver.resolve_overlap()
        disjoint_intervals = DataFrame(initial_intervals)

        # Only process additional rows if they exist
        if len(overlaps) > 1:  # Use overlaps, not self.intervals
            # Type-correct implementation of the nested function
            def resolve_and_add(row: Series) -> None:
                row_interval = Interval.create(
                    row,
                    interval.start_field,
                    interval.end_field,
                    interval.series_fields,
                    interval.metric_fields,
                )
                local_resolver = IntervalTransformer(interval, row_interval)
                resolved_intervals = local_resolver.resolve_overlap()
                for interval_data in resolved_intervals:
                    interval_inner = Interval.create(
                        interval_data,
                        interval.start_field,
                        interval.end_field,
                        interval.series_fields,
                        interval.metric_fields,
                    )
                    nonlocal disjoint_intervals
                    local_interval_utils = IntervalsUtils(disjoint_intervals)
                    local_interval_utils.disjoint_set = disjoint_intervals
                    disjoint_intervals = local_interval_utils.add_as_disjoint(
                        interval_inner
                    )

            # Use apply with explicit parameters to satisfy type checker
            # Type ignore comment added to suppress mypy error about apply
            overlaps.iloc[1:].apply(resolve_and_add, axis=1)  # type: ignore

        return disjoint_intervals
