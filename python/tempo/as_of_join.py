import copy
from abc import ABC, abstractmethod
from functools import reduce
from typing import Optional, Tuple

import pyspark.sql.functions as sfn
from pyspark.sql import Column, DataFrame, SparkSession

import tempo.tsdf as t_tsdf
from tempo.timeunit import TimeUnit
from tempo.tsschema import CompositeTSIndex, TSSchema


# Helpers


class _AsOfJoinCompositeTSIndex(CompositeTSIndex):
    """
    CompositeTSIndex subclass for as-of joins
    """

    @property
    def unit(self) -> Optional[TimeUnit]:
        return None

    def rangeExpr(self, reverse: bool = False) -> Column:
        raise NotImplementedError(
            "rangeExpr is not defined for as-of join composite ts index"
        )


# As-of join types


class AsOfJoiner(ABC):
    """
    Abstract class for as-of join strategies
    """

    def __init__(self, left_prefix: str = "left", right_prefix: str = "right"):
        self.left_prefix = left_prefix
        self.right_prefix = right_prefix

    def __call__(self, left: t_tsdf.TSDF, right: t_tsdf.TSDF) -> t_tsdf.TSDF:
        # check if the TSDFs are joinable
        self._checkAreJoinable(left, right)
        # prefix overlapping columns
        left, right = self._prefixOverlappingColumns(left, right)
        # perform the join
        return self._join(left, right)

    def commonSeriesIDs(self, left: t_tsdf.TSDF, right: t_tsdf.TSDF) -> set:
        """
        Returns the common series IDs between the left and right TSDFs
        """
        return set(left.series_ids).intersection(set(right.series_ids))

    def _prefixableColumns(self, left: t_tsdf.TSDF, right: t_tsdf.TSDF) -> set:
        """
        Returns the overlapping columns in the left and right TSDFs
        not including overlapping series IDs
        """
        return set(left.columns).intersection(
            set(right.columns)
        ) - self.commonSeriesIDs(left, right)

    def _prefixColumns(
        self, tsdf: t_tsdf.TSDF, prefixable_cols: set[str], prefix: str
    ) -> t_tsdf.TSDF:
        """
        Prefixes the columns in the TSDF
        """
        if prefix:
            tsdf = reduce(
                lambda cur_tsdf, c: cur_tsdf.withColumnRenamed(
                    c, "_".join([prefix, c])
                ),
                prefixable_cols,
                tsdf,
            )
        return tsdf

    def _prefixOverlappingColumns(
        self, left: t_tsdf.TSDF, right: t_tsdf.TSDF
    ) -> Tuple[t_tsdf.TSDF, t_tsdf.TSDF]:
        """
        Prefixes the overlapping columns in the left and right TSDFs
        """

        # find the columns to prefix
        prefixable_cols = self._prefixableColumns(left, right)

        # prefix columns (if we have a prefix to apply)
        left_prefixed = self._prefixColumns(left, prefixable_cols, self.left_prefix)
        right_prefixed = self._prefixColumns(right, prefixable_cols, self.right_prefix)

        return left_prefixed, right_prefixed

    def _checkAreJoinable(self, left: t_tsdf.TSDF, right: t_tsdf.TSDF) -> None:
        """
        Checks if the left and right TSDFs are joinable. If not, raises an exception.
        """
        # make sure the schemas are equivalent
        assert left.ts_schema == right.ts_schema

    @abstractmethod
    def _join(self, left: t_tsdf.TSDF, right: t_tsdf.TSDF) -> t_tsdf.TSDF:
        """
        Returns a new TSDF with the join of the left and right TSDFs
        """


class BroadcastAsOfJoiner(AsOfJoiner):
    """
    Strategy class for broadcast as-of joins
    """

    def __init__(
        self,
        spark: SparkSession,
        left_prefix: str = "left",
        right_prefix: str = "right",
        range_join_bin_size: int = 60,
    ):
        super().__init__(left_prefix, right_prefix)
        self.spark = spark
        self.range_join_bin_size = range_join_bin_size

    def _join(self, left: t_tsdf.TSDF, right: t_tsdf.TSDF) -> t_tsdf.TSDF:
        # TODO (v0.2 refactor): Fix timestamp timezone handling for composite indexes with nanosecond precision
        # Currently, broadcast joins with composite timestamp indexes may have timezone inconsistencies
        # that cause test failures. This should be addressed in the v0.2 refactor.

        # set the range join bin size to 60 seconds
        self.spark.conf.set(
            "spark.databricks.optimizer.rangeJoin.binSize",
            str(self.range_join_bin_size),
        )
        # find a leading column in the right TSDF
        w = right.baseWindow()

        # Get the comparable expression for the timestamp index
        # This handles both simple and composite timestamp indexes
        right_comparable_expr = right.ts_index.comparableExpr()

        # For composite indexes, comparableExpr() returns a list
        # We need the first element which is the most comparable field
        # (e.g., double_ts for nanosecond precision timestamps)
        if isinstance(right_comparable_expr, list):
            if len(right_comparable_expr) > 0:
                right_comparable_expr = right_comparable_expr[0]
            else:
                # Fallback to column name if no comparable expression
                right_comparable_expr = sfn.col(right.ts_index.colname)

        lead_colname = "lead_" + right.ts_index.colname
        right_with_lead = right.withColumn(
            lead_colname, sfn.lead(right_comparable_expr).over(w)
        )

        # perform the join - use left join to preserve all left rows
        join_series_ids = self.commonSeriesIDs(left, right)

        # First do the left join
        joined_df = left.df.join(right_with_lead.df, list(join_series_ids), how="left")

        # Then filter based on timestamp conditions
        # Keep rows where either:
        # 1. There's no matching right row (right timestamp is NULL)
        # 2. The left timestamp is between the right timestamp and lead timestamp
        # 3. Lead is NULL (last right row) and left timestamp >= right timestamp
        res_df = joined_df.where(
            sfn.col(right.ts_index.colname).isNull()
            | (
                sfn.col(right.ts_index.colname).isNotNull()
                & (
                    # Normal case: left timestamp is between right and lead
                    (
                        sfn.col(lead_colname).isNotNull()
                        & left.ts_index.between(right.ts_index, sfn.col(lead_colname))
                    )
                    |
                    # Last right row: lead is NULL and left >= right
                    (sfn.col(lead_colname).isNull() & (left.ts_index >= right.ts_index))
                )
            )
        ).drop(lead_colname)
        # return with the left-hand schema
        return t_tsdf.TSDF(res_df, ts_schema=left.ts_schema)


_DEFAULT_COMBINED_TS_COLNAME = "combined_ts"
_DEFAULT_RIGHT_ROW_COLNAME = "righthand_tbl_row"
_LEFT_HAND_ROW_INDICATOR = 1
_RIGHT_HAND_ROW_INDICATOR = -1


class UnionSortFilterAsOfJoiner(AsOfJoiner):
    """
    Implements the classic as-of join strategy of unioning the left and right
    TSDFs, sorting by the timestamp column, and then filtering out only the
    most recent matching rows for each right-hand side row
    """

    def __init__(
        self,
        left_prefix: str = "left",
        right_prefix: str = "right",
        skipNulls: bool = True,
        tolerance: Optional[int] = None,
    ):
        super().__init__(left_prefix, right_prefix)
        self.skipNulls = skipNulls
        self.tolerance = tolerance

    def _appendNullColumns(self, tsdf: t_tsdf.TSDF, cols: set[str]) -> t_tsdf.TSDF:
        """
        Appends null columns to the TSDF
        """
        return reduce(
            lambda cur_tsdf, col: cur_tsdf.withColumn(col, sfn.lit(None)), cols, tsdf
        )

    def _combine(self, left: t_tsdf.TSDF, right: t_tsdf.TSDF) -> t_tsdf.TSDF:
        """
        Combines the left and right TSDFs into a single TSDF
        """
        # union the left and right TSDFs
        unioned = left.unionByName(right)
        # coalesce a combined ts index
        left_comps = left.ts_index.comparableExpr()
        right_comps = right.ts_index.comparableExpr()
        if not isinstance(left_comps, list):
            left_comps = [left_comps]
        if not isinstance(right_comps, list):
            right_comps = [right_comps]
        if isinstance(left.ts_index, t_tsdf.CompositeTSIndex):
            comp_names = left.ts_index.component_fields
        else:
            comp_names = [left.ts_index.colname]
        combined_comps = [
            sfn.coalesce(lc, rc).alias(cn)
            for lc, rc, cn in zip(left_comps, right_comps, comp_names)
        ]
        # build an expression for an right-hand side indicator field
        right_ind = (
            sfn.when(
                sfn.col(left.ts_index.colname).isNotNull(), _LEFT_HAND_ROW_INDICATOR
            )
            .otherwise(_RIGHT_HAND_ROW_INDICATOR)
            .alias(_DEFAULT_RIGHT_ROW_COLNAME)
        )
        # combine all into a single struct column
        with_combined_ts = unioned.withColumn(
            _DEFAULT_COMBINED_TS_COLNAME, sfn.struct(*combined_comps, right_ind)
        )
        combined_ts_col = with_combined_ts.df.schema[_DEFAULT_COMBINED_TS_COLNAME]
        # construct a CompositeTSIndex from the combined ts index
        combined_comp_names = comp_names + [_DEFAULT_RIGHT_ROW_COLNAME]
        combined_tsidx = _AsOfJoinCompositeTSIndex(
            combined_ts_col, *combined_comp_names
        )
        # put it all together in a new TSDF
        return t_tsdf.TSDF(
            with_combined_ts.df, ts_schema=TSSchema(combined_tsidx, left.series_ids)
        )

    def _filterLastRightRow(
        self, combined: t_tsdf.TSDF, right_cols: set[str], last_left_tsschema: TSSchema
    ) -> t_tsdf.TSDF:
        """
        Filters out the last right-hand row for each left-hand row

        TODO (v0.2 refactor): Fix timestamp timezone handling for composite indexes with nanosecond precision
        The union sort filter join may produce timestamps with different timezone offsets compared to
        broadcast join, causing test failures. This should be addressed in the v0.2 refactor.
        """
        # find the last value for each column in the right-hand side
        w = combined.allBeforeWindow()
        # Type assertion: ts_index is a CompositeTSIndex for as-of joins
        assert isinstance(combined.ts_index, CompositeTSIndex)
        right_row_field = combined.ts_index.fieldPath(_DEFAULT_RIGHT_ROW_COLNAME)
        if self.skipNulls:
            last_right_cols = [
                sfn.last(col, True).over(w).alias(col) for col in right_cols
            ]
        else:
            last_right_cols = [
                sfn.last(
                    sfn.when(
                        sfn.col(right_row_field) == _RIGHT_HAND_ROW_INDICATOR,
                        sfn.struct(col),
                    ).otherwise(None),
                    True,
                )
                .over(w)
                .alias(col)
                for col in right_cols
            ]
        # get the last right-hand row for each left-hand row
        non_right_cols = list(set(combined.columns) - right_cols)
        last_right_vals = combined.select(*(non_right_cols + last_right_cols))
        # filter out the last right-hand row for each left-hand row
        as_of_df = last_right_vals.df.where(
            sfn.col(right_row_field) == _LEFT_HAND_ROW_INDICATOR
        ).drop(_DEFAULT_COMBINED_TS_COLNAME)
        # return with the left-hand schema
        return t_tsdf.TSDF(as_of_df, ts_schema=copy.deepcopy(last_left_tsschema))

    def _join(self, left: t_tsdf.TSDF, right: t_tsdf.TSDF) -> t_tsdf.TSDF:
        # find the new columns to add to the left and right TSDFs
        right_only_cols = set(right.columns) - set(left.columns)
        left_only_cols = set(left.columns) - set(right.columns)
        # append null columns to the left and right TSDFs
        extended_left = self._appendNullColumns(left, right_only_cols)
        extended_right = self._appendNullColumns(right, left_only_cols)
        # combine the left and right TSDFs
        combined = self._combine(extended_left, extended_right)
        # filter out the last right-hand row for each left-hand row
        as_of = self._filterLastRightRow(
            combined, right_only_cols, extended_left.ts_schema
        )
        # apply tolerance filter
        if self.tolerance is not None:
            as_of = self._toleranceFilter(as_of)
        return as_of

    def _toleranceFilter(self, as_of: t_tsdf.TSDF) -> t_tsdf.TSDF:
        """
        Filters out rows from the as_of TSDF that are outside the tolerance
        """
        # TODO: Implement tolerance filtering logic
        # For now, return the input TSDF unchanged
        return as_of


class SkewAsOfJoiner(UnionSortFilterAsOfJoiner):
    """
    Implements the as-of join strategy for skewed data
    """

    def __init__(
        self,
        left_prefix: str = "left",
        right_prefix: str = "right",
        skipNulls: bool = True,
        tolerance: Optional[int] = None,
        tsPartitionVal: Optional[int] = None,
        fraction: float = 0.5,
    ):
        super().__init__(left_prefix, right_prefix, skipNulls, tolerance)

    def _join(self, left: t_tsdf.TSDF, right: t_tsdf.TSDF) -> t_tsdf.TSDF:
        # TODO: Implement skew-specific join logic
        return super()._join(left, right)


# Helper functions


def get_spark_plan(df: DataFrame, spark: SparkSession) -> str:
    """
    Internal helper function to obtain the Spark plan for the input data frame

    Parameters
    :param df - input Spark data frame - the AS OF join has 2 data frames; this will be called for each
    :param spark - Spark session which is used to query the view obtained from the Spark data frame
    """

    df.createOrReplaceTempView("view")
    plan = spark.sql("explain cost select * from view").collect()[0][0]

    return plan


def get_bytes_from_plan(df: DataFrame, spark: SparkSession) -> float:
    """
    Internal helper function to obtain how many bytes in memory the Spark data
    frame is likely to take up. This is an upper bound and is obtained from the
    plan details in Spark

    Parameters
    :param df - input Spark data frame - the AS OF join has 2 data frames; this will be called for each
    :param spark - Spark session which is used to query the view obtained from the Spark data frame
    """

    plan = get_spark_plan(df, spark)

    import re

    search_result = re.search(r"sizeInBytes=.*([')])", plan, re.MULTILINE)
    if search_result is not None:
        result = search_result.group(0).replace(")", "")
    else:
        raise ValueError("Unable to obtain sizeInBytes from Spark plan")

    size = result.split("=")[1].split(" ")[0]
    units = result.split("=")[1].split(" ")[1]

    # perform to MB for threshold check
    if units == "GiB":
        plan_bytes = float(size) * 1024 * 1024 * 1024
    elif units == "MiB":
        plan_bytes = float(size) * 1024 * 1024
    elif units == "KiB":
        plan_bytes = float(size) * 1024
    else:
        plan_bytes = float(size)

    return plan_bytes


# choose 30MB as the cutoff for the broadcast
__DEFAULT_BROADCAST_BYTES_THRESHOLD = 30 * 1024 * 1024


def choose_as_of_join_strategy(
    left_tsdf: t_tsdf.TSDF,
    right_tsdf: t_tsdf.TSDF,
    left_prefix: Optional[str] = None,
    right_prefix: str = "right",
    tsPartitionVal: Optional[int] = None,
    fraction: float = 0.5,
    skipNulls: bool = True,
    sql_join_opt: bool = False,
    suppress_null_warning: bool = False,
    tolerance: Optional[int] = None,
) -> AsOfJoiner:
    """
    Returns an AsOfJoiner object based on the parameters passed in
    """

    # test if the broadcast join will be efficient
    if sql_join_opt:
        spark = SparkSession.builder.getOrCreate()
        left_bytes = get_bytes_from_plan(left_tsdf.df, spark)
        right_bytes = get_bytes_from_plan(right_tsdf.df, spark)

        bytes_threshold = __DEFAULT_BROADCAST_BYTES_THRESHOLD
        if (left_bytes < bytes_threshold) or (right_bytes < bytes_threshold):
            return BroadcastAsOfJoiner(spark, left_prefix or "left", right_prefix)

    # use the skew join if the partition value is passed in
    if tsPartitionVal is not None:
        return SkewAsOfJoiner(
            left_prefix=left_prefix or "left",
            right_prefix=right_prefix,
            skipNulls=skipNulls,
            tolerance=tolerance,
            tsPartitionVal=tsPartitionVal,
            fraction=fraction,
        )

    # default to use the union sort filter join
    return UnionSortFilterAsOfJoiner(
        left_prefix or "left", right_prefix, skipNulls, tolerance
    )
