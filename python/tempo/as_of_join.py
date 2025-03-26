import copy
from abc import ABC, abstractmethod
from functools import reduce
from typing import List, Optional, cast

import pyspark.sql.functions as sfn
from pyspark.sql import DataFrame, SparkSession, Column, Row

from tempo.tsdf import TSDF
from tempo.timeunit import TimeUnit
from tempo.tsschema import CompositeTSIndex, TSSchema, ensure_composite_index, get_component_fields


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

    def __call__(self, left: TSDF, right: TSDF) -> TSDF:
        # check if the TSDFs are joinable
        self._checkAreJoinable(left, right)
        # prefix overlapping columns
        left, right = self._prefixOverlappingColumns(left, right)
        # perform the join
        return self._join(left, right)

    def commonSeriesIDs(self, left: TSDF, right: TSDF) -> set:
        """
        Returns the common series IDs between the left and right TSDFs
        """
        return set(left.series_ids).intersection(set(right.series_ids))

    def _prefixableColumns(self, left: TSDF, right: TSDF) -> set:
        """
        Returns the overlapping columns in the left and right TSDFs
        not including overlapping series IDs
        """
        return set(left.columns).intersection(
            set(right.columns)
        ) - self.commonSeriesIDs(left, right)

    def _prefixColumns(
        self, tsdf: TSDF, prefixable_cols: set[str], prefix: str
    ) -> TSDF:
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
        self, left: TSDF, right: TSDF
    ) -> tuple[TSDF, TSDF]:
        """
        Prefixes the overlapping columns in the left and right TSDFs
        """

        # find the columns to prefix
        prefixable_cols = self._prefixableColumns(left, right)

        # prefix columns (if we have a prefix to apply)
        left_prefixed = self._prefixColumns(left, prefixable_cols, self.left_prefix)
        right_prefixed = self._prefixColumns(right, prefixable_cols, self.right_prefix)

        return left_prefixed, right_prefixed

    def _checkAreJoinable(self, left: TSDF, right: TSDF) -> None:
        """
        Checks if the left and right TSDFs are joinable. If not, raises an exception.
        """
        # make sure the schemas are equivalent
        assert left.ts_schema == right.ts_schema

    @abstractmethod
    def _join(self, left: TSDF, right: TSDF) -> TSDF:
        """
        Returns a new TSDF with the join of the left and right TSDFs
        """
        pass


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

    def _join(self, left: TSDF, right: TSDF) -> TSDF:
        # set the range join bin size to 60 seconds
        self.spark.conf.set(
            "spark.databricks.optimizer.rangeJoin.binSize",
            str(self.range_join_bin_size),
        )
        # find a leading column in the right TSDF
        w = right.baseWindow()
        lead_colname = "lead_" + right.ts_index.colname
        right_with_lead = right.withColumn(
            lead_colname, sfn.lead(right.ts_index.colname).over(w)
        )
        # perform the join
        join_series_ids = self.commonSeriesIDs(left, right)
        res_df = (
            left.df.join(right_with_lead.df, list(join_series_ids))
            .where(left.ts_index.between(right.ts_index, sfn.col(lead_colname)))
            .drop(lead_colname)
        )
        # return with the left-hand schema
        return TSDF(res_df, ts_schema=left.ts_schema)


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

    def _appendNullColumns(self, tsdf: TSDF, cols: set[str]) -> TSDF:
        """
        Appends null columns to the TSDF
        """
        return reduce(
            lambda cur_tsdf, col: cur_tsdf.withColumn(col, sfn.lit(None)), cols, tsdf
        )

    def _combine(self, left: TSDF, right: TSDF) -> TSDF:
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
        comp_names = get_component_fields(left.ts_index)
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
        return TSDF(
            with_combined_ts.df, ts_schema=TSSchema(combined_tsidx, left.series_ids)
        )

    def _filterLastRightRow(
        self, combined: TSDF, right_cols: set[str], last_left_tsschema: TSSchema
    ) -> TSDF:
        """
        Filters out the last right-hand row for each left-hand row
        """
        # find the last value for each column in the right-hand side
        w = combined.allBeforeWindow()
        right_row_field = ensure_composite_index(combined.ts_index).fieldPath(_DEFAULT_RIGHT_ROW_COLNAME)
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
        return TSDF(as_of_df, ts_schema=copy.deepcopy(last_left_tsschema))

    def _toleranceFilter(self, as_of: TSDF) -> TSDF:
        """
        Filters out rows from the as_of TSDF that are outside the tolerance
        """
        raise NotImplementedError

    def _join(self, left: TSDF, right: TSDF) -> TSDF:
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

    def _join(self, left: TSDF, right: TSDF) -> TSDF:
        raise NotImplementedError


# Helper functions


def get_spark_plan(df: DataFrame, spark: SparkSession) -> str:
    """
    Internal helper function to obtain the Spark plan for the input data frame

    Parameters
    :param df - input Spark data frame - the AS OF join has 2 data frames; this will be called for each
    :param spark - Spark session which is used to query the view obtained from the Spark data frame
    """

    df.createOrReplaceTempView("view")
    result = spark.sql("explain cost select * from view").collect()
    # Use cast to tell the type checker that result is a List[Row]
    rows = cast(List[Row], result)
    plan = rows[0][0]

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
    left_tsdf: TSDF,
    right_tsdf: TSDF,
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
            # Use a default value if left_prefix is None
            return BroadcastAsOfJoiner(spark, left_prefix or "left", right_prefix)

    # use the skew join if the partition value is passed in
    if tsPartitionVal is not None:
        if tsPartitionVal is not None:
            return SkewAsOfJoiner(
                left_prefix=left_prefix or "left",
                right_prefix=right_prefix,
                skipNulls=skipNulls,
                tolerance=tolerance,
                tsPartitionVal=tsPartitionVal,
                fraction=fraction
            )

    # default to use the union sort filter join
    return UnionSortFilterAsOfJoiner(left_prefix or "left", right_prefix, skipNulls, tolerance)

