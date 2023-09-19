from abc import ABC, abstractmethod
from typing import Optional
from functools import reduce

from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as sfn

import tempo.tsdf as t_tsdf

# As-of join types


joiner = MyAsOfJoinerImpl()
joiner(left_tsdf, right_tsdf)

class AsOfJoiner(ABC):
    """
    Abstract class for as-of join strategies
    """
    def __init__(self,
                 left_prefix: str = "left",
                 right_prefix: str = "right"):
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
        return (set(left.columns).intersection(set(right.columns)) -
                self.commonSeriesIDs(left, right))

    def _prefixColumns(self,
                       tsdf: t_tsdf.TSDF,
                       prefixable_cols: set[str],
                       prefix: str) -> t_tsdf.TSDF:
        """
        Prefixes the columns in the TSDF
        """
        if prefix:
            tsdf = reduce(
                lambda cur_tsdf, c:
                    cur_tsdf.withColumnRenamed(c, "_".join([prefix, c])),
                prefixable_cols,
                tsdf
            )
        return tsdf

    def _prefixOverlappingColumns(self,
                                  left: t_tsdf.TSDF,
                                  right: t_tsdf.TSDF) -> (t_tsdf.TSDF, t_tsdf.TSDF):
        """
        Prefixes the overlapping columns in the left and right TSDFs
        """

        # find the columns to prefix
        prefixable_cols = self._prefixableColumns(left, right)

        # prefix columns (if we have a prefix to apply)
        left_prefixed = self._prefixColumns(left, prefixable_cols, self.left_prefix)
        right_prefixed = self._prefixColumns(right, prefixable_cols, self.right_prefix)

        return left_prefixed, right_prefixed

    def _checkPartitionCols(self, left: t_tsdf.TSDF, right: t_tsdf.TSDF) -> None:
        for left_col, right_col in zip(left.series_ids, right.series_ids):
            if left_col != right_col:
                raise ValueError(
                    "left and right dataframe partition columns should have same name in same order"
                )

    def _validateTsColMatch(self, left: t_tsdf.TSDF, right: t_tsdf.TSDF) -> None:
        # TODO - can simplify this to get types from schema object
        left_ts_datatype = type(left.ts_index)
        right_ts_datatype = type(right.ts_index)
        if left_ts_datatype != right_ts_datatype:
            raise ValueError(
                "left and right dataframe timestamp index columns should have same type"
            )

    def _checkAreJoinable(self, left: t_tsdf.TSDF, right: t_tsdf.TSDF) -> None:
        """
        Checks if the left and right TSDFs are joinable. If not, raises an exception.
        """
        # make sure their partition columns are the same
        self._checkPartitionCols(left, right)
        # make sure their timeseries indices are the same
        self._validateTsColMatch(left, right)

    @abstractmethod
    def _join(self, left: t_tsdf.TSDF, right: t_tsdf.TSDF) -> t_tsdf.TSDF:
        """
        Returns a new TSDF with the join of the left and right TSDFs
        """
        pass


class BroadcastAsOfJoiner(AsOfJoiner):
    """
    Strategy class for broadcast as-of joins
    """

    def __init__(self,
                 spark: SparkSession,
                 left_prefix: str = "left",
                 right_prefix: str = "right",
                 range_join_bin_size: int = 60):
        super().__init__(left_prefix, right_prefix)
        self.spark = spark
        self.range_join_bin_size = range_join_bin_size

    def _join(self, left: t_tsdf.TSDF, right: t_tsdf.TSDF) -> t_tsdf.TSDF:
        # set the range join bin size to 60 seconds
        self.spark.conf.set("spark.databricks.optimizer.rangeJoin.binSize",
                            str(self.range_join_bin_size))
        # find a leading column in the right TSDF
        w = right.baseWindow()
        lead_colname = "lead_" + right.ts_col
        right_with_lead = right.withColumn(lead_colname,
                                           sfn.coalesce(
                                               sfn.lead(right.ts_col).over(w),
                                               sfn.lit("2099-01-01").cast("timestamp"))
                                           )
        # perform the join
        join_series_ids = self.commonSeriesIDs(left, right)
        res_df = (left.df.join(right_with_lead.df, list(join_series_ids))
                  .where(left.ts_index.betweenExpr(right.ts_col, lead_colname))
                  .drop(lead_colname))
        # return with the left-hand schema
        return t_tsdf.TSDF(res_df, ts_schema=left.ts_schema)


_DEFAULT_COMBINED_TS_COLNAME = "combined_ts"
_DEFAULT_RIGHT_ROW_COLNAME = "righthand_tbl_row"

class UnionSortFilterAsOfJoin(AsOfJoiner):
    """
    Implements the classic as-of join strategy of unioning the left and right
    TSDFs, sorting by the timestamp column, and then filtering out only the
    most recent matching rows for each right-hand side row
    """

    def __init__(self,
                 left_prefix: str = "left",
                 right_prefix: str = "right",
                 skipNulls: bool = True,
                 tolerance: Optional[int] = None):
        super().__init__(left_prefix, right_prefix)
        self.skipNulls = skipNulls
        self.tolerance = tolerance

    def _appendNullColumns(self, tsdf: t_tsdf.TSDF, cols: set[str]) -> t_tsdf.TSDF:
        """
        Appends null columns to the TSDF
        """
        return reduce(
            lambda cur_tsdf, col: cur_tsdf.withColumn(col, sfn.lit(None)),
            cols,
            tsdf)

    def _addRightRowIndicator(self, tsdf: t_tsdf.TSDF) -> t_tsdf.TSDF:
        """
        Adds a column to indicate if the row is from the right-hand side
        """
        # add indicator column
        right_row_ind_df = tsdf.df.withColumn(_DEFAULT_RIGHT_ROW_COLNAME,
                                              sfn.when(
                                                  sfn.col(tsdf.ts_col).isNotNull(),
                                                  1)
                                              .otherwise(-1))
        # extend the TSIndex with the indicator as a sub-sequence index
        if isinstance(tsdf.ts_index, t_tsdf.CompositeTSIndex):
            # TODO - implement extending an existing CompositeTSIndex
            raise NotImplementedError("Extending an existing CompositeTSIndex "
                                      "is not yet implemented")
        else:
            return t_tsdf.TSDF.fromSubsequenceCol(right_row_ind_df,
                                                  tsdf.ts_col,
                                                  _DEFAULT_RIGHT_ROW_COLNAME,
                                                  tsdf.series_ids)

    def _combine(self, left: t_tsdf.TSDF, right: t_tsdf.TSDF) -> t_tsdf.TSDF:
        """
        Combines the left and right TSDFs into a single TSDF
        """
        # union the left and right TSDFs
        unioned = left.unionByName(right)
        # coalesce the timestamp column,
        # and add a column to indicate if the row is from the right-hand side
        combined_ts = unioned.withColumn(_DEFAULT_COMBINED_TS_COLNAME,
                                         sfn.coalesce(left.ts_col,
                                                      right.ts_col))
        # add indicator column
        r_row_ind = combined_ts.withColumn(_DEFAULT_RIGHT_ROW_COLNAME,
                                           sfn.when(
                                               sfn.col(left.ts_col).isNotNull(),1)
                                           .otherwise(-1))
        # combine all the ts columns into a single struct column
        return t_tsdf.TSDF.makeCompositeIndexTSDF(r_row_ind.df,
                                                  [_DEFAULT_COMBINED_TS_COLNAME,
                                                   _DEFAULT_RIGHT_ROW_COLNAME],
                                                  combined_ts.series_ids,
                                                  [left.ts_col, right.ts_col])

    def _filterLastRightRow(self,
                            combined: t_tsdf.TSDF,
                            right_cols: set[str]) -> t_tsdf.TSDF:
        """
        Filters out the last right-hand row for each left-hand row
        """
        # find the last value for each column in the right-hand side
        w = combined.allBeforeWindow()
        last_right_vals = reduce(
            lambda cur_tsdf, col: cur_tsdf.withColumn(
                col,
                sfn.last(col, self.skipNulls).over(w),
            ),
            right_cols,
            combined,
        )
        # filter out the last right-hand row for each left-hand row
        r_row_ind: str = combined.ts_index.component(_DEFAULT_RIGHT_ROW_COLNAME)
        filtered = last_right_vals.where(sfn.col(r_row_ind) == 1).drop(r_row_ind)
        return filtered

    def _toleranceFilter(self, as_of: t_tsdf.TSDF) -> t_tsdf.TSDF:
        """
        Filters out rows from the as_of TSDF that are outside the tolerance
        """
        pass

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
        as_of = self._filterLastRightRow(combined, right_only_cols)
        # apply tolerance filter
        as_of = self._toleranceFilter(as_of)
        return as_of


# Helper functions

def __getSparkPlan(df: DataFrame, spark: SparkSession) -> str:
    """
    Internal helper function to obtain the Spark plan for the input data frame

    Parameters
    :param df - input Spark data frame - the AS OF join has 2 data frames; this will be called for each
    :param spark - Spark session which is used to query the view obtained from the Spark data frame
    """

    df.createOrReplaceTempView("view")
    plan = spark.sql("explain cost select * from view").collect()[0][0]

    return plan


def __getBytesFromPlan(df: DataFrame, spark: SparkSession) -> float:
    """
    Internal helper function to obtain how many bytes in memory the Spark data
    frame is likely to take up. This is an upper bound and is obtained from the
    plan details in Spark

    Parameters
    :param df - input Spark data frame - the AS OF join has 2 data frames; this will be called for each
    :param spark - Spark session which is used to query the view obtained from the Spark data frame
    """

    plan = __getSparkPlan(df, spark)

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


def choose_as_of_join_strategy(left_tsdf: t_tsdf.TSDF,
                               right_tsdf: t_tsdf.TSDF,
                               left_prefix: Optional[str] = None,
                               right_prefix: str = "right",
                               tsPartitionVal: Optional[int] = None,
                               fraction: float = 0.5,
                               skipNulls: bool = True,
                               sql_join_opt: bool = False,
                               suppress_null_warning: bool = False,
                               tolerance: Optional[int] = None,) -> "AsOfJoiner":
    """
    Returns an AsOfJoiner object based on the parameters passed in
    """

    # test if the broadcast join will be efficient
    if sql_join_opt:
        spark = SparkSession.builder.getOrCreate()
        left_bytes = __getBytesFromPlan(left_tsdf.df, spark)
        right_bytes = __getBytesFromPlan(right_tsdf.df, spark)

        bytes_threshold = __DEFAULT_BROADCAST_BYTES_THRESHOLD
        if (left_bytes < bytes_threshold) or (right_bytes < bytes_threshold):
            return BroadcastAsOfJoiner(spark, left_prefix, right_prefix)
