"""
As-of join strategies for Tempo.

This module provides various strategies for performing as-of joins on TSDFs:
- BroadcastAsOfJoiner: For small datasets that fit in memory
- UnionSortFilterAsOfJoiner: Default strategy for general cases
- SkewAsOfJoiner: For handling skewed data distributions
"""

import copy
import logging
import re
from abc import ABC, abstractmethod
from functools import reduce
from typing import Optional, Tuple, Union, List, TYPE_CHECKING

import pyspark.sql.functions as sfn
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql.window import Window

# Import related components
from tempo.tsschema import CompositeTSIndex, TSSchema
from tempo.timeunit import TimeUnit

# Avoid circular import
if TYPE_CHECKING:
    from tempo.tsdf import TSDF

logger = logging.getLogger(__name__)

# Constants for row indicators
_DEFAULT_COMBINED_TS_COLNAME = "combined_ts"
_DEFAULT_RIGHT_ROW_COLNAME = "righthand_tbl_row"
_LEFT_HAND_ROW_INDICATOR = 1
_RIGHT_HAND_ROW_INDICATOR = -1

# Default broadcast threshold: 30MB
_DEFAULT_BROADCAST_BYTES_THRESHOLD = 30 * 1024 * 1024


class _AsOfJoinCompositeTSIndex(CompositeTSIndex):
    """
    Special CompositeTSIndex subclass for as-of joins.
    Does not implement rangeExpr as it's not needed for join operations.
    """

    @property
    def unit(self) -> Optional[TimeUnit]:
        return None

    def rangeExpr(self, reverse: bool = False) -> Column:
        raise NotImplementedError(
            "rangeExpr is not defined for as-of join composite ts index"
        )


class AsOfJoiner(ABC):
    """
    Abstract base class for as-of join strategies.

    This class defines the common interface and behavior for all as-of join strategies.
    Subclasses implement specific join algorithms (broadcast, union-sort-filter, skew).
    """

    def __init__(self, left_prefix: str = "left", right_prefix: str = "right"):
        """
        Initialize the AsOfJoiner with column prefixes.

        :param left_prefix: Prefix for left DataFrame columns
        :param right_prefix: Prefix for right DataFrame columns
        """
        self.left_prefix = left_prefix
        self.right_prefix = right_prefix

    def __call__(self, left: 'TSDF', right: 'TSDF') -> 'TSDF':
        """
        Execute the as-of join.

        :param left: Left TSDF
        :param right: Right TSDF
        :return: Joined TSDF
        """
        # Check if the TSDFs are joinable
        self._checkAreJoinable(left, right)
        # Prefix overlapping columns to avoid conflicts
        left, right = self._prefixOverlappingColumns(left, right)
        # Perform the join
        return self._join(left, right)

    def commonSeriesIDs(self, left: "TSDF", right: "TSDF") -> set:
        """
        Returns the common series IDs between the left and right TSDFs.

        :param left: Left TSDF
        :param right: Right TSDF
        :return: Set of common series IDs
        """
        return set(left.series_ids).intersection(set(right.series_ids))

    def _prefixableColumns(self, left: "TSDF", right: "TSDF") -> set:
        """
        Returns the overlapping columns in the left and right TSDFs
        not including overlapping series IDs.
        """
        return set(left.columns).intersection(
            set(right.columns)
        ) - self.commonSeriesIDs(left, right)

    def _prefixColumns(
        self, tsdf: "TSDF", prefixable_cols: set, prefix: str
    ) -> "TSDF":
        """
        Prefixes the columns in the TSDF.
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
        self, left: "TSDF", right: "TSDF"
    ) -> tuple["TSDF", "TSDF"]:
        """
        Prefixes the overlapping columns in the left and right TSDFs.
        """
        # find the columns to prefix
        prefixable_cols = self._prefixableColumns(left, right)

        # prefix columns (if we have a prefix to apply)
        left_prefixed = self._prefixColumns(left, prefixable_cols, self.left_prefix)
        right_prefixed = self._prefixColumns(right, prefixable_cols, self.right_prefix)

        return left_prefixed, right_prefixed

    def _checkAreJoinable(self, left: "TSDF", right: "TSDF") -> None:
        """
        Checks if the left and right TSDFs are joinable.
        Raises an exception if they are not compatible.

        :param left: Left TSDF
        :param right: Right TSDF
        :raises ValueError: If TSDFs are not joinable
        """
        # Check schema equivalence
        if left.ts_schema != right.ts_schema:
            raise ValueError(
                f"Timestamp schemas must match. "
                f"Left: {left.ts_schema}, Right: {right.ts_schema}"
            )

        # Check series IDs compatibility
        if set(left.series_ids) != set(right.series_ids):
            raise ValueError(
                f"Series IDs must match. "
                f"Left: {left.series_ids}, Right: {right.series_ids}"
            )

    @abstractmethod
    def _join(self, left: "TSDF", right: "TSDF") -> "TSDF":
        """
        Performs the actual join operation.
        Must be implemented by subclasses.

        :param left: Left TSDF
        :param right: Right TSDF
        :return: Joined TSDF
        """
        pass


class BroadcastAsOfJoiner(AsOfJoiner):
    """
    Broadcast join strategy for as-of joins.

    Efficient for small datasets that can fit in memory.
    Uses Spark's broadcast join optimization.
    """

    def __init__(
        self,
        spark: SparkSession,
        left_prefix: str = "left",
        right_prefix: str = "right",
        range_join_bin_size: int = 60,
    ):
        """
        Initialize the BroadcastAsOfJoiner.

        :param spark: SparkSession
        :param left_prefix: Prefix for left DataFrame columns
        :param right_prefix: Prefix for right DataFrame columns
        :param range_join_bin_size: Bin size for range join optimization
        """
        super().__init__(left_prefix, right_prefix)
        self.spark = spark
        self.range_join_bin_size = range_join_bin_size

    def _join(self, left: "TSDF", right: "TSDF") -> "TSDF":
        """
        Performs broadcast as-of join.

        :param left: Left TSDF
        :param right: Right TSDF
        :return: Joined TSDF
        """
        # Set the range join bin size for optimization
        self.spark.conf.set(
            "spark.databricks.optimizer.rangeJoin.binSize",
            str(self.range_join_bin_size),
        )

        # Create window for lead calculation
        w = right.baseWindow()

        # Get the comparable expression for the timestamp index
        # This handles both simple and composite timestamp indexes
        right_comparable_expr = right.ts_index.comparableExpr()

        # CRITICAL FIX: Handle composite indexes properly
        # For composite indexes, comparableExpr() returns a list
        # We need the first element which is the most comparable field
        # (e.g., double_ts for nanosecond precision timestamps)
        if isinstance(right_comparable_expr, list):
            if len(right_comparable_expr) > 0:
                right_comparable_expr = right_comparable_expr[0]
            else:
                # Fallback to column name if no comparable expression
                right_comparable_expr = sfn.col(right.ts_index.colname)

        # Create lead column for range-based joining
        lead_colname = "lead_" + right.ts_index.colname
        right_with_lead = right.withColumn(
            lead_colname, sfn.lead(right_comparable_expr).over(w)
        )

        # No need for row number here as we'll handle duplicates after the join

        # Perform the join
        join_series_ids = self.commonSeriesIDs(left, right)

        # Get comparable expressions for timestamp comparison
        left_ts_expr = left.ts_index.comparableExpr()
        if isinstance(left_ts_expr, list):
            left_ts_expr = left_ts_expr[0]

        right_ts_expr = right.ts_index.comparableExpr()
        if isinstance(right_ts_expr, list):
            right_ts_expr = right_ts_expr[0]

        # Create between condition that handles NULL lead values
        # When lead is NULL, it means this is the last row in the partition
        # and should match all future left timestamps
        between_condition = (
            (left_ts_expr >= right_ts_expr) &
            (sfn.col(lead_colname).isNull() | (left_ts_expr < sfn.col(lead_colname)))
        )

        # Join and filter
        res_df = (
            left.df.join(right_with_lead.df, list(join_series_ids))
            .where(between_condition)
        )

        # Drop the lead column
        res_df = res_df.drop(lead_colname)

        # Return with the left-hand schema
        from tempo.tsdf import TSDF
        return TSDF(res_df, ts_schema=left.ts_schema)


class UnionSortFilterAsOfJoiner(AsOfJoiner):
    """
    Union-Sort-Filter join strategy for as-of joins.

    Default strategy that works for all cases. Unions the DataFrames,
    sorts by timestamp, and filters to get the last matching right row
    for each left row.
    """

    def __init__(
        self,
        left_prefix: str = "left",
        right_prefix: str = "right",
        skipNulls: bool = True,
        tolerance: Optional[int] = None,
    ):
        """
        Initialize the UnionSortFilterAsOfJoiner.

        :param left_prefix: Prefix for left DataFrame columns
        :param right_prefix: Prefix for right DataFrame columns
        :param skipNulls: Whether to skip null values in the join
        :param tolerance: Tolerance window in seconds (optional)
        """
        super().__init__(left_prefix, right_prefix)
        self.skipNulls = skipNulls
        self.tolerance = tolerance

    def _appendNullColumns(self, tsdf: "TSDF", cols: set) -> "TSDF":
        """
        Appends null columns to the TSDF.

        :param tsdf: "TSDF" to modify
        :param cols: Set of columns to add as nulls
        :return: "TSDF" with added null columns
        """
        return reduce(
            lambda cur_tsdf, col: cur_tsdf.withColumn(col, sfn.lit(None)),
            cols,
            tsdf
        )

    def _combine(self, left: "TSDF", right: "TSDF") -> "TSDF":
        """
        Combines the left and right TSDFs into a single TSDF.

        :param left: Left TSDF
        :param right: Right TSDF
        :return: Combined TSDF with special timestamp index
        """
        # Union the DataFrames
        unioned = left.unionByName(right)

        # Get comparable expressions for both sides
        left_comps = left.ts_index.comparableExpr()
        right_comps = right.ts_index.comparableExpr()

        # Ensure list format
        if not isinstance(left_comps, list):
            left_comps = [left_comps]
        if not isinstance(right_comps, list):
            right_comps = [right_comps]

        # Get component names
        if isinstance(left.ts_index, CompositeTSIndex):
            comp_names = left.ts_index.component_fields
        else:
            comp_names = [left.ts_index.colname]

        # Coalesce components
        combined_comps = [
            sfn.coalesce(lc, rc).alias(cn)
            for lc, rc, cn in zip(left_comps, right_comps, comp_names)
        ]

        # Build an expression for a right-hand side indicator field
        right_ind = (
            sfn.when(
                sfn.col(left.ts_index.colname).isNotNull(), _LEFT_HAND_ROW_INDICATOR
            )
            .otherwise(_RIGHT_HAND_ROW_INDICATOR)
            .alias(_DEFAULT_RIGHT_ROW_COLNAME)
        )

        # Combine all into a single struct column
        with_combined_ts = unioned.withColumn(
            _DEFAULT_COMBINED_TS_COLNAME, sfn.struct(*combined_comps, right_ind)
        )

        # Construct a CompositeTSIndex from the combined ts index
        combined_ts_col = with_combined_ts.df.schema[_DEFAULT_COMBINED_TS_COLNAME]
        combined_comp_names = comp_names + [_DEFAULT_RIGHT_ROW_COLNAME]
        combined_tsidx = _AsOfJoinCompositeTSIndex(
            combined_ts_col, *combined_comp_names
        )

        # Put it all together in a new TSDF
        # Import here to avoid circular dependency
        from tempo.tsdf import TSDF
        return TSDF(
            with_combined_ts.df,
            ts_schema=TSSchema(combined_tsidx, left.series_ids)
        )

    def _filterLastRightRow(
        self, combined: "TSDF", right_cols: set, last_left_tsschema: TSSchema
    ) -> "TSDF":
        """
        Filters out the last right-hand row for each left-hand row.

        :param combined: Combined TSDF
        :param right_cols: Set of right-only columns
        :param last_left_tsschema: Original left TSDF schema
        :return: Filtered TSDF with as-of join results
        """
        # Find the last value for each column in the right-hand side
        w = combined.allBeforeWindow()

        # Type assertion: ts_index is a CompositeTSIndex for as-of joins
        assert isinstance(combined.ts_index, CompositeTSIndex)
        right_row_field = combined.ts_index.fieldPath(_DEFAULT_RIGHT_ROW_COLNAME)

        if self.skipNulls:
            # Use Spark's last() with ignoreNulls=True
            last_right_cols = [
                sfn.last(col, True).over(w).alias(col) for col in right_cols
            ]
        else:
            # Include nulls in the window calculation
            last_right_cols = [
                sfn.last(
                    sfn.when(
                        sfn.col(right_row_field) == _RIGHT_HAND_ROW_INDICATOR,
                        sfn.struct(col),
                    ).otherwise(None),
                    True,
                )
                .over(w)[col]
                .alias(col)
                for col in right_cols
            ]

        # Get the last right-hand row for each left-hand row
        non_right_cols = list(set(combined.columns) - right_cols)
        last_right_vals = combined.select(*(non_right_cols + last_right_cols))

        # Filter out the last right-hand row for each left-hand row
        as_of_df = last_right_vals.df.where(
            sfn.col(right_row_field) == _LEFT_HAND_ROW_INDICATOR
        ).drop(_DEFAULT_COMBINED_TS_COLNAME)

        # Return with the left-hand schema
        # Import here to avoid circular dependency
        from tempo.tsdf import TSDF
        return TSDF(as_of_df, ts_schema=copy.deepcopy(last_left_tsschema))

    def _toleranceFilter(
        self,
        as_of: "TSDF",
        left_ts_col: str,
        right_ts_col: str,
        right_columns: List[str]
    ) -> "TSDF":
        """
        Filters out rows from the as_of TSDF that are outside the tolerance.

        :param as_of: As-of joined TSDF
        :param left_ts_col: Left timestamp column name
        :param right_ts_col: Right timestamp column name
        :param right_columns: List of right column names
        :return: Filtered TSDF
        """
        if self.tolerance is None:
            return as_of

        df = as_of.df

        # Calculate time difference (in seconds)
        tolerance_condition = (
            df[left_ts_col].cast("double") - df[right_ts_col].cast("double")
            > self.tolerance
        )

        # Set right columns to null for rows outside tolerance
        for right_col in right_columns:
            if right_col != right_ts_col:
                df = df.withColumn(
                    right_col,
                    sfn.when(tolerance_condition, sfn.lit(None)).otherwise(df[right_col])
                )

        # Finally, set right timestamp column to null
        df = df.withColumn(
            right_ts_col,
            sfn.when(tolerance_condition, sfn.lit(None)).otherwise(df[right_ts_col])
        )

        as_of.df = df
        return as_of

    def _join(self, left: "TSDF", right: "TSDF") -> "TSDF":
        """
        Performs union-sort-filter as-of join.

        :param left: Left TSDF
        :param right: Right TSDF
        :return: Joined TSDF
        """
        # Find the new columns to add to the left and right TSDFs
        right_only_cols = set(right.columns) - set(left.columns)
        left_only_cols = set(left.columns) - set(right.columns)

        # Append null columns to the left and right TSDFs
        extended_left = self._appendNullColumns(left, right_only_cols)
        extended_right = self._appendNullColumns(right, left_only_cols)

        # Combine the left and right TSDFs
        combined = self._combine(extended_left, extended_right)

        # Filter out the last right-hand row for each left-hand row
        as_of = self._filterLastRightRow(
            combined, right_only_cols, extended_left.ts_schema
        )

        # Apply tolerance filter if specified
        if self.tolerance is not None:
            # Get the timestamp column names with prefixes
            left_ts_col = f"{self.left_prefix}_{left.ts_col}" if self.left_prefix else left.ts_col
            right_ts_col = f"{self.right_prefix}_{right.ts_col}" if self.right_prefix else right.ts_col

            # Get list of right columns with prefixes
            right_columns = []
            for col in right_only_cols:
                if self.right_prefix:
                    right_columns.append(f"{self.right_prefix}_{col}")
                else:
                    right_columns.append(col)

            as_of = self._toleranceFilter(as_of, left_ts_col, right_ts_col, right_columns)

        return as_of


class SkewAsOfJoiner(UnionSortFilterAsOfJoiner):
    """
    Skew-aware join strategy for as-of joins.

    Optimized for datasets with skewed distributions.
    Uses time-based partitioning to handle skew.
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
        """
        Initialize the SkewAsOfJoiner.

        :param left_prefix: Prefix for left DataFrame columns
        :param right_prefix: Prefix for right DataFrame columns
        :param skipNulls: Whether to skip null values in the join
        :param tolerance: Tolerance window in seconds (optional)
        :param tsPartitionVal: Time partition value in seconds
        :param fraction: Overlap fraction for partitions
        """
        super().__init__(left_prefix, right_prefix, skipNulls, tolerance)
        self.tsPartitionVal = tsPartitionVal
        self.fraction = fraction

    def _applyTimePartitions(self, tsdf: "TSDF") -> "TSDF":
        """
        Apply time-based partitioning with overlap to handle skew.

        Creates overlapping time windows to handle edge cases where
        timestamps near partition boundaries need data from adjacent partitions.

        :param tsdf: "TSDF" to partition
        :return: Partitioned TSDF with overlap
        """
        if self.tsPartitionVal is None:
            return tsdf

        # Cast timestamp to double for partitioning
        partition_df = (
            tsdf.df.withColumn(
                "ts_col_double", sfn.col(tsdf.ts_col).cast("double")
            )
            .withColumn(
                "ts_partition",
                sfn.lit(self.tsPartitionVal)
                * (sfn.col("ts_col_double") / sfn.lit(self.tsPartitionVal)).cast("integer"),
            )
            .withColumn(
                "partition_remainder",
                (sfn.col("ts_col_double") - sfn.col("ts_partition"))
                / sfn.lit(self.tsPartitionVal),
            )
            .withColumn("is_original", sfn.lit(1))
        ).cache()  # Cache because it's used twice

        # Add overlap: include [1 - fraction] of previous time partition in next partition
        remainder_df = (
            partition_df.filter(sfn.col("partition_remainder") >= sfn.lit(1 - self.fraction))
            .withColumn(
                "ts_partition", sfn.col("ts_partition") + sfn.lit(self.tsPartitionVal)
            )
            .withColumn("is_original", sfn.lit(0))
        )

        # Union original and overlapped data
        df = partition_df.union(remainder_df).drop(
            "partition_remainder", "ts_col_double"
        )

        # Return TSDF with ts_partition added to series IDs
        # Import here to avoid circular dependency
        from tempo.tsdf import TSDF
        return TSDF(
            df, ts_col=tsdf.ts_col, series_ids=tsdf.series_ids + ["ts_partition"]
        )

    def _join(self, left: "TSDF", right: "TSDF") -> "TSDF":
        """
        Performs skew-aware as-of join using time-based partitioning.

        :param left: Left TSDF
        :param right: Right TSDF
        :return: Joined TSDF
        """
        # Log skew handling
        if self.tsPartitionVal:
            logger.info(f"Using SkewAsOfJoiner with partition value {self.tsPartitionVal}")

        # Find the new columns to add to the left and right TSDFs
        right_only_cols = set(right.columns) - set(left.columns)
        left_only_cols = set(left.columns) - set(right.columns)

        # Append null columns to the left and right TSDFs
        extended_left = self._appendNullColumns(left, right_only_cols)
        extended_right = self._appendNullColumns(right, left_only_cols)

        # Combine the left and right TSDFs
        combined = self._combine(extended_left, extended_right)

        # Apply time-based partitioning if specified
        if self.tsPartitionVal:
            combined = self._applyTimePartitions(combined)

        # Filter to get last right row for each left row
        as_of = self._filterLastRightRow(
            combined, right_only_cols, extended_left.ts_schema
        )

        # Remove overlapped data if time partitioning was used
        if self.tsPartitionVal:
            as_of.df = as_of.df.filter(sfn.col("is_original") == 1).drop(
                "ts_partition", "is_original"
            )

        # Apply tolerance filter if specified
        if self.tolerance is not None:
            # Get the timestamp column names with prefixes
            left_ts_col = f"{self.left_prefix}_{left.ts_col}" if self.left_prefix else left.ts_col
            right_ts_col = f"{self.right_prefix}_{right.ts_col}" if self.right_prefix else right.ts_col

            # Get list of right columns with prefixes
            right_columns = []
            for col in right_only_cols:
                if self.right_prefix:
                    right_columns.append(f"{self.right_prefix}_{col}")
                else:
                    right_columns.append(col)

            as_of = self._toleranceFilter(as_of, left_ts_col, right_ts_col, right_columns)

        return as_of


# Helper functions for strategy selection

def get_spark_plan(df: DataFrame, spark: SparkSession) -> str:
    """
    Get the Spark execution plan for a DataFrame.

    :param df: Input DataFrame
    :param spark: SparkSession
    :return: Spark plan as string
    """
    df.createOrReplaceTempView("view")
    plan = spark.sql("explain cost select * from view").collect()[0][0]
    return plan


def get_bytes_from_plan(df: DataFrame, spark: SparkSession) -> float:
    """
    Extract estimated size in bytes from Spark execution plan.

    :param df: Input DataFrame
    :param spark: SparkSession
    :return: Size in bytes
    """
    plan = get_spark_plan(df, spark)

    # Extract sizeInBytes from plan
    search_result = re.search(r"sizeInBytes=([0-9.]+)\s*([A-Za-z]+)", plan, re.MULTILINE)
    if search_result is None:
        logger.warning("Unable to obtain sizeInBytes from Spark plan")
        return float('inf')  # Return large number to avoid broadcast

    size = float(search_result.group(1))
    units = search_result.group(2)

    # Convert to bytes
    if units == "GiB":
        plan_bytes = size * 1024 * 1024 * 1024
    elif units == "MiB":
        plan_bytes = size * 1024 * 1024
    elif units == "KiB":
        plan_bytes = size * 1024
    else:
        plan_bytes = size

    return plan_bytes


def choose_as_of_join_strategy(
    left_tsdf: "TSDF",
    right_tsdf: "TSDF",
    left_prefix: Optional[str] = None,
    right_prefix: str = "right",
    tsPartitionVal: Optional[int] = None,
    fraction: float = 0.5,
    skipNulls: bool = True,
    sql_join_opt: bool = False,
    tolerance: Optional[int] = None,
) -> AsOfJoiner:
    """
    Automatically choose the optimal as-of join strategy based on data characteristics.

    Decision tree:
    1. If sql_join_opt and data < 30MB: BroadcastAsOfJoiner
    2. If tsPartitionVal is set: SkewAsOfJoiner
    3. Default: UnionSortFilterAsOfJoiner

    :param left_tsdf: Left TSDF
    :param right_tsdf: Right TSDF
    :param left_prefix: Prefix for left columns
    :param right_prefix: Prefix for right columns
    :param tsPartitionVal: Time partition value for skew handling
    :param fraction: Overlap fraction for partitions
    :param skipNulls: Whether to skip nulls
    :param sql_join_opt: Whether to use SQL optimization
    :param tolerance: Tolerance window in seconds
    :return: Appropriate AsOfJoiner instance
    """
    try:
        # Test if the broadcast join will be efficient
        if sql_join_opt:
            try:
                spark = SparkSession.builder.getOrCreate()
                left_bytes = get_bytes_from_plan(left_tsdf.df, spark)
                right_bytes = get_bytes_from_plan(right_tsdf.df, spark)

                # Use broadcast if either DataFrame is small enough
                if (left_bytes < _DEFAULT_BROADCAST_BYTES_THRESHOLD) or \
                   (right_bytes < _DEFAULT_BROADCAST_BYTES_THRESHOLD):
                    logger.info(
                        f"Using BroadcastAsOfJoiner "
                        f"(left: {left_bytes/1024/1024:.2f}MB, "
                        f"right: {right_bytes/1024/1024:.2f}MB)"
                    )
                    return BroadcastAsOfJoiner(spark, left_prefix or "left", right_prefix)
            except Exception as e:
                logger.warning(f"Could not estimate DataFrame size: {e}")
                # Fall through to other strategies

        # Use the skew join if the partition value is passed in
        if tsPartitionVal is not None:
            logger.info(f"Using SkewAsOfJoiner with partition value {tsPartitionVal}")
            return SkewAsOfJoiner(
                left_prefix=left_prefix or "left",
                right_prefix=right_prefix,
                skipNulls=skipNulls,
                tolerance=tolerance,
                tsPartitionVal=tsPartitionVal,
                fraction=fraction,
            )

        # Default to union-sort-filter join
        logger.info("Using default UnionSortFilterAsOfJoiner")
        return UnionSortFilterAsOfJoiner(
            left_prefix or "left",
            right_prefix,
            skipNulls,
            tolerance
        )

    except Exception as e:
        logger.error(f"Strategy selection failed: {e}")
        # Always fall back to safe default
        return UnionSortFilterAsOfJoiner(
            left_prefix or "left",
            right_prefix,
            skipNulls,
            tolerance
        )