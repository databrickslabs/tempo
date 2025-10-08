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
from typing import Any, Optional, Tuple, Union, List

import pyspark.sql.functions as sfn
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql.window import Window

# Import related components
from tempo.tsschema import CompositeTSIndex, TSSchema
from tempo.timeunit import TimeUnit

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

    def __call__(self, left, right) -> Tuple[DataFrame, TSSchema]:
        """
        Execute the as-of join.

        :param left: Left TSDF
        :param right: Right TSDF
        :return: Tuple of (joined DataFrame, TSSchema)
        """
        # Check if the TSDFs are joinable
        self._checkAreJoinable(left, right)
        # Prefix overlapping columns to avoid conflicts
        left, right = self._prefixOverlappingColumns(left, right)
        # Perform the join
        return self._join(left, right)

    def commonSeriesIDs(self, left, right) -> set:
        """
        Returns the common series IDs between the left and right TSDFs.

        :param left: Left TSDF
        :param right: Right TSDF
        :return: Set of common series IDs
        """
        return set(left.series_ids).intersection(set(right.series_ids))

    def _prefixableColumns(self, left, right) -> set:
        """
        Returns the overlapping columns in the left and right TSDFs
        not including overlapping series IDs.
        """
        return set(left.columns).intersection(
            set(right.columns)
        ) - self.commonSeriesIDs(left, right)

    def _prefixColumns(self, tsdf, prefixable_cols: set, prefix: str):
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

    def _prefixOverlappingColumns(self, left, right) -> tuple:
        """
        Prefixes the overlapping columns in the left and right TSDFs.
        """
        # find the columns to prefix
        prefixable_cols = self._prefixableColumns(left, right)

        # prefix columns (if we have a prefix to apply)
        left_prefixed = self._prefixColumns(left, prefixable_cols, self.left_prefix)
        right_prefixed = self._prefixColumns(right, prefixable_cols, self.right_prefix)

        return left_prefixed, right_prefixed

    def _checkAreJoinable(self, left, right) -> None:
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
    def _join(self, left, right) -> Tuple[DataFrame, TSSchema]:
        """
        Performs the actual join operation.
        Must be implemented by subclasses.

        :param left: Left TSDF
        :param right: Right TSDF
        :return: Tuple of (joined DataFrame, TSSchema)
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

    def _join(self, left, right) -> Tuple[DataFrame, TSSchema]:
        """
        Performs broadcast as-of join.

        :param left: Left TSDF
        :param right: Right TSDF
        :return: Tuple of (joined DataFrame, TSSchema)
        """
        # Handle empty right DataFrame - return left with null right columns
        if right.df.rdd.isEmpty():
            # Create result with left DataFrame and null columns from right
            result_df = left.df

            # Add all non-key columns from right as nulls with proper prefixes
            for field in right.df.schema.fields:
                col_name = field.name
                # Skip series columns and timestamp
                if col_name not in right.series_ids and col_name != right.ts_col:
                    # Apply prefix if needed
                    out_col_name = (
                        f"{self.right_prefix}_{col_name}"
                        if self.right_prefix
                        else col_name
                    )
                    result_df = result_df.withColumn(
                        out_col_name, sfn.lit(None).cast(field.dataType)
                    )

            # Add right timestamp column with prefix
            right_ts_name = (
                f"{self.right_prefix}_{right.ts_col}"
                if self.right_prefix
                else f"right_{right.ts_col}"
            )
            result_df = result_df.withColumn(
                right_ts_name,
                sfn.lit(None).cast(right.df.schema[right.ts_col].dataType),
            )

            return result_df, TSSchema(ts_idx=left.ts_index, series_ids=left.series_ids)

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

        # Alias DataFrames to avoid ambiguity when column names overlap
        left_aliased = left.df.alias("l")
        right_aliased = right_with_lead.df.alias("r")

        # Get timestamp column names (may be prefixed by parent)
        left_ts_col = left.ts_col
        right_ts_col = right.ts_col

        # Create between condition using aliased column references
        # This avoids ambiguity when both DataFrames have the same column names
        between_condition = (
            sfn.col(f"l.{left_ts_col}") >= sfn.col(f"r.{right_ts_col}")
        ) & (
            sfn.col(f"r.{lead_colname}").isNull()
            | (sfn.col(f"l.{left_ts_col}") < sfn.col(f"r.{lead_colname}"))
        )

        # Join and filter
        if join_series_ids:
            # Build join condition on series columns
            join_condition = None
            for series_col in join_series_ids:
                col_condition = sfn.col(f"l.{series_col}") == sfn.col(f"r.{series_col}")
                join_condition = (
                    col_condition
                    if join_condition is None
                    else join_condition & col_condition
                )

            # Join on series columns, then filter by temporal condition
            res_df = left_aliased.join(right_aliased, on=join_condition).where(
                between_condition
            )
        else:
            # For single series, we need to compare all records
            # Use a LEFT JOIN with temporal condition directly to let Spark optimize
            res_df = left_aliased.join(right_aliased, on=between_condition, how="left")

        # Drop the lead column
        res_df = res_df.drop(lead_colname)

        # Select columns to remove the "l." and "r." prefixes from the join
        # The left and right TSDFs were already prefixed by the parent class,
        # so we just need to select them with their existing names
        final_cols = []
        added_cols = set()

        # Add all left columns
        for col in left.columns:
            final_cols.append(sfn.col(f"l.{col}").alias(col))
            added_cols.add(col)

        # Add right columns (excluding series columns and already-added columns)
        for col in right.columns:
            if col not in right.series_ids and col not in added_cols:
                final_cols.append(sfn.col(f"r.{col}").alias(col))

        res_df = res_df.select(*final_cols)

        # Return DataFrame and schema tuple
        return res_df, left.ts_schema


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

    def _appendNullColumns(self, tsdf, cols: set):
        """
        Appends null columns to the TSDF.

        :param tsdf: TSDF to modify
        :param cols: Set of columns to add as nulls
        :return: TSDF with added null columns
        """
        return reduce(
            lambda cur_tsdf, col: cur_tsdf.withColumn(col, sfn.lit(None)), cols, tsdf
        )

    def _combine(self, left, right):
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
        # We need TSDF for internal operations only
        from tempo.tsdf import TSDF

        return TSDF(
            with_combined_ts.df, ts_schema=TSSchema(combined_tsidx, left.series_ids)
        )

    def _filterLastRightRow(
        self, combined, right_cols: set, last_left_tsschema: TSSchema
    ):
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
            # When skipNulls=True, we want to skip entire rows where any value column is NULL
            # Create a condition that checks if any value column (excluding timestamp) is NULL
            # Note: right_cols includes both timestamp and value columns from the right side

            # Get value columns (exclude timestamp columns from right_cols)
            # We assume timestamp columns contain 'timestamp' in their name
            value_cols = [col for col in right_cols if "timestamp" not in col.lower()]

            # Create a condition: row is valid if all value columns are non-null
            if value_cols:
                # Check if any value column is NULL
                any_null_condition = sfn.lit(False)
                for col in value_cols:
                    any_null_condition = any_null_condition | sfn.col(col).isNull()

                # Use last() with a conditional struct that excludes rows with NULL values
                last_right_cols = []
                for col in right_cols:
                    last_right_cols.append(
                        sfn.last(
                            sfn.when(
                                (sfn.col(right_row_field) == _RIGHT_HAND_ROW_INDICATOR)
                                & ~any_null_condition,
                                sfn.struct(col),
                            ).otherwise(None),
                            True,
                        )
                        .over(w)[col]
                        .alias(col)
                    )
            else:
                # No value columns to check, use simple last with ignoreNulls
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

        # We need TSDF for internal use but return raw data
        from tempo.tsdf import TSDF

        return TSDF(as_of_df, ts_schema=copy.deepcopy(last_left_tsschema))

    def _toleranceFilter(
        self, as_of, left_ts_col: str, right_ts_col: str, right_columns: List[str]
    ):
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
                    sfn.when(tolerance_condition, sfn.lit(None)).otherwise(
                        df[right_col]
                    ),
                )

        # Finally, set right timestamp column to null
        df = df.withColumn(
            right_ts_col,
            sfn.when(tolerance_condition, sfn.lit(None)).otherwise(df[right_ts_col]),
        )

        as_of.df = df
        return as_of

    def _join(self, left, right) -> Tuple[DataFrame, TSSchema]:
        """
        Performs union-sort-filter as-of join.

        :param left: Left TSDF
        :param right: Right TSDF
        :return: Tuple of (joined DataFrame, TSSchema)
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
            # Get the actual timestamp column names (already prefixed by _prefixOverlappingColumns)
            left_ts_col = left.ts_col
            right_ts_col = right.ts_col

            # Get list of right columns (already prefixed if needed)
            right_columns = list(right_only_cols)

            as_of = self._toleranceFilter(
                as_of, left_ts_col, right_ts_col, right_columns
            )

        # Return DataFrame and schema tuple
        return as_of.df, as_of.ts_schema


class SkewAsOfJoiner(AsOfJoiner):
    """
    Skew-aware join strategy for as-of joins.

    Leverages Spark's Adaptive Query Execution (AQE) to automatically handle
    skewed data distributions. For extreme skew cases, provides additional
    strategies including key separation and salting.

    This implementation prioritizes simplicity and Spark's built-in optimizations
    over manual partitioning schemes.
    """

    def __init__(
        self,
        spark: SparkSession,
        left_prefix: str = "left",
        right_prefix: str = "right",
        skipNulls: bool = True,
        tolerance: Optional[int] = None,
        skew_threshold: float = 0.2,  # If one key has >20% of data
        enable_salting: bool = False,
        salt_buckets: int = 10,
        tsPartitionVal: Optional[int] = None,  # Keep for backward compatibility
    ):
        """
        Initialize the SkewAsOfJoiner.

        :param spark: SparkSession instance
        :param left_prefix: Prefix for left DataFrame columns
        :param right_prefix: Prefix for right DataFrame columns
        :param skipNulls: Whether to skip null values in the join
        :param tolerance: Tolerance window in seconds (optional)
        :param skew_threshold: Threshold for detecting skewed keys (fraction of total data)
        :param enable_salting: Whether to use salting for extreme skew
        :param salt_buckets: Number of salt buckets when salting is enabled
        :param tsPartitionVal: [Deprecated] Time partition value in seconds (kept for compatibility)
        """
        super().__init__(left_prefix, right_prefix)
        self.spark = spark
        self.skipNulls = skipNulls
        self.tolerance = tolerance
        self.skew_threshold = skew_threshold
        self.enable_salting = enable_salting
        self.salt_buckets = salt_buckets
        self.tsPartitionVal = tsPartitionVal  # Deprecated but kept for compatibility

        # Configure AQE settings for this session
        self._configureAQE()

    def _configureAQE(self):
        """
        Configure Adaptive Query Execution settings for optimal skew handling.
        """
        # Enable AQE and skew join optimization
        self.spark.conf.set("spark.sql.adaptive.enabled", "true")
        self.spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")

        # Configure skew detection thresholds
        # A partition is considered skewed if it's 5x larger than median
        self.spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5")
        # Or if it's larger than 256MB
        self.spark.conf.set(
            "spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256MB"
        )

        # Enable coalescing of small partitions
        self.spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")

        logger.info("Configured AQE for skew handling")

    def _detectSkewedKeys(self, left, right) -> List[Any]:
        """
        Detect heavily skewed keys in the data.

        :param left: Left TSDF
        :param right: Right TSDF
        :return: List of skewed key values
        """
        if not left.series_ids:
            return []  # No series keys to be skewed

        # Sample the data if it's too large
        sample_fraction = min(1.0, 1000000 / left.df.count())
        if sample_fraction < 1.0:
            left_sample = left.df.sample(fraction=sample_fraction)
        else:
            left_sample = left.df

        # Group by series keys and count
        series_cols = left.series_ids
        key_counts = (
            left_sample.groupBy(*series_cols)
            .count()
            .withColumn("total_count", sfn.sum("count").over(Window.partitionBy()))
            .withColumn("fraction", sfn.col("count") / sfn.col("total_count"))
        )

        # Find keys that exceed the skew threshold
        skewed_keys = (
            key_counts.filter(sfn.col("fraction") > self.skew_threshold)
            .select(*series_cols)
            .collect()
        )

        if skewed_keys:
            logger.info(
                f"Detected {len(skewed_keys)} skewed keys exceeding {self.skew_threshold:.0%} threshold"
            )
            # Convert Row objects to tuples for easier handling
            return [tuple(row) for row in skewed_keys]

        return []

    def _join(self, left, right) -> Tuple[DataFrame, TSSchema]:
        """
        Performs skew-aware as-of join using AQE and optional skew handling strategies.

        :param left: Left TSDF
        :param right: Right TSDF
        :return: Tuple of (joined DataFrame, TSSchema)
        """
        logger.info("Using SkewAsOfJoiner with AQE optimization")

        # Detect if we have severely skewed keys
        skewed_keys = (
            self._detectSkewedKeys(left, right) if self.skew_threshold < 1.0 else []
        )

        if not skewed_keys:
            # No extreme skew detected, use standard AQE-optimized join
            return self._standardAsOfJoin(left, right)
        else:
            # Extreme skew detected, use separate processing
            logger.info(f"Processing {len(skewed_keys)} skewed keys separately")
            return self._skewSeparatedJoin(left, right, skewed_keys)

    def _standardAsOfJoin(self, left, right) -> Tuple[DataFrame, TSSchema]:
        """
        Perform standard as-of join with AQE optimization.

        This lets Spark's AQE handle any moderate skew automatically.
        """
        # Add hints for potential skew on series columns
        left_df = left.df
        right_df = right.df

        if left.series_ids:
            left_df = left_df.hint("skew", left.series_ids)
            right_df = right_df.hint("skew", right.series_ids)

        # Perform the as-of join using a range join approach
        # First, add a window to get the lead timestamp for each right row
        window_spec = Window.partitionBy(*left.series_ids).orderBy(right.ts_col)
        right_with_lead = right_df.withColumn(
            "lead_ts", sfn.lead(right.ts_col).over(window_spec)
        )

        # Join condition: series match AND left.ts between right.ts and right.lead_ts
        join_conditions = []

        # Add series column conditions if they exist
        for col in left.series_ids:
            join_conditions.append(sfn.col(f"l.{col}") == sfn.col(f"r.{col}"))

        # Add temporal join condition
        temporal_condition = (
            sfn.col(f"l.{left.ts_col}") >= sfn.col(f"r.{right.ts_col}")
        ) & (
            (sfn.col("r.lead_ts").isNull())
            | (sfn.col(f"l.{left.ts_col}") < sfn.col("r.lead_ts"))
        )

        if join_conditions:
            # Combine series and temporal conditions
            full_condition = join_conditions[0]
            for cond in join_conditions[1:]:
                full_condition = full_condition & cond
            full_condition = full_condition & temporal_condition
        else:
            # Only temporal condition for single series
            full_condition = temporal_condition

        # Perform the join
        joined = left_df.alias("l").join(
            right_with_lead.alias("r"), on=full_condition, how="left"
        )

        # Apply skipNulls filter if needed
        if self.skipNulls:
            joined = self._applySkipNulls(joined, right)

        # Select final columns with proper prefixes
        result_df = self._selectFinalColumns(joined, left, right)

        # Apply tolerance filter if specified
        if self.tolerance is not None:
            result_df = self._applyToleranceFilter(result_df, left, right)

        # Create result schema
        result_schema = TSSchema(ts_idx=left.ts_index, series_ids=left.series_ids)

        return result_df, result_schema

    def _skewSeparatedJoin(
        self, left, right, skewed_keys
    ) -> Tuple[DataFrame, TSSchema]:
        """
        Handle skewed and non-skewed keys separately, then union results.

        :param left: Left TSDF
        :param right: Right TSDF
        :param skewed_keys: List of skewed key values
        :return: Tuple of (joined DataFrame, TSSchema)
        """
        # Build filter conditions for skewed keys
        if len(left.series_ids) == 1:
            # Single series column
            skewed_filter = sfn.col(left.series_ids[0]).isin(
                [k[0] for k in skewed_keys]
            )
        else:
            # Multiple series columns - need complex filter
            skewed_filter = sfn.lit(False)
            for key_tuple in skewed_keys:
                key_condition = sfn.lit(True)
                for i, col in enumerate(left.series_ids):
                    key_condition = key_condition & (sfn.col(col) == key_tuple[i])
                skewed_filter = skewed_filter | key_condition

        # Split data
        left_skewed = TSDF(
            left.df.filter(skewed_filter), left.ts_col, left.series_ids, left.ts_index
        )
        left_normal = TSDF(
            left.df.filter(~skewed_filter), left.ts_col, left.series_ids, left.ts_index
        )
        right_skewed = TSDF(
            right.df.filter(skewed_filter),
            right.ts_col,
            right.series_ids,
            right.ts_index,
        )
        right_normal = TSDF(
            right.df.filter(~skewed_filter),
            right.ts_col,
            right.series_ids,
            right.ts_index,
        )

        # Process non-skewed with standard join
        normal_result, schema = self._standardAsOfJoin(left_normal, right_normal)

        # Process skewed with salting or broadcast (depending on size)
        if self.enable_salting:
            skewed_result, _ = self._saltedAsOfJoin(left_skewed, right_skewed)
        else:
            # Try broadcast if right side is small enough
            right_size = get_bytes_from_plan(right_skewed.df, self.spark)
            if right_size < 100 * 1024 * 1024:  # 100MB threshold for skewed
                logger.info("Using broadcast for skewed keys")
                skewed_result, _ = BroadcastAsOfJoiner(
                    self.spark, self.left_prefix, self.right_prefix
                )(left_skewed, right_skewed)
            else:
                # Fall back to standard join with AQE
                skewed_result, _ = self._standardAsOfJoin(left_skewed, right_skewed)

        # Union results
        result_df = normal_result.unionByName(skewed_result, allowMissingColumns=True)

        return result_df, schema

    def _saltedAsOfJoin(self, left, right) -> Tuple[DataFrame, TSSchema]:
        """
        Perform salted as-of join for extreme skew cases.

        :param left: Left TSDF
        :param right: Right TSDF
        :return: Tuple of (joined DataFrame, TSSchema)
        """
        logger.info(f"Using salted join with {self.salt_buckets} buckets")

        # Add deterministic salt to left based on hash of series keys
        if left.series_ids:
            salt_expr = (
                sfn.abs(sfn.hash(*[sfn.col(c) for c in left.series_ids]))
                % self.salt_buckets
            )
        else:
            salt_expr = sfn.abs(sfn.hash(sfn.col(left.ts_col))) % self.salt_buckets

        left_salted = left.df.withColumn("__salt", salt_expr)

        # Explode right to all salt buckets
        right_salted = right.df.crossJoin(
            self.spark.range(self.salt_buckets).select(sfn.col("id").alias("__salt"))
        )

        # Now perform the join including salt in the join key
        join_conditions = [
            sfn.col(f"l.{col}") == sfn.col(f"r.{col}") for col in left.series_ids
        ]
        join_conditions.append(sfn.col("l.__salt") == sfn.col("r.__salt"))

        # Add temporal condition using window approach
        window_spec = Window.partitionBy(
            *[f"r.{col}" for col in left.series_ids], "r.__salt"
        ).orderBy(f"r.{right.ts_col}")
        right_with_lead = right_salted.withColumn(
            "lead_ts", sfn.lead(f"r.{right.ts_col}").over(window_spec)
        )

        join_conditions.append(
            (sfn.col(f"l.{left.ts_col}") >= sfn.col(f"r.{right.ts_col}"))
            & (
                (sfn.col("lead_ts").isNull())
                | (sfn.col(f"l.{left.ts_col}") < sfn.col("lead_ts"))
            )
        )

        # Perform the salted join
        joined = left_salted.alias("l").join(
            right_with_lead.alias("r"), on=join_conditions, how="left"
        )

        # Remove salt columns and select final columns
        result_df = joined.drop("__salt", "lead_ts")
        result_df = self._selectFinalColumns(result_df, left, right)

        # Apply filters
        if self.skipNulls:
            result_df = self._applySkipNulls(result_df, right)
        if self.tolerance is not None:
            result_df = self._applyToleranceFilter(result_df, left, right)

        result_schema = TSSchema(ts_idx=left.ts_index, series_ids=left.series_ids)

        return result_df, result_schema

    def _selectFinalColumns(self, joined_df, left, right):
        """
        Select and rename columns for the final output.

        NOTE: The left and right TSDFs have already been prefixed by the parent
        class via _prefixOverlappingColumns. We just need to select the columns
        and remove the join aliases (l. and r.).

        :param joined_df: DataFrame with joined data (aliased as 'l' and 'r')
        :param left: Left TSDF (already prefixed by parent)
        :param right: Right TSDF (already prefixed by parent)
        :return: DataFrame with properly named columns
        """
        final_cols = []
        added_cols = set()

        # Add all left columns with their current names (already prefixed if needed)
        for col in left.columns:
            final_cols.append(sfn.col(f"l.{col}").alias(col))
            added_cols.add(col)

        # Add right columns (excluding series columns and already-added columns)
        for col in right.columns:
            if col not in right.series_ids and col not in added_cols:
                final_cols.append(sfn.col(f"r.{col}").alias(col))

        return joined_df.select(*final_cols)

    def _applySkipNulls(self, joined_df, right):
        """
        Apply skipNulls logic to filter out rows with null right values.

        When skipNulls is True, filters out rows where any right-side value
        column contains NULL (excluding timestamp and series columns).

        :param joined_df: DataFrame with joined data
        :param right: Right TSDF for column metadata
        :return: DataFrame with null rows filtered based on skipNulls setting
        """
        # Get right value columns (non-timestamp, non-series)
        right_value_cols = [
            col
            for col in right.columns
            if col != right.ts_col and col not in right.series_ids
        ]

        if not right_value_cols:
            return joined_df

        # Check if any right value column is NULL
        null_check = sfn.lit(False)
        for col in right_value_cols:
            if f"r.{col}" in joined_df.columns:
                null_check = null_check | sfn.col(f"r.{col}").isNull()

        # Keep rows where right timestamp is null (no match) or no nulls in values
        return joined_df.filter(sfn.col(f"r.{right.ts_col}").isNull() | ~null_check)

    def _applyToleranceFilter(self, result_df, left, right):
        """
        Apply tolerance filter to null out right columns outside tolerance window.

        Sets all right-side columns to NULL when the time difference between
        left and right timestamps exceeds the specified tolerance.

        NOTE: The left and right TSDFs have already been prefixed by the parent
        class, so we use their column names as-is.

        :param result_df: DataFrame with joined and selected columns
        :param left: Left TSDF (already prefixed by parent)
        :param right: Right TSDF (already prefixed by parent)
        :return: DataFrame with tolerance filter applied
        """
        # Use column names as they exist (already prefixed by parent if needed)
        left_ts_col = left.ts_col
        right_ts_col = right.ts_col

        # Calculate time difference
        time_diff = sfn.col(left_ts_col).cast("double") - sfn.col(right_ts_col).cast(
            "double"
        )
        outside_tolerance = time_diff > self.tolerance

        # Identify right columns by checking which ones came from right TSDF
        # These are the columns that are NOT in the left TSDF (excluding series columns)
        left_cols = set(left.columns)
        right_only_cols = [col for col in result_df.columns if col not in left_cols]

        # Null out right columns if outside tolerance
        for col in right_only_cols:
            result_df = result_df.withColumn(
                col, sfn.when(outside_tolerance, sfn.lit(None)).otherwise(sfn.col(col))
            )

        return result_df


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
    Extract the estimated size in bytes from the Spark execution plan.

    :param df: Input DataFrame
    :param spark: SparkSession
    :return: Size in bytes
    """
    plan = get_spark_plan(df, spark)

    # Extract sizeInBytes from plan
    search_result = re.search(
        r"sizeInBytes=([0-9.]+)\s*([A-Za-z]+)", plan, re.MULTILINE
    )
    if search_result is None:
        logger.warning("Unable to obtain sizeInBytes from Spark plan")
        return float("inf")  # Return large number to avoid broadcast

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
    left_tsdf,
    right_tsdf,
    spark: SparkSession,
    left_prefix: Optional[str] = None,
    right_prefix: str = "right",
    tsPartitionVal: Optional[int] = None,
    fraction: float = 0.5,
    skipNulls: bool = True,
    tolerance: Optional[int] = None,
) -> AsOfJoiner:
    """
    Automatically choose the optimal as-of join strategy based on data characteristics.

    Decision tree:
    1. If tsPartitionVal is set: SkewAsOfJoiner (for handling skewed data)
    2. If either DataFrame < 30MB: BroadcastAsOfJoiner (for small data)
    3. Default: UnionSortFilterAsOfJoiner (for general cases)

    :param left_tsdf: Left TSDF
    :param right_tsdf: Right TSDF
    :param spark: SparkSession instance
    :param left_prefix: Prefix for left columns
    :param right_prefix: Prefix for right columns
    :param tsPartitionVal: Time partition value for skew handling
    :param fraction: Overlap fraction for partitions (deprecated, kept for compatibility)
    :param skipNulls: Whether to skip nulls
    :param tolerance: Tolerance window in seconds
    :return: Appropriate AsOfJoiner instance
    """
    try:
        # Use the skew join if the partition value is passed in (highest priority)
        if tsPartitionVal is not None:
            logger.info(f"Using SkewAsOfJoiner with partition value {tsPartitionVal}")
            return SkewAsOfJoiner(
                spark=spark,
                left_prefix=left_prefix or "left",
                right_prefix=right_prefix,
                skipNulls=skipNulls,
                tolerance=tolerance,
                tsPartitionVal=tsPartitionVal,
            )

        # Test if the broadcast join will be efficient (check sizes automatically)
        try:
            left_bytes = get_bytes_from_plan(left_tsdf.df, spark)
            right_bytes = get_bytes_from_plan(right_tsdf.df, spark)

            # Use broadcast if either DataFrame is small enough
            if (left_bytes < _DEFAULT_BROADCAST_BYTES_THRESHOLD) or (
                right_bytes < _DEFAULT_BROADCAST_BYTES_THRESHOLD
            ):
                logger.info(
                    f"Using BroadcastAsOfJoiner "
                    f"(left: {left_bytes/1024/1024:.2f}MB, "
                    f"right: {right_bytes/1024/1024:.2f}MB)"
                )
                return BroadcastAsOfJoiner(spark, left_prefix or "left", right_prefix)
        except Exception as e:
            logger.debug(
                f"Could not estimate DataFrame size: {e}. Using default strategy."
            )
            # Fall through to default strategy

        # Default to union-sort-filter join
        logger.info("Using default UnionSortFilterAsOfJoiner")
        return UnionSortFilterAsOfJoiner(
            left_prefix or "left", right_prefix, skipNulls, tolerance
        )

    except Exception as e:
        logger.error(f"Strategy selection failed: {e}")
        # Always fall back to safe default
        return UnionSortFilterAsOfJoiner(
            left_prefix or "left", right_prefix, skipNulls, tolerance
        )


def _detectSignificantSkew(left_tsdf, right_tsdf, threshold: float = 0.3) -> bool:
    """
    Quick heuristic to detect if data has significant skew.

    :param left_tsdf: Left TSDF
    :param right_tsdf: Right TSDF
    :param threshold: Coefficient of variation threshold for skew detection
    :return: True if significant skew is detected
    """
    try:
        if not left_tsdf.series_ids:
            return False  # No series to be skewed on

        # Quick check: sample and look at partition sizes
        # This is a heuristic - not perfect but fast
        sample_size = min(10000, left_tsdf.df.count())
        if sample_size < 100:
            return False  # Too small to have meaningful skew

        # Group by series and count
        series_counts = (
            left_tsdf.df.sample(fraction=min(1.0, sample_size / left_tsdf.df.count()))
            .groupBy(*left_tsdf.series_ids)
            .count()
            .select(sfn.stddev("count").alias("std"), sfn.avg("count").alias("avg"))
            .collect()[0]
        )

        if series_counts["avg"] and series_counts["std"]:
            # Coefficient of variation > threshold indicates skew
            cv = series_counts["std"] / series_counts["avg"]
            if cv > threshold:
                logger.info(f"Detected data skew (CV={cv:.2f})")
                return True
    except Exception as e:
        logger.debug(f"Could not detect skew: {e}")

    return False
