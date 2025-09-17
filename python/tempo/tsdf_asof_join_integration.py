"""
Integration of strategy pattern into TSDF.asofJoin method.

This file contains the modified asofJoin method that should replace the existing
implementation in tsdf.py. It maintains backward compatibility while using the
new strategy pattern.
"""

from typing import Optional
from tempo.joins.strategies import (
    choose_as_of_join_strategy,
    BroadcastAsOfJoiner,
    UnionSortFilterAsOfJoiner,
    SkewAsOfJoiner,
)
from pyspark.sql import SparkSession

def asofJoin(
    self,
    right_tsdf: 'TSDF',
    left_prefix: Optional[str] = None,
    right_prefix: str = "right",
    tsPartitionVal: Optional[int] = None,
    fraction: float = 0.5,
    skipNulls: bool = True,
    sql_join_opt: bool = False,
    suppress_null_warning: bool = False,
    tolerance: Optional[int] = None,
    strategy: Optional[str] = None,  # NEW: manual strategy override
) -> 'TSDF':
    """
    Performs an as-of join between two time-series using the strategy pattern.

    If a tsPartitionVal is specified, it will use the SkewAsOfJoiner strategy
    to handle skewed data distributions through time-based partitioning.

    NOTE: partition cols have to be the same for both Dataframes.

    Parameters:
    :param right_tsdf - right-hand data frame containing columns to merge in
    :param left_prefix - optional prefix for base data frame
    :param right_prefix - optional prefix for right-hand data frame
    :param tsPartitionVal - value to break up each partition into time brackets
    :param fraction - overlap fraction for time partitions
    :param skipNulls - whether to skip nulls when joining in values
    :param sql_join_opt - if set to True, will use broadcast join if efficient
    :param suppress_null_warning - suppress warnings about null values in partitioned joins
    :param tolerance - only join values within this tolerance range (inclusive), expressed in seconds
    :param strategy - Optional manual strategy selection ("broadcast", "union_sort_filter", "skew", "auto")

    Returns:
    :return - A new TSDF with the as-of join result
    """

    # Log warning for skew join if applicable
    if tsPartitionVal is not None and not suppress_null_warning:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            "You are using the skew version of the AS OF join. This may result in null values if there are any "
            "values outside of the maximum lookback. For maximum efficiency, choose smaller values of maximum "
            "lookback, trading off performance and potential blank AS OF values for sparse keys"
        )

    # Manual strategy override
    if strategy:
        if strategy == "broadcast":
            spark = SparkSession.builder.getOrCreate()
            joiner = BroadcastAsOfJoiner(spark, left_prefix or "", right_prefix)
        elif strategy == "union_sort_filter":
            joiner = UnionSortFilterAsOfJoiner(left_prefix or "", right_prefix, skipNulls, tolerance)
        elif strategy == "skew":
            joiner = SkewAsOfJoiner(
                left_prefix or "",
                right_prefix,
                skipNulls,
                tolerance,
                tsPartitionVal,
                fraction
            )
        else:  # "auto" or invalid
            joiner = choose_as_of_join_strategy(
                self,
                right_tsdf,
                left_prefix,
                right_prefix,
                tsPartitionVal,
                fraction,
                skipNulls,
                sql_join_opt,
                tolerance
            )
    else:
        # Automatic strategy selection based on parameters
        joiner = choose_as_of_join_strategy(
            self,
            right_tsdf,
            left_prefix,
            right_prefix,
            tsPartitionVal,
            fraction,
            skipNulls,
            sql_join_opt,
            tolerance
        )

    # Execute the join using the selected strategy
    return joiner(self, right_tsdf)


# Alternative implementation that maintains more of the original structure
# This can be used if we want to keep more of the existing code intact

def asofJoin_minimal_changes(
    self,
    right_tsdf: 'TSDF',
    left_prefix: Optional[str] = None,
    right_prefix: str = "right",
    tsPartitionVal: Optional[int] = None,
    fraction: float = 0.5,
    skipNulls: bool = True,
    sql_join_opt: bool = False,
    suppress_null_warning: bool = False,
    tolerance: Optional[int] = None,
    use_strategy_pattern: bool = True,  # Feature flag for gradual rollout
) -> 'TSDF':
    """
    Alternative implementation with feature flag for gradual rollout.

    When use_strategy_pattern is True, uses the new strategy pattern.
    When False, falls back to the original implementation.
    """

    if use_strategy_pattern:
        # Use the new strategy pattern
        joiner = choose_as_of_join_strategy(
            self,
            right_tsdf,
            left_prefix,
            right_prefix,
            tsPartitionVal,
            fraction,
            skipNulls,
            sql_join_opt,
            tolerance
        )
        return joiner(self, right_tsdf)
    else:
        # Call the original implementation (would be the existing code)
        # This would be the current implementation in tsdf.py
        return self._original_asofJoin(
            right_tsdf,
            left_prefix,
            right_prefix,
            tsPartitionVal,
            fraction,
            skipNulls,
            sql_join_opt,
            suppress_null_warning,
            tolerance
        )