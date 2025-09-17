"""
Patch for integrating strategy pattern into TSDF.asofJoin method.

This file shows the minimal changes needed to integrate the strategy pattern
while maintaining backward compatibility.
"""

# Add this import at the top of tsdf.py (around line 27 with other imports)
IMPORT_TO_ADD = """
from tempo.joins.strategies import (
    choose_as_of_join_strategy,
    BroadcastAsOfJoiner,
    UnionSortFilterAsOfJoiner,
    SkewAsOfJoiner,
)
"""

# Replace the asofJoin method (lines 931-1133) with this version:
def asofJoin(
    self,
    right_tsdf: TSDF,
    left_prefix: Optional[str] = None,
    right_prefix: str = "right",
    tsPartitionVal: Optional[int] = None,
    fraction: float = 0.5,
    skipNulls: bool = True,
    sql_join_opt: bool = False,
    suppress_null_warning: bool = False,
    tolerance: Optional[int] = None,
    use_strategy_pattern: bool = False,  # Feature flag for gradual rollout
) -> TSDF:
    """
    Performs an as-of join between two time-series.

    With use_strategy_pattern=True, uses the new modular strategy pattern
    for improved maintainability and performance optimization.

    Parameters remain the same for backward compatibility.
    """

    # Feature flag allows gradual rollout and easy rollback
    if use_strategy_pattern:
        # Log warning for skew join if applicable
        if tsPartitionVal is not None and not suppress_null_warning:
            logger.warning(
                "You are using the skew version of the AS OF join. This may result in null values if there are any "
                "values outside of the maximum lookback. For maximum efficiency, choose smaller values of maximum "
                "lookback, trading off performance and potential blank AS OF values for sparse keys"
            )

        # Use the new strategy pattern
        try:
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

            # Log which strategy was selected for debugging
            logger.info(f"Using as-of join strategy: {joiner.__class__.__name__}")

            return joiner(self, right_tsdf)

        except Exception as e:
            # If strategy pattern fails, log and fall back to original
            logger.error(f"Strategy pattern failed: {e}. Falling back to original implementation.")
            # Fall through to original implementation

    # Original implementation starts here (current lines 961-1133)
    # [Keep all the existing code as-is for backward compatibility]

    # first block of logic checks whether a standard range join will suffice
    left_df = self.df
    right_df = right_tsdf.df

    # test if the broadcast join will be efficient
    if sql_join_opt:
        spark = SparkSession.builder.getOrCreate()
        left_bytes = self.__getBytesFromPlan(left_df, spark)
        right_bytes = self.__getBytesFromPlan(right_df, spark)

        # choose 30MB as the cutoff for the broadcast
        bytes_threshold = 30 * 1024 * 1024
        if (left_bytes < bytes_threshold) or (right_bytes < bytes_threshold):
            spark.conf.set("spark.databricks.optimizer.rangeJoin.binSize", 60)
            partition_cols = right_tsdf.series_ids
            left_cols = list(set(left_df.columns) - set(self.series_ids))
            right_cols = list(set(right_df.columns) - set(right_tsdf.series_ids))

            left_prefix = left_prefix + "_" if left_prefix else ""
            right_prefix = right_prefix + "_" if right_prefix else ""

            w = Window.partitionBy(*partition_cols).orderBy(
                right_prefix + right_tsdf.ts_col
            )

            new_left_ts_col = left_prefix + self.ts_col
            series_cols = [sfn.col(c) for c in self.series_ids]
            new_left_cols = [
                sfn.col(c).alias(left_prefix + c) for c in left_cols
            ] + series_cols
            new_right_cols = [
                sfn.col(c).alias(right_prefix + c) for c in right_cols
            ] + series_cols
            quotes_df_w_lag = right_df.select(*new_right_cols).withColumn(
                "lead_" + right_tsdf.ts_col,
                sfn.lead(right_prefix + right_tsdf.ts_col).over(w),
            )
            left_df = left_df.select(*new_left_cols)
            res = (
                left_df.join(quotes_df_w_lag, partition_cols)
                .where(
                    left_df[new_left_ts_col].between(
                        sfn.col(right_prefix + right_tsdf.ts_col),
                        sfn.coalesce(
                            sfn.col("lead_" + right_tsdf.ts_col),
                            sfn.lit("2099-01-01").cast("timestamp"),
                        ),
                    )
                )
                .drop("lead_" + right_tsdf.ts_col)
            )
            return TSDF(res, ts_col=new_left_ts_col, series_ids=self.series_ids)

    # [Rest of original implementation continues...]
    # [Lines 1015-1133 remain unchanged]