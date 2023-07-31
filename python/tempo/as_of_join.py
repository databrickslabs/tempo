from abc import ABC, abstractmethod
from typing import Optional
from functools import reduce

from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as sfn

import tempo.tsdf as t_tsdf

# As-of join types

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

    def _overlappingColumns(self, left: t_tsdf.TSDF, right: t_tsdf.TSDF) -> set:
        """
        Returns the overlapping columns in the left and right TSDFs
        """
        # find the columns to prefix
        prefixable_left_cols = set(left.columns) - set(left.series_ids)
        prefixable_right_cols = set(right.columns) - set(right.series_ids)
        return prefixable_left_cols.intersection(prefixable_right_cols)

    def _prefixColumns(self,
                       tsdf: t_tsdf.TSDF,
                       prefixable_cols: set[str],
                       prefix: str) -> t_tsdf.TSDF:
        """
        Prefixes the columns in the TSDF
        """
        if prefix:
            tsdf = reduce(
                lambda tsdf, c:
                    tsdf.withColumnRenamed(c, "_".join([prefix, c])),
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
        prefixable_cols = self._overlappingColumns(left, right)

        # prefix columns (if we have a prefix to apply)
        left_prefixed = self._prefixColumns(left, prefixable_cols, self.left_prefix)
        right_prefixed = self._prefixColumns(right, prefixable_cols, self.right_prefix)

        return left_prefixed, right_prefixed

    def _checkAreJoinable(self, left: t_tsdf.TSDF, right: t_tsdf.TSDF) -> t_tsdf.TSDF:
        """
        Checks if the left and right TSDFs are joinable
        """
        pass

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
                 right_prefix: str = "right"):
        super().__init__(left_prefix, right_prefix)
        self.spark = spark

    def _join(self, left: t_tsdf.TSDF, right: t_tsdf.TSDF) -> t_tsdf.TSDF:
        self.spark.conf.set("spark.databricks.optimizer.rangeJoin.binSize", 60)
        partition_cols = right.series_ids
        left_cols = list(set(left.df.columns) - set(left.series_ids))
        right_cols = list(set(right.df.columns) - set(right.series_ids))

        left_prefix = (
            ""
            if ((self.left_prefix is None) | (self.left_prefix == ""))
            else self.left_prefix + "_"
        )
        right_prefix = (
            ""
            if ((self.right_prefix is None) | (self.right_prefix == ""))
            else self.right_prefix + "_"
        )

        w = right.baseWindow()

        new_left_ts_col = left_prefix + self.ts_col
        series_cols = [sfn.col(c) for c in self.series_ids]
        new_left_cols = [
                            sfn.col(c).alias(left_prefix + c) for c in left_cols
                        ] + series_cols
        new_right_cols = [
                             sfn.col(c).alias(right_prefix + c) for c in right_cols
                         ] + series_cols
        quotes_df_w_lag = right.select(*new_right_cols).withColumn(
            "lead_" + right.ts_col,
            sfn.lead(right_prefix + right.ts_col).over(w),
        )
        left_df = left.select(*new_left_cols)
        res = (
            left_df._join(quotes_df_w_lag, partition_cols)
            .where(
                left_df[new_left_ts_col].between(
                    sfn.col(right_prefix + right.ts_col),
                    sfn.coalesce(
                        sfn.col("lead_" + right.ts_col),
                        sfn.lit("2099-01-01").cast("timestamp"),
                    ),
                )
            )
            .drop("lead_" + right.ts_col)
        )
        return res


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
