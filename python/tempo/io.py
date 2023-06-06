from __future__ import annotations

import logging
import os
from collections import deque
from typing import Optional

import pyspark.sql.functions as sql_fn
from pyspark.sql import SparkSession
from pyspark.sql.utils import ParseException

import tempo.tsdf as t_tsdf

logger = logging.getLogger(__name__)


def write(
    tsdf: t_tsdf.TSDF,
    spark: SparkSession,
    tabName: str,
    optimizationCols: Optional[list[str]] = None,
) -> None:
    """
    param: tsdf: input TSDF object to write
    param: tabName Delta output table name
    param: optimizationCols list of columns to optimize on (time)
    """
    # hilbert curves more evenly distribute performance for querying multiple columns for Delta tables
    spark.conf.set("spark.databricks.io.skipping.mdc.curve", "hilbert")

    df = tsdf.df
    ts_col = tsdf.ts_col
    partitionCols = tsdf.partitionCols
    if optimizationCols:
        optimizationCols = optimizationCols + ["event_time"]
    else:
        optimizationCols = ["event_time"]

    useDeltaOpt = os.getenv("DATABRICKS_RUNTIME_VERSION") is not None

    view_df = df.withColumn("event_dt", sql_fn.to_date(sql_fn.col(ts_col))).withColumn(
        "event_time",
        sql_fn.translate(
            sql_fn.split(sql_fn.col(ts_col).cast("string"), " ")[1], ":", ""
        ).cast("double"),
    )
    view_cols = deque(view_df.columns)
    view_cols.rotate(1)
    view_df = view_df.select(*list(view_cols))

    view_df.write.mode("overwrite").partitionBy("event_dt").format("delta").saveAsTable(
        tabName
    )

    if useDeltaOpt:
        try:
            spark.sql(
                "optimize {} zorder by {}".format(
                    tabName, "(" + ",".join(partitionCols + optimizationCols) + ")"
                )
            )
        except ParseException as e:
            logger.error(
                "Delta optimizations attempted, but was not successful.\nError: {}".format(
                    e
                )
            )
    else:
        logger.warning(
            "Delta optimizations attempted on a non-Databricks platform. "
            "Switch to use Databricks Runtime to get optimization advantages."
        )
