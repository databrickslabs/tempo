from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import pyspark.sql.functions as sfn
import tempo.tsdf as t_tsdf
from pyspark.sql import SparkSession
from pyspark.sql.utils import ParseException

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

    view_df = df.withColumn("event_dt", sfn.to_date(sfn.col(ts_col))).withColumn(
        "event_time",
        sfn.translate(sfn.split(sfn.col(ts_col).cast("string"), " ")[1], ":", "").cast(
            "double"
        ),
    )
    view_cols = deque(view_df.columns)
    view_cols.rotate(1)
    view_df = view_df.select(*list(view_cols))

    view_df.write.mode("overwrite").partitionBy("event_dt").format("delta").saveAsTable(
        tabName
    )

    if optimizationCols:
        try:
            spark.sql(
                "optimize {} zorder by {}".format(
                    tabName,
                    "(" + ",".join(partitionCols + optimizationCols + [ts_col]) + ")",
                )
            )
        except ParseException as e:
            logger.error(
                "Delta optimizations attempted, but was not successful.\nError: {}".format(
                    e
                )
            )
