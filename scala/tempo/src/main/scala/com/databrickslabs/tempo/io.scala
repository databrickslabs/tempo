package com.databrickslabs.tempo

import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession


object TSDFWriters {
  /**
    * param: tsdf: input TSDF object to write
    * param: tabName Delta output table name
    * param: optimizationCols list of columns to optimize on (time)
    */
  def write(tsdf: TSDF, spark: SparkSession, tabName: String, optimizationCols: Option[Seq[String]] = None): Unit = {

    // hilbert curves more evenly distribute performance for querying multiple columns for Delta tables
    spark.conf.set("spark.databricks.io.skipping.mdc.curve", "hilbert")

    val df = tsdf.df
    val ts_col = tsdf.tsColumn.name
    val partitionCols = tsdf.partitionCols

    var optCols = Seq("")

    optimizationCols match {
      case Some(oc) => {
        optCols = oc ++ Seq("event_time")
      }
      case None => {
         optCols = Seq("event_time")
      }
    }

    var useDeltaOpt = true

    try {
      val ddf = spark.range(10).write.mode("overwrite").partitionBy("event_dt").format("delta").saveAsTable("ddft")
      spark.sql("optimize ddft")
    } catch {
      case foo: Exception => {
        println("Running in local mode")
        useDeltaOpt = false
      }
    }

    val view_df = df.withColumn("event_dt", to_date(col(ts_col))).withColumn("event_time", translate(split(col(ts_col).cast("string"), " ")(1), ":", " ").cast("double"))

    view_df.write.mode("overwrite").partitionBy("event_dt").format("delta").saveAsTable(tabName)

    if (useDeltaOpt) {
      val optColsStr = optCols.mkString(",")
      spark.sql("optimize $tabName zorder by $optColsStr")
    } else {
      print("Delta optimizations attempted on a non-Databricks platform. Switch to use Databricks Runtime to get optimization advantages.")
    }
  }
}