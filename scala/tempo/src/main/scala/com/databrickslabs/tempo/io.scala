package com.databrickslabs.tempo

import com.databrickslabs.tempo.utils.SparkSessionWrapper
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession


/**
The following methods provide support for writing the TSDF object to Delta Lake based on the partition columns specified along with the time as ZORDER columns (with Hilbert space-filling curve optimization)
 */
object TSDFWriters extends SparkSessionWrapper {
  /**
    * param: tsdf: input TSDF object to write
    * param: tabName Delta output table name
    * param: tabPath is the external table path which can be used to save a table to an arbitrary blob location
    * param: optimizationCols list of columns to optimize on (time) in addition to the partition key columns from the original TSDF definition
    */
  def write(tsdf: TSDF, tabName: String, tabPath : String = "", optimizationCols: Option[Seq[String]] = None): Unit = {

    // hilbert curves more evenly distribute performance for querying multiple columns for Delta tables
    spark.conf.set("spark.databricks.io.skipping.mdc.curve", "hilbert")

    val df = tsdf.df
    val ts_col = tsdf.tsColumn.name
    val partitionCols = tsdf.partitionCols

    var optCols = partitionCols.map(_.name)

    optimizationCols match {
      case Some(oc) => {
        optCols = optCols ++ oc ++ Seq("event_time")
      }
      case None => {
         optCols = optCols ++ Seq("event_time")
      }
    }

    val useDeltaOpt = !(sys.env.get("DATABRICKS_RUNTIME_VERSION") == None)

    val view_df = df.withColumn("event_dt", to_date(col(ts_col))).withColumn("event_time", translate(split(col(ts_col).cast("string"), " ")(1), ":", " ").cast("double"))

    tabPath == "" match {
      case false => view_df.write.option("path", tabPath).mode("overwrite").partitionBy("event_dt").format("delta").saveAsTable(tabName)
      case true =>  view_df.write.mode("overwrite").partitionBy("event_dt").format("delta").saveAsTable(tabName)

    }

    if (useDeltaOpt) {
      val optColsStr = optCols.mkString(",")
      spark.sql("optimize $tabName zorder by $optColsStr")
    } else {
      print("Delta optimizations attempted on a non-Databricks platform. Switch to use Databricks Runtime to get optimization advantages.")
    }
  }
}