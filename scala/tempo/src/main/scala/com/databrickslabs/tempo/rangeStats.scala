package com.databrickslabs.tempo

import org.apache.spark.sql.expressions.{Window, WindowSpec}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructField}
import org.apache.spark.sql.{Column, DataFrame}

object rangeStats {


  val SUMMARY_FUNCS: List[(String, String => Column)] = List(
    ("mean_", {c: String => mean(c)}),
    ("count_",{c: String => count(c)}),
    ("sum_" , {c: String => sum(c)}),
    ("min_" , {c: String => min(c)}),
    ("max_" ,{c: String => max(c)}),
    ("stddev_" ,{c: String => stddev(c)}))

  // For functions that are derived from SUMMARY_FUNCS (and do not need a window function)
  val DERIVE_FUNCS: List[(String, String => Column)] = List(
    ("zscore_", {c: String => (col(c) - col("mean_" + c)) / col("stddev_" + c)}))

  // helper function to get aggregation stats for all agg functions in SUMMARY_FUNCS for a given metric column
  def doSummarize(df: DataFrame, colName: String, window_spec: WindowSpec): DataFrame  = {
    SUMMARY_FUNCS.foldLeft(df)((df, f) => df.withColumn(f._1 + colName, f._2(colName).over(window_spec)))
  }

  def doDerive(df: DataFrame, colName: String): DataFrame = {
    DERIVE_FUNCS.foldLeft(df)((df,f) => df.withColumn(f._1 + colName, f._2(colName)))
  }

  def rangeStatsExec(
    tsdf: TSDF,
    summarise_type: String = "range",
    colsToSummarise: Seq[String],
    rangeBackWindowSecs: Int =1000
  ): TSDF = {

    val innerColsToSummarise: Seq[String] = colsToSummarise match {
      case Seq() => tsdf.measureColumns.map(_.name)
      case _ => colsToSummarise
    }

    //TODO: make use of pre-defined windowing methods.
    val window_spec = Window
      .partitionBy(tsdf.partitionCols.map(x => col(x.name)):_*)
      .orderBy(tsdf.tsColumn.name)
    // mean, count, min, max, sum, stddev . Create higher order function that

    val summaryDF = innerColsToSummarise
      .foldLeft(tsdf.df)((df, colName) => {
        val summaryDf = doSummarize(df, colName, window_spec)
        val derivedDf = doDerive(summaryDf, colName)
        derivedDf
      })

    TSDF(summaryDF, tsdf.tsColumn.name, tsdf.partitionCols.map(_.name):_*)
  }


}
