package com.databrickslabs.tempo
import org.apache.spark.sql.expressions.{Window, WindowSpec}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Column, DataFrame}
import scala.math.pow

object EMA {
  // Constructs an approximate EMA in the fashion of:
  // EMA = e * lag(col,0) + e * (1 - e) * lag(col, 1) + e * (1 - e)^2 * lag(col, 2) etc, up until windoww
  def emaExec(tsdf: TSDF, colName: String, window: Int, exp_factor: Double): TSDF = {

    val emaColName = "ema_" + colName
    val df = tsdf.df.withColumn(emaColName, lit(0))

    //TODO: make use of pre-defined windowing methods.
    val window_spec = Window
      .partitionBy(tsdf.partitionCols.map(x => col(x.name)):_*)
      .orderBy(tsdf.tsColumn.name)

    val tempLagCol = "temp_lag_col"

    // TODO: Check whether tempLagCol may need a different name each time
    def emaIncrement(df: DataFrame, k: Int): DataFrame = {
      val weight = exp_factor * pow(1 - exp_factor, k)

      val updateDf = df
        .withColumn(tempLagCol, lit(weight) * lag(colName, k).over(window_spec))
        .withColumn(emaColName, col(emaColName) + coalesce(col(tempLagCol), lit(0)))
        .drop(tempLagCol)

      updateDf
    }

    val emaDF = Range(0,window+1).foldLeft(df)((df:DataFrame, k: Int) => emaIncrement(df,k))

    TSDF(emaDF, tsdf.tsColumn.name, tsdf.partitionCols.map(_.name):_*)
  }
}
