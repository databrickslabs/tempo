package com.databrickslabs.tempo

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

import scala.math.pow

// EMA = exponential moving average. The following implementation is an approximation based on built-in Spark methods
object EMA {
  // Constructs an approximate EMA in the fashion of:
  // EMA = e * lag(col,0) + e * (1 - e) * lag(col, 1) + e * (1 - e)^2 * lag(col, 2) etc, up until window
  def emaExec(tsdf: TSDF, colName: String, window: Int, exp_factor: Double): TSDF =
  {
    val emaColName = "ema_" + colName
    val df = tsdf.df.withColumn(emaColName, lit(0))
    val window_spec = tsdf.baseWindow()
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

    TSDF(emaDF, tsdf.tsColumn.name, tsdf.partitionCols.map(_.name))
  }
}
