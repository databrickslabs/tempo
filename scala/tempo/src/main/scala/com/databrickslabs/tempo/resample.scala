package com.databrickslabs.tempo

import org.apache.spark.sql.functions._
import TSDF._

/**
  * The following object contains methods for resampling (up to millions of time series in parallel).
  */
object resample {

// define global frequency options
val SEC = "sec"
val MIN = "min"
val HR = "hr"

// define global aggregate function options for downsampling
val CLOSEST_LEAD = "closest_lead"
val MIN_LEAD = "min_lead"
val MAX_LEAD = "max_lead"
val MEAN_LEAD = "mean_lead"

val allowableFreqs = List(SEC, MIN, HR)

  /**
    *
    * @param tsdf - TSDF object as input
    * @param freq - frequency at which to upsample
    * @return - return a TSDF with a new aggregate key (called agg_key)
    */
  private[tempo] def __appendAggKey(tsdf : TSDF, freq : String) : TSDF = {
    var df = tsdf.df
    checkAllowableFreq(freq)

    // compute timestamp columns
    val sec_col = second(col(tsdf.tsColumn.name))
    val min_col = minute(col(tsdf.tsColumn.name))
    val hour_col = hour(col(tsdf.tsColumn.name))
    var agg_key = col(tsdf.tsColumn.name)

    if (freq == SEC) {
      agg_key = concat(col(tsdf.tsColumn.name).cast("date"), lit(" "), lpad(hour_col, 2, "0"), lit(":"), lpad(min_col, 2, "0"), lit(":"), lpad(sec_col, 2, "0")).cast("timestamp")
    } else if (freq == MIN) {
        agg_key = concat(col(tsdf.tsColumn.name).cast("date"), lit(" "), lpad(hour_col, 2, "0"), lit(":"), lpad(min_col, 2, "0"), lit(":"), lit("00")).cast("timestamp")
    } else if (freq == HR) {
        agg_key = concat(col(tsdf.tsColumn.name).cast("date"), lit(" "), lpad(hour_col, 2, "0"), lit(":"), lit("00"), lit(":"), lit("00")).cast("timestamp")
    }

    df = df.withColumn("agg_key", agg_key)
    return TSDF(df, tsColumnName= tsdf.tsColumn.name, partitionColumnNames = tsdf.partitionCols.map(x => x.name): _*)
  }

  /**
    *
    * @param tsdf - input TSDF object
    * @param freq - aggregate function
    * @param func - aggregate function
    * @param metricCols - columns used for aggregates
    * return - TSDF object with newly aggregated timestamp as ts_col with aggregated values
    */
  def rs_agg(tsdf : TSDF, freq : String, func : String, metricCols : Option[List[String]] = None) : TSDF =  {

       validateFuncExists(func)

       var tsdf_w_aggkey = __appendAggKey(tsdf, freq)
       val df = tsdf_w_aggkey.df
       var adjustedMetricCols = List("")

       var groupingCols = tsdf.partitionCols.map(x => x.name) :+ "agg_key"

       adjustedMetricCols = metricCols match {
         case Some(s) => metricCols.get
         case None => (df.columns.toSeq diff (groupingCols :+ tsdf.tsColumn.name)).toList
       }


       var metricCol = col("")
       var groupedRes = df.withColumn("struct_cols", struct( (Seq(tsdf.tsColumn.name) ++ adjustedMetricCols.toSeq).map(col): _*)).groupBy(groupingCols map col: _*)
       var res = groupedRes.agg(min("struct_cols").alias("closest_data")).select("*", "closest_data.*").drop("closest_data")

       if (func == CLOSEST_LEAD)  {
         groupedRes = df.withColumn("struct_cols", struct( (Seq(tsdf.tsColumn.name) ++ adjustedMetricCols.toSeq).map(col): _*)).groupBy(groupingCols map col: _*)
         res = groupedRes.agg(min("struct_cols").alias("closest_data")).select("*", "closest_data.*").drop("closest_data")
       } else if (func == MEAN_LEAD) {
         val exprs = adjustedMetricCols.map(k => avg(df(s"$k")).alias("avg_" + s"$k"))
         res = df.groupBy(groupingCols map col: _*).agg(exprs.head, exprs.tail:_*)
       } else if (func == MIN_LEAD) {
         val exprs = adjustedMetricCols.map(k => min(df(s"$k")).alias("avg_" + s"$k"))
         res = df.groupBy(groupingCols map col: _*).agg(exprs.head, exprs.tail:_*)
       } else if (func == MAX_LEAD) {
         val exprs = adjustedMetricCols.map(k => max(df(s"$k")).alias("avg_" + s"$k"))
         res = df.groupBy(groupingCols map col: _*).agg(exprs.head, exprs.tail:_*)
       }

       res = res.drop(tsdf.tsColumn.name).withColumnRenamed("agg_key", tsdf.tsColumn.name)
       return(TSDF(res, tsColumnName = tsdf.tsColumn.name, partitionColumnNames = tsdf.partitionCols.map(x => x.name): _*))
  }


       def checkAllowableFreq(freq : String) : Unit = {
         if (!allowableFreqs.contains(freq) ) {
         throw new IllegalArgumentException ("Allowable grouping frequencies are sec (second), min (minute), hr (hour)")
       } else {}
       }


       def validateFuncExists(func : String): Unit = {
         if (func == None) {
           throw new IllegalArgumentException("Aggregate function missing. Provide one of the allowable functions: " + List(CLOSEST_LEAD, MIN_LEAD, MAX_LEAD, MEAN_LEAD).mkString(","))

         }
       }
  }