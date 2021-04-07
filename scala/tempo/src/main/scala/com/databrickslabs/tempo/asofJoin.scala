package com.databrickslabs.tempo

import org.apache.spark.sql.expressions.{Window, WindowSpec}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructField

/*
For many large-scale time series in various industries, the timestamps occur at irregular times. The AS OF join functionality here allows the user to merge time series A and time series B (replicated to millions of time series pairs A & B) by specifying only a partition column and timestamp column. The returned result will have card(time series A) with additional columns from time series B, namely the last non-null observation from time series B as of the time of time series A for all points in time series A.
 */
object asofJoin {

  def checkEqualPartitionCols(leftTSDF: TSDF, rightTSDF: TSDF): Boolean = {
    leftTSDF.partitionCols.zip(rightTSDF.partitionCols).forall(p => p._1 == p._2)
  }

  // add prefix to all specified columns - useful since repeated AS OF joins are likely chained together
  def addPrefixToColumns(tsdf: TSDF, col_list: Seq[StructField], prefix: String): TSDF = {


    // Prefix all column
    val prefixedDF = col_list.foldLeft(tsdf.df)((df, colStruct) => df
      .withColumnRenamed(colStruct.name, prefix + colStruct.name))

    // update ts_colName if it is in col_list
    val ts_colName = if(col_list.contains(tsdf.tsColumn)) prefix + tsdf.tsColumn.name else tsdf.tsColumn.name

    // update any partition column names that are in col_list
    val partitionColstrings = tsdf.partitionCols
      .foldLeft(Seq[String]())((newPartCols, PartCol) =>
        newPartCols :+ (if (col_list.contains(PartCol)) prefix + PartCol.name else PartCol.name))

    TSDF(prefixedDF,ts_colName, partitionColstrings:_*)
  }

  // Add columns from other DF ahead of combining them through Union
  // TODO: implement unit test
  def addColumnsFromOtherDF(tsdf: TSDF, other_cols: Seq[StructField]): TSDF = {

    val newDF = other_cols.foldLeft(tsdf.df)((df, colStruct) => df
      .withColumn(colStruct.name, lit(null)))

    // TODO: fix partition column names, it's not nice this way
    TSDF(newDF, tsdf.tsColumn.name, tsdf.partitionCols.map(_.name):_*)
  }

  // union the time series A and time series B data frames
  def combineTSDF(leftTSDF: TSDF, rightTSDF: TSDF): TSDF = {

    val combinedTsCol = "combined_ts"

    val left_rec_ind = leftTSDF.df.columns.filter(x => x.endsWith("rec_ind"))(0)
    val right_rec_ind = rightTSDF.df.columns.filter(x => x.endsWith("rec_ind"))(0)

    val combinedDF = leftTSDF.df
      .unionByName(rightTSDF.df)
      .withColumn(combinedTsCol,
        coalesce(col(leftTSDF.tsColumn.name), col(rightTSDF.tsColumn.name)))
        .withColumn("rec_ind", coalesce(col(left_rec_ind), col(right_rec_ind)))

    TSDF(combinedDF, combinedTsCol, leftTSDF.partitionCols.map(_.name):_*)
  }

  // helper method to obtain the columns from time series B to paste onto time series A
  def getLastRightRow(tsdf: TSDF, left_ts_col: StructField, rightCols: Seq[StructField], maxLookback : Int): TSDF = {
    // TODO: Add functionality for secondary sort key

    var maxLookbackValue = if (maxLookback == 0) Int.MaxValue else maxLookback
    val window_spec: WindowSpec =  Window
      .partitionBy(tsdf.partitionCols.map(x => col(x.name)):_*)
      .orderBy(tsdf.tsColumn.name, "rec_ind")
      .rowsBetween(-maxLookbackValue , 0L)

    val left_rec_ind = tsdf.df.columns.filter(x => x.endsWith("rec_ind"))(0)
    val right_rec_ind = tsdf.df.columns.filter(x => x.endsWith("rec_ind"))(1)

    // use the built-in Spark window last function (ignore nulls) to get the last record from the AS OF data framae
    // also record how many missing values are received so we report this per partition and aggregate the number of partition keys with missing values
    var df = rightCols
      .foldLeft(tsdf.df)((df, rightCol) =>
        df.withColumn(rightCol.name, last(df(rightCol.name), true)
          .over(window_spec)))
        //.withColumn("non_null_ct" + rightCol.name, count(rightCol.name).over(window_spec))
      .filter(col(left_ts_col.name).isNotNull).drop(col(tsdf.tsColumn.name)).drop(left_rec_ind).drop(right_rec_ind)

    df = df.drop("rec_ind")

    TSDF(df, left_ts_col.name, tsdf.partitionCols.map(_.name):_*)
  }

  // Create smaller time-based ranges inside of the TSDF partition
  def getTimePartitions(combinedTSDF: TSDF, tsPartitionVal: Int, fraction: Double): TSDF = {

    val tsColDouble = "ts_col_double"
    val tsColPartition = "ts_partition"
    val partitionRemainder = "partition_remainder"
    val isOriginal = "is_original"

    val partitionDF = combinedTSDF.df
      .withColumn(tsColDouble, col(combinedTSDF.tsColumn.name).cast("double"))
      .withColumn(tsColPartition, lit(tsPartitionVal) * (col(tsColDouble) / lit(tsPartitionVal)).cast("integer"))
      .withColumn(partitionRemainder,(col(tsColDouble) - col(tsColPartition)) / lit(tsPartitionVal))
      .withColumn(isOriginal, lit(1)).cache()

    // add [1 - fraction] of previous time partition to the next partition.
    val remainderDF = partitionDF
      .filter(col(partitionRemainder) >= lit(1 - fraction))
      .withColumn(tsColPartition, col(tsColPartition) + lit(tsPartitionVal))
      .withColumn(isOriginal, lit(0))

    val df = partitionDF.union(remainderDF).drop(partitionRemainder, tsColDouble)
    val newPartitionColNames = combinedTSDF.partitionCols.map(_.name) :+ tsColPartition

    partitionDF.unpersist()

    TSDF(df, combinedTSDF.tsColumn.name, newPartitionColNames:_*)
  }

  // wrapper method for executing the AS OF join based on various parameters
  def asofJoinExec(
    _leftTSDF: TSDF,
    _rightTSDF: TSDF,
    leftPrefix: Option[String],
    rightPrefix: String, maxLookback : Int, tsPartitionVal: Option[Int],
    fraction: Double = 0.1): TSDF= {

    var leftTSDF: TSDF = _leftTSDF.withColumn("rec_ind", lit(1))
    var rightTSDF: TSDF = _rightTSDF.withColumn("rec_ind", lit(-1))

    // set time partition value to maxLookback in case it is specified and tsPartitionVal > maxLookback
    val restrictLookback = (maxLookback > 0)
    var timePartition = tsPartitionVal match {
      case Some(s) => s
      case _ => 0
    }
    if (maxLookback > 0 & timePartition > 0) {timePartition = maxLookback}

    if(!checkEqualPartitionCols(leftTSDF,rightTSDF)) {
      throw new IllegalArgumentException("Partition columns of left and right TSDF should be equal.")
    }

    // add Prefixes to TSDFs
    val prefixedLeftTSDF = leftPrefix match {
      case Some(prefix_value) => addPrefixToColumns(
        leftTSDF,
        leftTSDF.observationColumns :+ leftTSDF.tsColumn,
        prefix_value)
      case None => leftTSDF
    }

    val prefixedRightTSDF = addPrefixToColumns(
      rightTSDF,
      rightTSDF.observationColumns :+ rightTSDF.tsColumn,
      rightPrefix)

    // get all non-partition columns (including ts col)
    val leftCols = prefixedLeftTSDF.observationColumns :+ prefixedLeftTSDF.tsColumn
    val rightCols = prefixedRightTSDF.observationColumns :+ prefixedRightTSDF.tsColumn

    // perform asof join
    val combinedTSDF = combineTSDF(
      addColumnsFromOtherDF(prefixedLeftTSDF, rightCols),
      addColumnsFromOtherDF(prefixedRightTSDF, leftCols))

    val asofTSDF: TSDF = timePartition > 0 match {
      case true => {
        val timePartitionedTSDF = getTimePartitions(combinedTSDF, timePartition, fraction)

        val asofDF = getLastRightRow(timePartitionedTSDF,prefixedLeftTSDF.tsColumn,rightCols, maxLookback).df
          .filter(col("is_original") === lit(1))
          .drop("ts_partition","is_original")

        TSDF(asofDF, prefixedLeftTSDF.tsColumn.name, leftTSDF.partitionCols.map(_.name):_*)
      }
      case false => getLastRightRow(combinedTSDF, prefixedLeftTSDF.tsColumn,rightCols, maxLookback)
    }
    asofTSDF
  }
}
