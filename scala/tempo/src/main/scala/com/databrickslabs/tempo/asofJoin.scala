package com.databrickslabs.tempo

import org.apache.spark.sql.expressions.{Window, WindowSpec}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructField

object asofJoin {

  def checkEqualPartitionCols(leftTSDF: TSDF, rightTSDF: TSDF): Boolean = {
    leftTSDF.partitionCols.zip(rightTSDF.partitionCols).forall(p => p._1 == p._2)
  }

  // add prefix to all specified columns
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

  //
  def combineTSDF(leftTSDF: TSDF, rightTSDF: TSDF): TSDF = {

    val combinedTsCol = "combined_ts"

    val combinedDF = leftTSDF.df
      .unionByName(rightTSDF.df)
      .withColumn(combinedTsCol,
        coalesce(col(leftTSDF.tsColumn.name), col(rightTSDF.tsColumn.name)))

    TSDF(combinedDF, combinedTsCol, leftTSDF.partitionCols.map(_.name):_*)
  }

  def getLastRightRow(tsdf: TSDF, left_ts_col: StructField, rightCols: Seq[StructField]): TSDF = {
    // TODO: Add functionality for sort_keys (@ricardo)


    val window_spec: WindowSpec =  Window
      .partitionBy(tsdf.partitionCols.map(x => col(x.name)):_*)
      .orderBy(tsdf.tsColumn.name)

    val df = rightCols
      .foldLeft(tsdf.df)((df, rightCol) =>
        df.withColumn(rightCol.name, last(df(rightCol.name), true)
          .over(window_spec)))
      .filter(col(left_ts_col.name).isNotNull).drop(col(tsdf.tsColumn.name))

    TSDF(df, left_ts_col.name, tsdf.partitionCols.map(_.name):_*)
  }

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

    TSDF(df, combinedTSDF.tsColumn.name, newPartitionColNames:_*)
  }

  def asofJoinExec(
    leftTSDF: TSDF,
    rightTSDF: TSDF,
    leftPrefix: Option[String],
    rightPrefix: String, tsPartitionVal: Option[Int],
    fraction: Double = 0.1): TSDF= {

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

    val asofTSDF: TSDF = tsPartitionVal match {
      case Some(partitionValue) => {
        val timePartitionedTSDF = getTimePartitions(combinedTSDF, partitionValue, fraction)
        val asofDF = getLastRightRow(timePartitionedTSDF,prefixedLeftTSDF.tsColumn,rightCols).df
          .filter(col("is_original") === lit(1))
          .drop("ts_partition","is_original")

        TSDF(asofDF, prefixedLeftTSDF.tsColumn.name, leftTSDF.partitionCols.map(_.name):_*)
      }
      case None => getLastRightRow(combinedTSDF, prefixedLeftTSDF.tsColumn,rightCols)
    }
    asofTSDF
  }
}
