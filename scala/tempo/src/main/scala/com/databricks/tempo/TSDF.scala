package com.databricks.tempo

import org.apache.spark.sql.types.{DateType, DoubleType, FloatType, IntegerType, LongType, NumericType, StructField, TimestampType}
import org.apache.spark.sql.{Column, DataFrame}

class TSDF(val df: DataFrame,
           val tsCol: StructField,
           val partitionCols: Seq[StructField] = Seq() )
{
  // types for timestamps
  private val tsColTypes = Seq( IntegerType, LongType, DoubleType, FloatType, TimestampType, DateType )

  // validate the timestamp column
  assert( df.schema.fields.contains(tsCol), s"DataFrame does not contain timeseries column ${tsCol}!" )
  assert( tsColTypes.contains(tsCol.dataType),
          s"Timeseries column ${tsCol} is of a type that cannot be used for a time index!")

  // validate the partition columns
  partitionCols.foreach( pCol => assert( df.schema.fields.contains(pCol),
                                         s"DataFrame does not contain partition column ${pCol}") )

  // take a note of the other columns (not timeseries or partition)
  val otherCols = df.schema.fields.filter(col => (col != tsCol && ! partitionCols.contains(col)))

  // take note of measure columns (that have a numeric type)
  val measureCols = otherCols.filter(col => col.dataType.isInstanceOf[NumericType])
}
