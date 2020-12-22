package com.databricks.tempo

import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.param.{Param, ParamMap, StringArrayParam}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types._

/**
 * TSDF - the Timeseries DataFrame
 *
 * @constructor creates a new instance of [[TSDF]]
 * @param df the source [[DataFrame]]
 * @param tsCol the name of the timeseries column
 * @param sequencingColumns other sequence columns (used to generate the total ordering)
 * @param partitionCols columns used to distinguish different sequences from one another
 */
class TSDF(override val df: DataFrame,
           val tsCol: String,
           override val sequencingColumns: Seq[String],
           override val partitionCols: Seq[String] )
  extends SeqDF( df, (tsCol +: sequencingColumns), partitionCols)
{
  /** Validate the timeseries column */
  SeqDF.columnValidator(df.schema)("timeseries",TSDF.tsColTypes)(tsCol)

  /** Helper Constructors */

  /**
   * @constructor creates a new instance of [[TSDF]]
   * @param df the input [[DataFrame]]
   * @param tsCol the name of the timeseries column
   * @param partitionCols columns used to distinguish different sequences from one another
   */
  def this(df: DataFrame, tsCol: String, partitionCols: Seq[String]) =
    this(df, tsCol, Seq(), partitionCols)
}

object TSDF
{
  // valid types for timeseries index
  final val tsColTypes = Seq( TimestampType, DateType )
}

/**
 * Transformer to convert a [[DataFrame]] to a [[TSDF]]
 * @param uid
 */
class ToTSDF(override val uid: String)
  extends PipelineStage
{
  override def copy(extra: ParamMap): ToTSDF =
    defaultCopy(extra)

  /** Params */

  /** @group param */
  final val tsColName: Param[String] = new Param[String](this, "tsColName", "Name of the timeseries index column")
  setDefault(tsColName, "event_ts")

  /** @group param */
  final val partitionColNames: StringArrayParam =
    new StringArrayParam(this,
                         "partitionColNames",
                         "Names of the partitioning columns")

  /** @group getParam */
  final def getTsColName: String = $(tsColName)

  /** @group getParam */
  final def getPartitionColNames: Array[String] = $(partitionColNames)

  /** @group setParam */
  final def setTsColName(value: String): ToTSDF =
    set(tsColName, value)

  /** @group setParam */
  final def setPartitionColNames(values: Array[String]): ToTSDF =
    set(partitionColNames, values)

  /** Transform */

  override def transformSchema(schema: StructType): StructType =
  {
    // make sure the timeseries column exists and is of the right type
    val tsCol = schema.find(_.name.toLowerCase == $(tsColName).toLowerCase)
    assert( tsCol.isDefined, s"The given timeseries column ${$(tsColName)} does not exist in this DataFrame!" )
    assert( TSDF.tsColTypes.contains(tsCol.get.dataType),
            s"The given timeseries column ${$(tsColName)} of type ${tsCol.get.dataType} cannot be used as a temporal index!" )

    // validate the partition columns
    $(partitionColNames).foreach( pCol => assert(schema.fieldNames
                                                       .map(_.toLowerCase)
                                                       .contains(pCol.toLowerCase),
                                                 s"The partition column ${pCol} does not exist in this DataFrame!" ) )

    // all good - return the schema unchanged
    schema
  }

  def transform(df: DataFrame): TSDF =
    new TSDF(df, $(tsColName), $(partitionColNames))
}