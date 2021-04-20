package com.databrickslabs.tempo.ml

import com.databrickslabs.tempo.{TSDF, TSStructType}
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.param.{Param, ParamMap, Params, StringArrayParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType

/**
 * Base type for [[TSDF]] Transformers
 *
 * @constructor
 * @param uid unique id
 */
abstract class TSDFTransformer(override val uid: String)
  extends Params
{
  override def copy(extra: ParamMap): TSDFTransformer

  /**
   * Check transform validity and derive the output schema from the input schema.
   *
   * We check validity for interactions between parameters during `transformSchema` and
   * raise an exception if any parameter value is invalid. Parameter value checks which
   * do not depend on other parameters are handled by `Param.validate()`.
   *
   * Typical implementation should first conduct verification on schema change and parameter
   * validity, including complex parameter interaction checks.
   */
  def transformSchema(schema: TSStructType): TSStructType

  /**
   * Transform the given [[TSDF]]
   * @param tsdf the timeseries dataframe to transform
   * @return the transformed timeseries dataframe
   */
  def transform(tsdf: TSDF): TSDF
}

/**
 * Transformer to convert a [[DataFrame]] to a [[TSDF]]
 *
 * @param uid
 */
class ToTSDF(override val uid: String)
  extends PipelineStage
{
  /**
   * @constructor create a new instance of the [[ToTSDF]] Transformer
   */
  def this() = this(Identifiable.randomUID("ToTSDF"))

  override def copy(extra: ParamMap): ToTSDF = defaultCopy(extra)

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

  def transformSchema(schema: StructType): StructType =
  {
    // make sure the timeseries column exists and is of the right type
    val tsCol = schema.find(_.name.toLowerCase == $(tsColName).toLowerCase)
    assert( tsCol.isDefined, s"The given timeseries column ${$(tsColName)} does not exist in this DataFrame!" )
    assert( TSDF.validTSColumnTypes.contains(tsCol.get.dataType),
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
  {
    transformSchema(df.schema)

    TSDF(df, $(tsColName), $(partitionColNames))
  }
}