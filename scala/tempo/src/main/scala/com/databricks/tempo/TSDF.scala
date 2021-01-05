package com.databricks.tempo

import com.databricks.tempo.Alignments.Alignment
import com.databricks.tempo.Orderings.{Ordering, RangeBased, RowBased}
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.param.{Param, ParamMap, StringArrayParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.{Window, WindowSpec}
import org.apache.spark.sql.types._

import java.time.temporal.{ChronoUnit, TemporalAmount, TemporalUnit}
import java.time.{Duration, Period}
import scala.reflect.runtime.universe.TypeTag

/**
 * TSDF: the Timeseries DataFrame
 *
 * @constructor creates a new instance of [[TSDF]]
 * @param df the source [[DataFrame]]
 * @param tsCol the name of the timeseries column
 * @param partitionCols columns used to distinguish different sequences from one another
 * @tparam T type of the timeseries column
 * @tparam D type of the sampling period
 */
sealed abstract class TSDF[T <: DataType, D <: TemporalAmount](override val df: DataFrame,
                                                               val tsCol: String,
                                                               override val partitionCols: Seq[String] )
                                                              (implicit val tt: TypeTag[T], implicit val dt: TypeTag[D])
  extends SeqDF(df, tsCol, partitionCols)
{
  /** Validate the timeseries column */
  SeqDF.columnValidator(df.schema)("timeseries",TSDF.tsColTypes)(tsCol)

  /**
   * The base unit of this [[TSDF]]
   */
  val base: TemporalUnit

  /**
   * Converts an amount of time as defined in terms of the parameter [[D]]
   * to a value in terms of the timeseries column type [[T]]
   * @param amount the amount of time to convert
   * @return the corresponding value in the units of the timeseries column
   */
  protected def timeInBaseUnits(amount: D): Long = amount.get(base)

  /**
   * Construct a window for this [[TSDF]]
   * @param start starting bound for the window range
   * @param end ending boutnd for the window range
   * @return a window appropriate for applying functions to this [[TSDF]]
   */
  protected def rangeBetweenBaseWindow(start: Long, end: Long): WindowSpec =
  {
    // make sure our bounds make sense
    assert(end > start)

    // order by total timestamp as a Long column
    val w = Window.orderBy(df(tsCol).cast(LongType))

    // partition if needed
    val baseWindow = if( this.isPartitioned )
                       w.partitionBy(partitionCols.head, partitionCols.tail :_*)
                     else
                       w
    // apply rows between
    baseWindow.rowsBetween(start, end)
  }

  /**
   * Construct a window for this [[TSDF]]
   * @param ordering the type of window ordering to use
   * @param alignment the type of window alignment to build
   * @param size the size of the window
   * @param includesNow whether the window should include the current observation
   * @return a window appropriate for calculating window operations across this [[TSDF]]
   */
  override def window(ordering: Ordering, alignment: Alignment, size: Long, includesNow: Boolean): WindowSpec =
    ordering match {
      case RowBased => WindowHelpers.buildWindow(rowsBetweenWindow)(alignment,size,includesNow)
      case RangeBased => WindowHelpers.buildWindow(rangeBetweenBaseWindow)(alignment,size,includesNow)
    }

  /**
   * Construct a window for this [[TSDF]]
   * @param ordering the type of window ordering to use
   * @param alignment the type of window alignment to build
   * @param size the size of the window
   * @param includesNow whether the window should include the current observation
   * @return a window appropriate for calculating window operations across this [[TSDF]]
   */
  def window(ordering: Ordering, alignment: Alignment, size: D, includesNow: Boolean): WindowSpec =
    window(ordering, alignment, timeInBaseUnits(size), includesNow)
}

/**
 *
 * @param df the source [[DataFrame]]
 * @param tsCol the name of the timeseries column
 * @param partitionCols columns used to distinguish different sequences from one another
 */
class TimestampDF(override val df: DataFrame,
                  override val tsCol: String,
                  override val partitionCols: Seq[String] )
  extends TSDF[TimestampType,Duration](df, tsCol, partitionCols)
{
  /**
   * The base unit of this [[TSDF]]
   */
  override val base: TemporalUnit = ChronoUnit.SECONDS
}

/**
 *
 * @param df the source [[DataFrame]]
 * @param tsCol the name of the timeseries column
 * @param partitionCols columns used to distinguish different sequences from one another
 */
class DateDF(override val df: DataFrame,
             override val tsCol: String,
             override val partitionCols: Seq[String] )
  extends TSDF[DateType,Period](df, tsCol, partitionCols)
{
  /**
   * The base unit of this [[TSDF]]
   */
  override val base: TemporalUnit = ChronoUnit.DAYS
}

object TSDF
{
  /**
   * Valid timestamp column types
   */
  final val tsColTypes = Seq( TimestampType, DateType )

  /**
   * Helper constructor
   * @param df
   * @param tsColName
   * @param partitionCols
   * @return
   */
  def apply(df: DataFrame, tsColName: String, partitionCols: Seq[String]): TSDF[_,_] =
    df.schema.fields.find(_.name.toLowerCase == tsColName.toLowerCase) match {
      case Some(tsCol) => tsCol.dataType match {
        case TimestampType => new TimestampDF(df, tsColName, partitionCols)
        case DateType => new DateDF(df, tsColName, partitionCols)
        case colType =>
          throw new IllegalArgumentException(s"Column $tsColName is of type $colType, which cannot be used as a timeseries index!")
      }
      case None => throw new IllegalArgumentException(s"No column named $tsColName exists in the given DataFrame!")
    }
}

/**
 * Transformer to convert a [[DataFrame]] to a [[TSDF]]
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

  def transform(df: DataFrame): TSDF[_,_] =
    TSDF(df, $(tsColName), $(partitionColNames))
}