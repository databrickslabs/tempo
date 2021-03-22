package com.databrickslabs.tempo.ml.stat

import com.databrickslabs.tempo.{TSDF, TSStructType}
import com.databrickslabs.tempo.ml.{HasMeasureCol, HasWindow, Orderings, TSDFTransformer, WindowAlignments}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.{DataTypes, StructField, StructType}
import org.apache.spark.sql.functions.mean

/**
 * Transformer for simple moving averages
 * @param uid unique id
 */
class SMA(uid: String)
	extends TSDFTransformer(uid) with HasWindow with HasMeasureCol with HasOutputCol
{
	def this() = this(Identifiable.randomUID("SMA"))

	override def copy(extra: ParamMap): TSDFTransformer = defaultCopy(extra)

	// custom output column names

	private def setDefaultOutputColumn =
		if($(outputCol) == null || $(outputCol).isEmpty)
			set(outputCol, "sma_"+ $(windowAlignment) +"_"+ $(windowSize) +"_"+ $(measureCol))

	/** @group setParam */
	override def setWindowSize(value: Long): SMA.this.type =
	{
		super.setWindowSize(value)
		setDefaultOutputColumn
		this
	}



	// transform

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
	override def transformSchema(schema: TSStructType): TSStructType =
	{
		validateMeasureCol(schema)
		// make sure we don't already have the output column
		assert(! schema.fieldNames.contains($(outputCol)),
		       s"TSDF already contains the output column ${$(outputCol)}")

		new TSStructType(schema.fields :+ StructField($(outputCol), DataTypes.DoubleType),
		              schema.tsColumn,
		              schema.partitionCols :_*)
	}

	/**
	 * Transform the given [[TSDF]]
	 *
	 * @param tsdf the timeseries dataframe to transform
	 * @return the transformed timeseries dataframe
	 */
	override def transform(tsdf: TSDF): TSDF =
	{
		transformSchema(tsdf.schema)
		tsdf.withColumn($(outputCol), mean($(measureCol)).over(buildWindow(tsdf)))
	}
}

object SMA
{
	import com.databrickslabs.tempo.ml.Orderings._
	import com.databrickslabs.tempo.ml.WindowAlignments._

	/**
	 *
	 * @param size
	 * @param ordering
	 * @param alignment
	 * @param includesNow
	 * @return
	 */
	def apply(size: Long,
	          ordering: Ordering,
	          alignment: WindowAlignment,
	          includesNow: Boolean): SMA =
		new SMA().setWindowSize(size)
		         .setWindowOrdering(ordering)
		         .setWindowAlignment(alignment)
		         .setIncludesNow(includesNow)

	/**
	 *
	 * @param size
	 * @param ordering
	 * @param alignment
	 * @return
	 */
	def apply(size: Long,
	          ordering: Ordering,
	          alignment: WindowAlignment): SMA =
		apply(size,ordering,alignment,true)

	/**
	 *
	 * @param size
	 * @param ordering
	 * @return
	 */
	def apply(size: Long, ordering: Ordering): SMA =
		apply(size,ordering,trailing)

	/**
	 *
	 * @param size
	 * @return
	 */
	def apply(size:Long): SMA =
		apply(size,row)

	/**
	 *
	 * @param size
	 * @param ordering
	 * @param alignment
	 * @param includesNow
	 * @return
	 */
	def apply( size: Long,
	           ordering: String,
	           alignment: String,
	           includesNow: Boolean): SMA =
		apply(size, Orderings.withName(ordering), WindowAlignments.withName(alignment), includesNow)

	/**
	 *
	 * @param size
	 * @param ordering
	 * @param alignment
	 * @return
	 */
	def apply( size: Long,
	           ordering: String,
	           alignment: String): SMA =
		apply(size,Orderings.withName(ordering),WindowAlignments.withName(alignment))

	/**
	 *
	 * @param size
	 * @param ordering
	 * @return
	 */
	def apply(size: Long, ordering: String): SMA =
		apply(size,Orderings.withName(ordering),trailing)
}