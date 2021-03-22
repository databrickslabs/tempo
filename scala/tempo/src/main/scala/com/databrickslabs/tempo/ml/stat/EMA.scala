package com.databrickslabs.tempo.ml.stat

import com.databrickslabs.tempo.{TSDF, TSStructType}
import com.databrickslabs.tempo.ml.{HasMeasureCol, HasWindow, TSDFTransformer}
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.param.shared.HasOutputCol
import org.apache.spark.ml.util.Identifiable

import scala.math.pow

class EMA(uid: String)
	extends TSDFTransformer(uid) with HasWindow with HasMeasureCol with HasOutputCol
{
	def this() = this(Identifiable.randomUID("EMA"))

	override def copy(extra: ParamMap): TSDFTransformer = defaultCopy(extra)

	// parameters

	/** @group param */
	final val expFactor = new Param[Double]( this,
	                                         "expFactor",
	                                         "exponential factor" )

	/** @group getParam */
	final def getExpFactor: Double = $(expFactor)

	/** @group setParam */
	final def setExpFactor(value: Double): EMA =
		set(expFactor,value)

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
	override def transformSchema(schema: TSStructType): TSStructType = ???

	/**
	 *
	 * @param k
	 * @return
	 */
	private def weightFn(k: Int): Double =
		$(expFactor) * pow(1.0 - $(expFactor), k)

	/**
	 * Transform the given [[TSDF]]
	 *
	 * @param tsdf the timeseries dataframe to transform
	 * @return the transformed timeseries dataframe
	 */
	override def transform(tsdf: TSDF): TSDF = ???
}
