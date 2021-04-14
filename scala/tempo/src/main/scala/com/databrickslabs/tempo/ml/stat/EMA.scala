package com.databrickslabs.tempo.ml.stat

import com.databrickslabs.tempo.{TSDF, TSStructType}
import com.databrickslabs.tempo.ml.{HasMeasureCol, HasWindow, TSDFTransformer}
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.param.shared.HasOutputCol
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.{DataTypes, StructField}
import org.apache.spark.sql.{functions => fn}

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
	setDefault(expFactor,EMA.DEFAULT_EXP_FACTOR)

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
	override def transformSchema(schema: TSStructType): TSStructType =
	{
		// make sure we have a valid measure column
		validateMeasureCol(schema)
		// make sure we don't already have the output column
		assert(! schema.fieldNames.contains($(outputCol)),
		       s"TSDF already contains the output column ${$(outputCol)}")

		new TSStructType(schema.fields :+ StructField($(outputCol), DataTypes.DoubleType),
		                 schema.tsColumn,
		                 schema.partitionCols :_*)
	}

	/**
	 *
	 * @param k
	 * @return
	 */
	private def weightFn(k: Int): Double =
		$(expFactor) * pow(1.0 - $(expFactor), k)

	private case class LagCol(lag: Int, colname: String)
	private def lagColName(i: Int): String = $(measureCol) +s"_templag_$i"

	/**
	 * Transform the given [[TSDF]]
	 *
	 * @param tsdf the timeseries dataframe to transform
	 * @return the transformed timeseries dataframe
	 */
	override def transform(tsdf: TSDF): TSDF =
	{
		// set up the window
		val window = buildWindow(tsdf)

		// temporary lag column names
		val lag_cols = Seq.range(1,$(windowSize).toInt).map( i => LagCol(i, lagColName(i)) )

		// compute lag columns
		val with_lag_cols = lag_cols.foldLeft(tsdf)( (df, lc) =>
			                                             df.withColumn(lc.colname,
			                                                           fn.lit(weightFn(lc.lag)) * fn.lag($(measureCol), lc.lag).over(window)) )
		// consolidate lag columns into EMA column
		val sum_lag_cols = lag_cols.map( lc => fn.when(fn.col(lc.colname).isNull, fn.lit(0))
		                                         .otherwise(fn.col(lc.colname)) )
		                           .reduce( (a,b) => a + b )
		val with_ema_col = with_lag_cols.withColumn($(outputCol), sum_lag_cols)

		// return the final tsdf without the temp lags
		with_ema_col.drop( lag_cols.map(_.colname) :_* )
	}
}

object EMA
{
	val DEFAULT_EXP_FACTOR = 0.2
}
