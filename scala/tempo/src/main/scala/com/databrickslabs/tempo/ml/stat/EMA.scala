package com.databrickslabs.tempo.ml.stat

import com.databrickslabs.tempo.ml.Orderings.Ordering
import com.databrickslabs.tempo.ml.WindowAlignments.WindowAlignment
import com.databrickslabs.tempo.ml._
import com.databrickslabs.tempo.{TSDF, TsStructType}
import org.apache.spark.ml.param.{DoubleParam, IntParam, LongParam, Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.{DataTypes, StructField}
import org.apache.spark.sql.{functions => fn}

import scala.math.pow

class EMA(uid: String)
	extends TSDFTransformer(uid) with HasMeasureCol with HasOutputCol
{
	def this() = this(Identifiable.randomUID("EMA"))

	override def copy(extra: ParamMap): TSDFTransformer = defaultCopy(extra)

	// parameters

	/** @group param */
	final val N = new IntParam(this,
	                           "period",
	                           "The size of the lookback period over which we calculate the EMA")

	/** @group getParam */
	final def getN: Int = $(N)

	/** @group setParam */
	final def setN( value: Int ): EMA = set(N,value)

	/** @group param */
	final val alpha = new DoubleParam(this,
	                                  "expFactor",
	                                  "Exponential Factor Alpha")
	setDefault(alpha, EMA.DEFAULT_ALPHA)

	/** @group getParam */
	final def getAlpha: Double = $(alpha)

	/** @group setParam */
	final def setAlpha(value: Double): EMA =
		set(alpha, value)

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
	override def transformSchema(schema: TsStructType): TsStructType =
	{
		// make sure we have a valid measure column
		validateMeasureCol(schema)
		// make sure we don't already have the output column
		assert(! schema.fieldNames.contains($(outputCol)),
		       s"TSDF already contains the output column ${$(outputCol)}")

		new TsStructType(schema.fields :+ StructField($(outputCol), DataTypes.DoubleType),
		                 schema.tsColumn,
		                 schema.partitionCols)
	}

	/**
	 *
	 * @param k
	 * @return
	 */
	private def weightFn(k: Int): Double =
		$(alpha) * pow(1.0 - $(alpha), k)

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
		val window = tsdf.baseWindow()

		// temporary lag column names
		val lag_cols = Seq.range(0,($(N)+1)).map( i => LagCol(i, lagColName(i)) )

		// compute lag columns
		val with_lag_cols =
			lag_cols.foldLeft(tsdf)( (df, lc) =>
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
	val DEFAULT_ALPHA = 0.2

	/**
	 *
	 * @param measureCol
	 * @param emaCol
	 * @param N
	 * @param alpha
	 * @return
	 */
	def apply(measureCol: String,
	          emaCol: String,
	          N: Int,
	          alpha: Double): EMA =
		new EMA().setMeasureCol(measureCol)
		         .setOutputCol(emaCol)
		         .setN(N)
		         .setAlpha(alpha)

	/**
	 *
	 * @param measureCol
	 * @param N
	 * @param alpha
	 * @return
	 */
	def apply(measureCol: String,
	          N: Int,
	          alpha: Double): EMA =
		new EMA().setMeasureCol(measureCol)
		         .setN(N)
		         .setAlpha(alpha)

	/**
	 *
	 * @param measureCol
	 * @param emaCol
	 * @param N
	 * @return
	 */
	def apply(measureCol: String,
	          emaCol: String,
	          N: Int): EMA =
		apply(measureCol, emaCol, N, DEFAULT_ALPHA)

	/**
	 *
	 * @param measureCol
	 * @param N
	 * @return
	 */
	def apply(measureCol: String,
	          N: Int): EMA =
		apply(measureCol, N, DEFAULT_ALPHA)
}
