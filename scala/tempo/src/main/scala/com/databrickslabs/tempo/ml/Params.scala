package com.databrickslabs.tempo.ml

import com.databrickslabs.tempo.{TSDF, TSStructType}
import org.apache.spark.ml.param.{BooleanParam, LongParam, Param, ParamValidators, Params}
import org.apache.spark.sql.expressions.{Window, WindowSpec}
import org.apache.spark.sql.types.StructType

/**
 * Types of window orderings
 */
object Orderings extends Enumeration
{
	type Ordering = Value
	val row, range = Value
}

/**
 * Types of window alignments
 */
object WindowAlignments extends Enumeration
{
	type WindowAlignment = Value
	val leading, centered, trailing = Value
}

trait HasMeasureCol extends Params
{
	/**
	 * Param for measure column name.
	 * @group param
	 */
	final val measureCol: Param[String] = new Param[String](this, "measureCol", "measure column name")

	/** @group getParam */
	final def getMeasureCol: String = $(measureCol)

	/** @group setParam */
	final def setMeasureCol(value: String): this.type =
		set(measureCol,value)

	protected def validateMeasureCol(schema: TSStructType): Unit =
		assert( schema.measureColumns.map(_.name.toLowerCase).contains($(measureCol).toLowerCase),
		        s"The measure column parameter must refer to a measure (numeric) column in the dataframe")
}

/**
 * Parameters for components that use Windows
 */
trait HasWindow extends Params
{
	import Orderings._
	import WindowAlignments._

	/**
	 * Parameter to specify a Window's Ordering
	 * @group param
	 */
	final val windowOrdering = new Param[Ordering](this,
	                                               "windowOrdering",
	                                               "Window Ordering")

	/** @group getParam */
	final def getWindowOrdering: Ordering = $(windowOrdering)

	/** @group setParam */
	def setWindowOrdering(value: Ordering): this.type =
		set(windowOrdering,value)

	/** @group setParam */
	def setWindowOrdering(value: String): this.type =
		setWindowOrdering(Orderings.withName(value))

	/**
	 * Parameter to specify a Window's Alignment
	 * @group param
	 */
	final val windowAlignment = new Param[WindowAlignment](this,"windowAlignment","Window Aligment")

	/** @group getParam */
	final def getWindowAlignment: WindowAlignment = $(windowAlignment)

	/** @group setParam */
	def setWindowAlignment(value: WindowAlignment): this.type =
		set(windowAlignment,value)

	/** @group setParam */
	def setWindowAlignment(value: String): this.type =
		setWindowAlignment(WindowAlignments.withName(value))

	/**
	 * Parameter to specify a Window's Size
	 * @group param
	 */
	final val windowSize = new LongParam(this,
	                                     "windowSize",
	                                     "Window size",
	                                     ParamValidators.gt[Long](0))

	/** @group getParam */
	final def getWindowSize: Long = $(windowSize)

	/** @group setParam */
	def setWindowSize(value: Long): this.type =
		set(windowSize,value)

	/**
	 * Parameer to set whether or not a window includes the current row
	 * @group param
	 */
	final val includesNow = new BooleanParam(this,
	                                         "includesNow",
	                                         "Whether or not the window includes current row")
	setDefault(includesNow,false)

	/** @group getParam */
	final def getIncludesNow: Boolean = $(includesNow)

	/** @group setParam */
	def setIncludesNow(value: Boolean): this.type =
		set(includesNow,value)

	/**
	 * Build a [[WindowSpec]] for the given [[TSDF]] based on the set parameters
	 * @param forTSDF the [[TSDF]] to build a window for
	 * @return an appropriately constructed [[WindowSpec]]
	 */
	protected def buildWindow(forTSDF: TSDF): WindowSpec =
	{
		// get builder method
		def windowBuilder(length: Long, offset: Long): WindowSpec =
			$(windowOrdering) match {
				case `row` => forTSDF.windowOverRows(length, offset)
				case `range` => forTSDF.windowOverRange(length, offset)
			}

		// build the window
		$(windowAlignment) match {
			case `leading` =>
				if($(includesNow))
					windowBuilder($(windowSize)-1,Window.currentRow)
				else
					windowBuilder($(windowSize),1)
			case `trailing` =>
				if($(includesNow))
					windowBuilder(-($(windowSize)-1),Window.currentRow)
				else
					windowBuilder(-$(windowSize),-1)
			case `centered` =>
				windowBuilder($(windowSize), -(0.5 * ($(windowSize)-1)).floor.toLong)
		}
	}
}