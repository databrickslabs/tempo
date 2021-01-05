package com.databricks.tempo

import org.apache.spark.sql.expressions.{Window, WindowSpec}

/**
 * Types of window orderings
 */
object Orderings extends Enumeration
{
	type Ordering = Value
	val RowBased, RangeBased = Value
}

/**
 * Types of window alignments
 */
object Alignments extends Enumeration
{
	type Alignment = Value
	val Leading, Centered, Trailing = Value
}

object WindowHelpers
{
	import Alignments._

	/**
	 * A function to construct a window from (start,end) bounds
	 */
	type WindowBuilder = (Long,Long) => WindowSpec

	/**
	 * Calculate bounds for windows
	 * @param builder function to construct a window given start and end bounds
	 * @param alignment the alignment type of the window
	 * @param size the size of the window
	 * @param includesNow whether the window should include the current observation
	 * @return a window with bounds as defined by the given arguments
	 */
	def buildWindow(builder: WindowBuilder)
	               (alignment: Alignment, size: Long, includesNow: Boolean = false): WindowSpec =
		alignment match {
			case Leading => buildLeadingWindow(builder)(size, includesNow)
			case Centered => buildCenteredWindow(builder)(size)
			case Trailing => buildTrailingWindow(builder)(size, includesNow)
		}

	/**
	 * Calculate bounds for leading windows
	 * @param builder function to construct a window given start and end bounds
	 * @param size the size of the window
	 * @param includesNow whether the window should include the current observation
	 * @return a leading window with bounds as defined by the given arguments
	 */
	def buildLeadingWindow(builder: WindowBuilder)
	                      (size: Long, includesNow: Boolean = false): WindowSpec =
		if(includesNow) {
			assert( size > 1 )
			builder(Window.currentRow,(size-1))
		} else {
			assert( size > 0 )
			builder(1,size)
		}

	/**
	 * Calculate bounds for centered windows
	 * @param builder function to construct a window given start and end bounds
	 * @param size the size of the window
	 * @return a centered window with bounds as defined by the given arguments
	 */
	def buildCenteredWindow(builder: WindowBuilder)(size: Long): WindowSpec =
	{
		assert( size > 1 )
		builder( -(0.5 * (size-1)).floor.toLong, (0.5 * (size-1)).ceil.toLong )
	}

	/**
	 * Calculate bounds for trailing windows
	 * @param builder function to construct a window given start and end bounds
	 * @param size the size of the window
	 * @param includesNow whether the window should include the current observation
	 * @return a trailing window with bounds as defined by the given arguments
	 */
	def buildTrailingWindow(builder: WindowBuilder)
	                       (size: Long, includesNow: Boolean = false): WindowSpec =
		if(includesNow) {
			assert( size > 1 )
			builder( -(size-1), Window.currentRow)
		} else {
			assert( size > 0 )
			builder(-size, -1)
		}
}
