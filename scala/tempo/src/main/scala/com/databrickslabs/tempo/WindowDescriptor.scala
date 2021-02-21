package com.databrickslabs.tempo

import java.time.Duration

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
object WindowAlignments extends Enumeration
{
	type WindowAlignment = Value
	val Leading, Centered, Trailing = Value
}

/**
 * A description of a window that can be created for a specific [[TSDF]]
 */
sealed trait WindowDescriptor
{
	/**
	 * @param tsdf the [[TSDF]] to use
	 * @return a [[WindowSpec]] that instantiates this [[WindowDescriptor]] for the given [[TSDF]]
	 */
	def apply( tsdf: TSDF ): WindowSpec
}

/**
 * Description of how to construct a window from numeric lengths and offsets
 * @param ordering how the window should be ordered
 * @param length the window length
 * @param offset the window offset
 */
final case class LongWindowDescriptor( ordering: Orderings.Ordering, length: Long, offset: Long = Window.currentRow )
	extends WindowDescriptor
{
	import Orderings._

	/**
	 * @param tsdf the [[TSDF]] to use
	 *  @return a [[WindowSpec]] that instantiates this [[WindowDescriptor]] for the given [[TSDF]]
	 */
	def apply( tsdf: TSDF ): WindowSpec =
		ordering match {
			case RowBased => tsdf.windowOverRows(length,offset)
			case RangeBased => tsdf.windowOverRange(length,offset)
		}
}

object WindowDescriptor
{
	import Orderings._
	import WindowAlignments._

	/**
	 * Build an aligned [[WindowDescriptor]]
	 * @param alignment the alignment of the window
	 * @param ordering the ordering of the window
	 * @param size the size of the window
	 * @param includesNow whether the window should include the current row/value (centered windows always include the current row)
	 * @return a [[WindowDescriptor]] describing an aligned window as defined by the parameters
	 */
	def apply( alignment: WindowAlignment, ordering: Ordering, size: Long, includesNow: Boolean = false): WindowDescriptor =
		alignment match {
			case Leading => leadingWindow(ordering, size, includesNow)
			case Centered => centeredWindow(ordering, size)
			case Trailing => trailingWindow(ordering, size, includesNow)
		}

	/**
	 * Build a leading window descriptor
	 * @param ordering the ordering of the window
	 * @param size the size of the window
	 * @param includesNow whether the window should include the current observation
	 * @return a [[WindowDescriptor]] representing a leading window
	 */
	def leadingWindow(ordering: Ordering, size: Long, includesNow: Boolean = false): WindowDescriptor =
	{
		assert( size > 0 )
		if(includesNow)
			LongWindowDescriptor(ordering,size,Window.currentRow)
		else
			LongWindowDescriptor(ordering,size,1)
	}

	/**
	 * Build a centered window
	 * @param ordering the ordering of the window
	 * @param size the size of the window
	 * @return a [[WindowDescriptor]] for a centered window
	 */
	def centeredWindow(ordering: Ordering, size: Long): WindowDescriptor =
	{
		assert( size > 1 )
		LongWindowDescriptor(ordering, size, -(0.5 * (size-1)).floor.toLong)
	}

	/**
	 * Build a trailing window
	 * @param ordering the ordering of the window
	 * @param size the size of the window
	 * @param includesNow whether the window should include the current observation
	 * @return a [[WindowDescriptor]] representing a trailing window
	 */
	def trailingWindow(ordering: Ordering, size: Long, includesNow: Boolean = false): WindowDescriptor =
	{
		assert(size > 0)
		if(includesNow)
			LongWindowDescriptor(ordering, -size, Window.currentRow)
		else
			LongWindowDescriptor(ordering, -size, -1)
	}
}