package com.databricks.tempo

import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.sql.expressions.{Window, WindowSpec}
import org.apache.spark.sql.functions.row_number
import org.apache.spark.sql.types._

/**
 * The timeseries DataFrame
 */
sealed trait TSDF
{
	// core backing values

	/**
	 * The [[DataFrame]] backing this [[TSDF]]
	 */
	val df: DataFrame

	/**
	 * The timeseries index column
	 */
	val tsColumn: StructField

	/**
	 * Partitioning columns (used to separate one logical timeseries from another)
	 */
	val partitionColumns: Seq[StructField]

	// derived attributes

	/**
	 * Is this [[TSDF]] partitioned?
	 */
	val isPartitioned: Boolean

	/**
	 * Columns that define the structure of the [[TSDF]].
	 * These should be protected from arbitrary modification.
	 */
	val structuralColumns: Seq[StructField]

	/**
	 * Observation columns
	 */
	val observationColumns: Seq[StructField]

	/**
	 * Measure columns (numeric observations)
	 */
	val measureColumns: Seq[StructField]

	// transformation functions

	/**
	 * Add or modify partition columns
	 * @param partitionCols the names of columns used to partition the SeqDF
	 * @return a new SeqDF instance, partitioned according to the given columns
	 */
	def partitionedBy(partitionCols: String*): TSDF

	def select(cols: Column*): TSDF

	def select(col: String, cols: String*): TSDF

	def selectExpr(exprs: String*): TSDF

	def filter(condition: Column): TSDF

	def filter(conditionExpr: String): TSDF

	def where(condition: Column): TSDF

	def where(conditionExpr: String): TSDF

	def limit(n: Int): TSDF

	def union(other: TSDF): TSDF

	def unionAll(other: TSDF): TSDF

	def withColumn(colName: String, col: Column): TSDF

	def withColumnRenamed(existingName: String, newName: String): TSDF

	def drop(colName: String): TSDF

	def drop(colNames: String*): TSDF

	def drop(col: Column): TSDF

	// window builder functions

	/**
	 * Construct a base window for this [[TDSF]]
	 * @return a base [[WindowSpec]] from which other windows can be constructed
	 */
	protected def baseWindow(): WindowSpec

	/**
	 * Construct a row-based window for this [[TSDF]]
	 * @param start start row for the window
	 * @param end end row for the window
	 * @return a [[WindowSpec]] appropriate for applying functions to this [[TSDF]]
	 */
	def windowBetweenRows(start: Long, end: Long): WindowSpec

	/**
	 * Construct a row-based window for this [[TSDF]]
	 * @param length length in rows of the window
	 * @param offset offset of the window from the current row
	 * @return a [[WindowSpec]] appropriate for applying functions to this [[TSDF]]
	 */
	def windowOverRows(length: Long, offset: Long = Window.currentRow): WindowSpec

	/**
	 * Construct a range-based window for this [[TSDF]]
	 * @param start start value for the window
	 * @param end end value for the window
	 * @return a [[WindowSpec]] appropriate for applying functions to this [[TSDF]]
	 */
	def windowBetweenRange(start: Long, end: Long): WindowSpec

	/**
	 * Construct a range-based window for this [[TSDF]]
	 * @param length length of the window
	 * @param offset offset of the window from the current
	 * @return a [[WindowSpec]] appropriate for applying functions to this [[TSDF]]
	 */
	def windowOverRange(length: Long, offset: Long = Window.currentRow): WindowSpec
}

/**
 * Core implementation of [[TSDF]]
 * @param df source [[DataFrame]]
 * @param tsColumn timeseries column
 * @param partitionColumns partitioning columns
 */
private[tempo] sealed class BaseTSDF(val df: DataFrame,
                                     val tsColumn: StructField,
                                     val partitionColumns: StructField* )
	extends TSDF
{
	// Validate the arguments
	assert( df.schema.contains(tsColumn),
	        s"The given DataFrame does not contain the given timeseries column ${tsColumn.name}!")
	assert( TSDF.validTSColumnTypes.contains(tsColumn.dataType),
	        s"The given timeseries column ${tsColumn.name} is of a type (${tsColumn.dataType}) that cannot be used as an index!")

	partitionColumns.foreach( pCol => assert( df.schema.contains(pCol),
	                                          s"The given DataFrame does not contain the given partition column ${pCol.name}!" ))

	// Define some other values

	/**
	 * Is this [[TSDF]] partitioned?
	 */
	val isPartitioned: Boolean = partitionColumns.nonEmpty

	/**
	 * Columns that define the structure of the [[TSDF]].
	 * These should be protected from arbitrary modification.
	 */
	val structuralColumns: Seq[StructField] = Seq(tsColumn) ++ partitionColumns

	/**
	 * Observation columns
	 */
	val observationColumns: Seq[StructField] =
		df.schema.filter( structuralColumns.contains(_) )

	/**
	 * Measure columns (numeric observations)
	 */
	val measureColumns: Seq[StructField] =
		observationColumns.filter(_.dataType.isInstanceOf[NumericType])

	// Transformations

	/**
	 * Add or modify partition columns
	 *
	 * @param partitionCols the names of columns used to partition the SeqDF
	 * @return a new SeqDF instance, partitioned according to the given columns
	 */
	def partitionedBy(partitionCols: String*): TSDF =
		TSDF(df, tsColumn.name, partitionCols :_*)

	def select(cols: Column*): TSDF = ???
	def select(col: String,
	                    cols: String*): TSDF = ???
	def selectExpr(exprs: String*): TSDF = ???
	def filter(condition: Column): TSDF = ???
	def filter(conditionExpr: String): TSDF = ???
	def where(condition: Column): TSDF = ???
	def where(conditionExpr: String): TSDF = ???
	def limit(n: Int): TSDF = ???
	def union(other: TSDF): TSDF = ???
	def unionAll(other: TSDF): TSDF = ???
	def withColumn(colName: String,
	                        col: Column): TSDF = ???
	def withColumnRenamed(existingName: String,
	                               newName: String): TSDF = ???
	def drop(colName: String): TSDF = ???
	def drop(colNames: String*): TSDF = ???
	def drop(col: Column): TSDF = ???

	// Window builder functions

	/**
	 * Construct a base window for this [[TDSF]]
	 *
	 * @return a base [[WindowSpec]] from which other windows can be constructed
	 */
	protected def baseWindow(): WindowSpec =
	{
		// order by total ordering column
		val w = Window.orderBy(tsColumn.name)

		// partition if needed
		if( this.isPartitioned )
			w.partitionBy(partitionColumns.head.name, partitionColumns.tail.map(_.name) :_*)
		else
			w
	}

	/**
	 * Construct a row-based window for this [[TSDF]]
	 *
	 * @param start start row for the window
	 * @param end   end row for the window
	 * @return a [[WindowSpec]] appropriate for applying functions to this [[TSDF]]
	 */
	def windowBetweenRows(start: Long, end: Long): WindowSpec =
		baseWindow().rowsBetween(start, end)

	/**
	 * Construct a range-based window for this [[TSDF]]
	 *
	 * @param start start value for the window
	 * @param end   end value for the window
	 * @return a [[WindowSpec]] appropriate for applying functions to this [[TSDF]]
	 */
	def windowBetweenRange(start: Long, end: Long): WindowSpec =
		baseWindow().rangeBetween(start, end)

	/**
	 * Construct a row-based window for this [[TSDF]]
	 *
	 * @param length length in rows of the window
	 * @param offset offset of the window from the current row
	 * @return a [[WindowSpec]] appropriate for applying functions to this [[TSDF]]
	 */
	def windowOverRows(length: Long, offset: Long): WindowSpec =
		windowBetweenRows(offset, (offset + length - 1))

	/**
	 * Construct a range-based window for this [[TSDF]]
	 *
	 * @param length length of the window
	 * @param offset offset of the window from the current
	 * @return a [[WindowSpec]] appropriate for applying functions to this [[TSDF]]
	 */
	def windowOverRange(length: Long, offset: Long): WindowSpec =
		windowBetweenRange(offset, (offset + length - 1))
}

/**
 * [[TSDF]] companion object - contains static constructors
 */
object TSDF
{
	// Utility functions

	final val DEFAULT_SEQ_COLNAME = "sequence_num"

	/**
	 * Valid timestamp column types
	 */
	final val validTSColumnTypes = Seq[DataType](ByteType,
	                                             ShortType,
	                                             IntegerType,
	                                             LongType,
	                                             TimestampType,
	                                             DateType)

	/**
	 * Retrieve a column by name from the dataframe by name
	 * @param df the [[DataFrame]] with the column
	 * @param colName the name of the column
	 * @return the named column of the [[DataFrame]], if it exists,
	 *         otherwise a [[NoSuchElementException]] is thrown
	 */
	private def colByName(df: DataFrame)(colName: String): StructField =
		df.schema.find(_.name.toLowerCase() == colName.toLowerCase()).get

	// TSDF Constructors

	/**
	 * @constructor
	 * @param df the source [[DataFrame]]
	 * @param tsColumnName the name of the timeseries column
	 * @param partitionColumnNames the names of the paritioning columns
	 * @return a [[TSDF]] object
	 */
	def apply( df: DataFrame,
	           tsColumnName: String,
	           partitionColumnNames: String* ): TSDF =
	{
		val colFinder = colByName(df) _
		val tsColumn = colFinder(tsColumnName)
		val partitionColumns = partitionColumnNames.map(colFinder)

		new BaseTSDF(df, tsColumn, partitionColumns :_*)
	}

	/**
	 * @constructor
	 * @param df
	 * @param orderingColumns
	 * @param sequenceColName
	 * @param partitionColumns
	 */
	def apply( df: DataFrame,
	           orderingColumns: Seq[String],
	           sequenceColName: String,
	           partitionColumns: String* ): TSDF =
	{
		// define window for ordering according to the combined sequence columns
		val seq_win =
			if( partitionColumns.nonEmpty)
				Window.partitionBy(partitionColumns.map(df(_)) :_*)
				      .orderBy(orderingColumns.map(df(_)) :_*)
			else
				Window.orderBy(orderingColumns.map(df(_)) :_*)

		// add total ordering column
		val newDF = df.withColumn(sequenceColName, row_number().cast(LongType).over(seq_win))

		// construct our TSDF
		apply( newDF, sequenceColName, partitionColumns :_* )
	}

	/**
	 * @constructor
	 * @param df
	 * @param orderingColumns
	 * @param partitionColumns
	 * @return
	 */
	def apply(df: DataFrame,
	          orderingColumns: Seq[String],
	          partitionColumns: String* ): TSDF =
		apply( df, orderingColumns, DEFAULT_SEQ_COLNAME, partitionColumns :_* )
}