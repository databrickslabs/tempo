package com.databricks.tempo

import com.databricks.tempo.Alignments.Alignment
import com.databricks.tempo.Orderings.{Ordering, RangeBased, RowBased}
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.param.{Param, ParamMap, StringArrayParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.{Window, WindowSpec}
import org.apache.spark.sql.functions.row_number
import org.apache.spark.sql.types._

/**
 * A DataFrame ordered by an explicit sequencing column
 *
 * @constructor create a new [[SeqDF]] instance
 * @param sourceDF the source [[DataFrame]]
 * @param totalOrderingCol the sequence column (must be of an [[org.apache.spark.sql.types.IntegralType]])
 * @param sequencingColumns other sequence columns (used to generate the total ordering)
 * @param partitionCols columns used to distinguish different sequences from one another
 */
class SeqDF(sourceDF: DataFrame,
            val totalOrderingCol: String,
            val sequencingColumns: Seq[String],
            val partitionCols: Seq[String] )
{
	/** Validate arguments */
	private val columnValidator = SeqDF.columnValidator(sourceDF.schema) _

	// the source dataframe should not contain the total ordering column
	assert( ! sourceDF.schema.fieldNames.map(_.toLowerCase).contains(totalOrderingCol.toLowerCase),
	        s"Source DataFrame already contains a column named $totalOrderingCol!" )

	// validate sequencing columns
	private val seqColValidator = columnValidator("sequencing",Seq())(_)
	assert( sequencingColumns.nonEmpty, "There must be at least one sequencing column!" )
	sequencingColumns.foreach(seqColValidator)

	// validate the partition columns
	private val partitionValidator = columnValidator("partition",Seq())(_)
	partitionCols.foreach( partitionValidator )

	/** Generate total ordering over the DataFrame */

	// define window for ordering according to the combined sequence columns
	private val to_win =
		if( partitionCols.nonEmpty)
			Window.partitionBy(partitionCols.map(sourceDF(_)) :_*)
			      .orderBy(sequencingColumns.map(sourceDF(_)) :_*)
		else
			Window.orderBy(sequencingColumns.map(sourceDF(_)) :_*)

	// add total ordering column
	val df = sourceDF.withColumn(totalOrderingCol, row_number().cast(LongType).over(to_win))

	/** Calculate other properties */

	// if the SeqDF is partitioned or not
	val isPartitioned: Boolean = partitionCols.nonEmpty

	// take a note of the observation columns (not sequence or partition)
	val observationColumns: Seq[String] = df.schema.fieldNames.filter(col => {
		val lc_col = col.toLowerCase
		( lc_col != totalOrderingCol.toLowerCase &&
		  !sequencingColumns.map(_.toLowerCase).contains(lc_col) &&
		  ! partitionCols.map(_.toLowerCase).contains(lc_col))
	})

	// take note of measure columns (that have a numeric type)
	val measureColumns: Seq[String] = observationColumns.filter(col => df.schema
	                                                                     .find(_.name.toLowerCase == col.toLowerCase)
	                                                                     .get.dataType.isInstanceOf[NumericType])

	/** Helper Constructors */

	/**
	 * @constructor Create a new instance of [[SeqDF]]
	 * @param sourceDF the source [[DataFrame]]]
	 * @param sequencingCols sequence columns that give the total ordering, in order of decreasing precedence
	 * @param partitionCols columns used to distinguish different sequences from one another
	 */
	def this(sourceDF: DataFrame, sequencingCols: Seq[String], partitionCols: Seq[String]) =
		this( sourceDF,
		      SeqDF.GENERATED_TOTAL_ORDERING_COL,
		      sequencingCols,
		      partitionCols)

	/**
	 * @constructor Create a new instance of [[SeqDF]]
	 * @param sourceDF the source [[DataFrame]]
	 * @param sequenceCol a sequence column used to define ordering within partitions
	 * @param partitionCols columns used to distingusih different sequence from one another
	 */
	def this(sourceDF: DataFrame, sequenceCol: String, partitionCols: Seq[String]) =
		this(sourceDF, Seq(sequenceCol), partitionCols)

	/**
	 * Use the given column names to define a combined total ordering over the SeqDF
	 * @param sequencingCols the columns, in order of decreasing precedence, over which to define a total ordering
	 * @return a new SeqDF instance, with total ordering defined by the given columns in decreasing precedence
	 */
	def orderedBy(sequencingCols: String*): SeqDF =
		new SeqDF(this.df, this.totalOrderingCol, sequencingCols, this.partitionCols)

	/**
	 * Add or modify partition columns
	 * @param partitionCols the names of columns used to partition the SeqDF
	 * @return a new SeqDF instance, partitioned according to the given columns
	 */
	def partitionedBy(partitionCols: String*): SeqDF =
		new SeqDF(this.df, this.totalOrderingCol, this.sequencingColumns, partitionCols)

	/** Window Builders */

	/**
	 * Construct a window for this [[SeqDF]]
	 * @param start start row for the window
	 * @param end end row for the window
	 * @return a window appropriate for applying functions to this [[SeqDF]]
	 */
	def rowsBetweenWindow(start: Long, end: Long): WindowSpec =
	{
		// make sure our bounds make sense
		assert(end > start)

		// order by total ordering column
		val w = Window.orderBy(totalOrderingCol)

		// partition if needed
		val baseWindow = if( this.isPartitioned )
			                 w.partitionBy(partitionCols.head, partitionCols.tail :_*)
		                 else
			                 w
		// apply rows between
		baseWindow.rowsBetween(start, end)
	}

	/**
	 * Construct a window for this [[SeqDF]]
	 * @param ordering the type of window ordering to use
	 * @param alignment the type of window alignment to build
	 * @param size the size (in number of sequential observations) of the window
	 * @param includesNow whether the window should include the current observation
	 * @return a window appropriate for calculating window operations across this [[SeqDF]]
	 */
	def window(ordering: Ordering, alignment: Alignment, size: Long, includesNow: Boolean = false): WindowSpec =
		ordering match {
			case RowBased => WindowHelpers.buildWindow(rowsBetweenWindow)(alignment,size,includesNow)
			case RangeBased =>
				throw new IllegalArgumentException(s"SeqDF does not suport Range Based DataFrames")
		}
}

object SeqDF
{
	/**
	 * The column types valid for a primary sequence column
	 */
	final val validSeqTypes = Seq[DataType](ByteType, ShortType, IntegerType, LongType)

	/**
	 * Default column name for generated composite sequences
	 */
	final val GENERATED_TOTAL_ORDERING_COL = "__TOTAL_ORDER"

	/**
	 * Helper to validate a column
	 * @param schema
	 * @param columnRole
	 * @param allowedTypes
	 * @param columnName
	 */
	private[tempo] def columnValidator(schema: StructType)(columnRole: String, allowedTypes: Seq[DataType])(columnName: String): Unit =
	{
		// make sure the column exists
		val col = schema.fields.find(_.name.toLowerCase == columnName.toLowerCase)
		assert( col.isDefined, s"DataFrame does not contain a $columnRole column named $columnName" )

		// validate column type
		if( allowedTypes.nonEmpty ) {
			val colType = col.get.dataType
			assert( allowedTypes.contains(colType),
			        s"Column $columnName of type $colType cannot be used for a $columnRole column!" )
		}
	}
}

/**
 * Transformer to convert a [[DataFrame]] to a [[SeqDF]]
 *
 * @constructor create a new instance of the [[ToSeqDF]] Transformer
 * @param uid unique ID for this transformer
 */
class ToSeqDF(override val uid: String)
	extends PipelineStage
{
	/**
	 * @constructor create a new instance of the [[ToSeqDF]] Transformer
	 */
	def this() = this(Identifiable.randomUID("ToSeqDF"))

	override def copy(extra: ParamMap): PipelineStage = defaultCopy(extra)

	/** Params */

	/** @group param */
	final val totalOrderingColName: Param[String] = new Param[String](this,
	                                                                  "totalOrderingColName",
	                                                                  "Name of the sequence index column")
	setDefault(totalOrderingColName, SeqDF.GENERATED_TOTAL_ORDERING_COL)

	/** @group param */
	final val sequencingColNames: StringArrayParam =
		new StringArrayParam(this,
		                     "sequencingColNames",
		                     "Names of columns that define the sequencing of the DataFrame in combination")

	/** @group param */
	final val partitionColNames: StringArrayParam =
		new StringArrayParam(this,
		                     "partitionColNames",
		                     "Names of the partitioning columns")

	/** @group getParam */
	final def getTotalOrderingColName: String = $(totalOrderingColName)

	/** @group getParam */
	final def getSequencingColNames: Array[String] = $(sequencingColNames)

	/** @group getParam */
	final def getPartitionColNames: Array[String] = $(partitionColNames)

	/** @group setParam */
	final def setTotalOrderingColName(value: String): ToSeqDF =
		set(totalOrderingColName, value)

	/** @group setParam */
	final def setSequencingColNames(values: Array[String]): ToSeqDF =
		set(sequencingColNames, values)

	/** @group setParam */
	final def setPartitionColNames(values: Array[String]): ToSeqDF =
		set(partitionColNames, values)

	/** Transform */

	/**
	 * Validate and transform the schema
 	 * @param schema schema of the input [[DataFrame]]
	 * @return a modified schema of the output, transformed SeqDF
	 */
	override def transformSchema(schema: StructType): StructType =
	{
		// get a column validator
		val columnValidator = SeqDF.columnValidator(schema)_

		// validate the sequence columns
		if( $(sequencingColNames).nonEmpty )
			$(sequencingColNames).foreach( columnValidator("sequence",Seq()) )
		else
			columnValidator("sequence",SeqDF.validSeqTypes)($(totalOrderingColName))

		// validate the partition columns
		if( $(partitionColNames).nonEmpty )
			$(partitionColNames).foreach(columnValidator("partition",Seq()))

		// add generated column if we're constructing one from combined sequence columnss
		if( $(sequencingColNames).nonEmpty )
			StructType( schema.fields :+ StructField(SeqDF.GENERATED_TOTAL_ORDERING_COL,LongType) )
		else // otherwise, we have not modified the schema
			schema
	}

	/**
	 * Transform the [[DataFrame]] into a [[SeqDF]]
	 * @param df the input [[DataFrame]]
	 * @return a [[SeqDF]] instance, representing the input [[DataFrame]]
	 */
	def transform(df: DataFrame): SeqDF =
		new SeqDF(df, $(totalOrderingColName), $(sequencingColNames), $(partitionColNames))
}
