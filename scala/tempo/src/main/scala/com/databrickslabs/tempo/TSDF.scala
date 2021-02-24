package com.databrickslabs.tempo

import java.io.FileNotFoundException

import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.sql.expressions.{Window, WindowSpec}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import resample._
import asofJoin._

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
	val partitionCols: Seq[StructField]

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

	def asofJoin(rightTSDF: TSDF,
		leftPrefix: String,
		rightPrefix: String = "right_",
		tsPartitionVal: Int = 0,
		fraction: Double = 0.1) : TSDF

	def resample(freq : String, func : String) : TSDF

	def withLookbackFeatures(featureCols : List[String], lookbackWindowSize : Integer, exactSize : Boolean = true, featureColName : String = "features") : TSDF

	def describe() : DataFrame

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
 * @param partitionCols partitioning columns
 */
private[tempo] sealed class BaseTSDF(val df: DataFrame,
                                     val tsColumn: StructField,
                                     val partitionCols: StructField* )
	extends TSDF
{
	// Validate the arguments
	assert( df.schema.contains(tsColumn),
	        s"The given DataFrame does not contain the given timeseries column ${tsColumn.name}!")
	assert( TSDF.validTSColumnTypes.contains(tsColumn.dataType),
	        s"The given timeseries column ${tsColumn.name} is of a type (${tsColumn.dataType}) that cannot be used as an index!")

	partitionCols.foreach( pCol => assert( df.schema.contains(pCol),
	                                          s"The given DataFrame does not contain the given partition column ${pCol.name}!" ))

	// Define some other values

	/**
	 * Is this [[TSDF]] partitioned?
	 */
	val isPartitioned: Boolean = partitionCols.nonEmpty

	/**
	 * Columns that define the structure of the [[TSDF]].
	 * These should be protected from arbitrary modification.
	 */
	val structuralColumns: Seq[StructField] = Seq(tsColumn) ++ partitionCols

	/**
	 * Observation columns
	 */
	val observationColumns: Seq[StructField] =
		df.schema.filter(!structuralColumns.contains(_))

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
			w.partitionBy(partitionCols.head.name, partitionCols.tail.map(_.name) :_*)
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

	// asof Join functions
	// TODO: probably rewrite, but overloading methods seemed to break. the ifElse stuff is a quick fix.
  def asofJoin(
		rightTSDF: TSDF,
		leftPrefix: String = "",
		rightPrefix: String = "right_",
		tsPartitionVal: Int = 0,
		fraction: Double = 0.1): TSDF = {

		if (leftPrefix == "" && tsPartitionVal == 0) {
			asofJoinExec(this,rightTSDF, leftPrefix = None, rightPrefix, tsPartitionVal = None, fraction)
		}
		else if(leftPrefix == "") {
			asofJoinExec(this, rightTSDF, leftPrefix = None, rightPrefix, Some(tsPartitionVal), fraction)
		}
		else if(tsPartitionVal == 0) {
			asofJoinExec(this, rightTSDF, Some(leftPrefix), rightPrefix, tsPartitionVal = None)
		}
		else {
			asofJoinExec(this, rightTSDF, Some(leftPrefix), rightPrefix, Some(tsPartitionVal), fraction)
		}
	}

	/**
		* Describe a TSDF object using a global summary across all time series (anywhere from 10 to millions) as well as the standard Spark data frame stats. Missing vals
		* Summary
		* global - unique time series based on partition columns, min/max times, granularity - lowest precision in the time series timestamp column
		* count / mean / stddev / min / max - standard Spark data frame describe() output
		* missing_vals_pct - percentage (from 0 to 100) of missing values.
		*
		* @return
		*/
	def describe() : DataFrame = {

		// extract the double version of the timestamp column to summarize
		val double_ts_col = this.tsColumn.name + "_dbl"

		val this_df = this.df.withColumn(double_ts_col, col(this.tsColumn.name).cast("double"))

		// summary missing value percentages
		val missing_vals_pct_cols = List(lit("missing_vals_pct").alias("summary")) ++ (for {c <- this_df.dtypes if (c._2 != "timestamp")} yield (lit(100) * count(when(col(c._1).isNull(), c._1)) / count(lit(1))).alias(c._1)).toList

		val missing_vals = this_df.select(missing_vals_pct_cols:_*)


		// describe stats
		var desc_stats = this_df.describe().union(missing_vals)
		val unique_ts = this_df.select(this.partitionCols.map(x => x.name) map col: _*).distinct().count

		val max_ts = this_df.select(max(col(this.tsColumn.name)).alias("max_ts")).collect()(0)
		val min_ts = this_df.select(min(col(this.tsColumn.name)).alias("max_ts")).collect()(0)
		val gran = this_df.selectExpr(s"""min(case when $double_ts_col - cast($double_ts_col as integer) > 0 then '1-millis'
                  when $double_ts_col % 60 != 0 then '2-seconds'
                  when $double_ts_col % 3600 != 0 then '3-minutes'
                  when $double_ts_col % 86400 != 0 then '4-hours'
                  else '5-days' end) granularity""").collect()(0)(0).toString.substring(2)

		val non_summary_cols = for {c <- desc_stats.columns if c != "summary"} yield col(c)
		val non_summary_col_strs = for {c <- desc_stats.columns if c != "summary"} yield c

		val preCols = List(col("summary"), lit(None).alias("unique_ts_count"), lit(None).alias("min_ts"),
			lit(None).alias("max_ts"), lit(None).alias("granularity")) ++ (non_summary_cols.toList)
		desc_stats = desc_stats.select(preCols:_*)

		val non_summary_cols_blank = List(lit("global").alias("summary"),lit(unique_ts).alias("unique_ts_count"), lit(min_ts).alias("min_ts"), lit(max_ts).alias("max_ts"), lit(gran).alias("granularity")) ++ (for {c <- non_summary_col_strs} yield lit(None).alias(c)).toList
		// add in single record with global summary attributes and the previously computed missing value and Spark data frame describe stats
		val global_smry_rec = desc_stats.limit(1).select(non_summary_cols_blank: _*)

		val full_smry = global_smry_rec.union(desc_stats)

    return(full_smry)
	}


	/**
		*
		* @param freq - frequency for upsample - valid inputs are "hr", "min", "sec" corresponding to hour, minute, or second
		* @param func - function used to aggregate input
		* @return - TSDF object with sample data using aggregate function
		*/
	def resample(freq : String, func : String) : TSDF = {
	validateFuncExists(func)
	val enriched_tsdf = rs_agg(this, freq, func)
	return(enriched_tsdf)
	}
	/**
		* Creates a 2-D feature tensor suitable for training an ML model to predict current values from the history of
		* some set of features. This function creates a new column containing, for each observation, a 2-D array of the values
		* of some number of other columns over a trailing "lookback" window from the previous observation up to some maximum
		* number of past observations.
		* :param featureCols: the names of one or more feature columns to be aggregated into the feature column
		* :param lookbackWindowSize: The size of lookback window (in terms of past observations). Must be an integer >= 1
		* :param exactSize: If True (the default), then the resulting DataFrame will only include observations where the
		* generated feature column contains arrays of length lookbackWindowSize. This implies that it will truncate
		* observations that occurred less than lookbackWindowSize from the start of the timeseries. If False, no truncation
		* occurs, and the column may contain arrays less than lookbackWindowSize in length.
		* :param featureColName: The name of the feature column to be generated. Defaults to "features"
		* :return: a DataFrame with a feature column named featureColName containing the lookback feature tensor
		*
		* @param featureCols
		* @param lookbackWindowSize
		* @param exactSize
		* @param featureColName
		* @return
		*/
	def withLookbackFeatures(featureCols : List[String], lookbackWindowSize : Integer, exactSize : Boolean = true, featureColName : String = "features") : TSDF = {
		// first, join all featureCols into a single array column
		val tempArrayColName = "__TempArrayCol"
		val feat_array_tsdf = df.withColumn(tempArrayColName, lit(featureCols).cast("array"))

		// construct a lookback array
		val lookback_win = windowBetweenRows(-lookbackWindowSize, -1)
		val lookback_tsdf = (feat_array_tsdf.withColumn(featureColName,
			collect_list(col(tempArrayColName)).over(lookback_win))
			.drop(tempArrayColName))

		val pCols = partitionCols.map(_.name).mkString


		// make sure only windows of exact size are allowed
		if (exactSize) {
			return TSDF(lookback_tsdf.where(size(col(featureColName)) === lookbackWindowSize), tsColumnName = tsColumn.name, partitionColumnNames = pCols)
		}

		TSDF( lookback_tsdf, tsColumnName = tsColumn.name, partitionColumnNames = pCols )

	}
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
		val partitionCols = partitionColumnNames.map(colFinder)

		new BaseTSDF(df, tsColumn, partitionCols :_*)
	}

	/**
	 * @constructor
	 * @param df
	 * @param orderingColumns
	 * @param sequenceColName
	 * @param partitionCols
	 */
	def apply( df: DataFrame,
	           orderingColumns: Seq[String],
	           sequenceColName: String,
						 partitionCols: String* ): TSDF =
	{
		// define window for ordering according to the combined sequence columns
		val seq_win =
			if( partitionCols.nonEmpty)
				Window.partitionBy(partitionCols.map(df(_)) :_*)
				      .orderBy(orderingColumns.map(df(_)) :_*)
			else
				Window.orderBy(orderingColumns.map(df(_)) :_*)

		// add total ordering column
		val newDF = df.withColumn(sequenceColName, row_number().cast(LongType).over(seq_win))

		// construct our TSDF
		apply( newDF, sequenceColName, partitionCols :_* )
	}

	/**
	 * @constructor
	 * @param df
	 * @param orderingColumns
	 * @param partitionCols
	 * @return
	 */
	def apply(df: DataFrame,
	          orderingColumns: Seq[String],
						partitionCols: String* ): TSDF =
		apply( df, orderingColumns, DEFAULT_SEQ_COLNAME, partitionCols :_* )
}


object programExecute {
	def main(args: Array[String]): Unit = {

		println("Hello from main of class")
	}
}