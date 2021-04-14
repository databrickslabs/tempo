package com.databrickslabs.tempo

import com.databrickslabs.tempo.EMA._
import com.databrickslabs.tempo.asofJoin._
import com.databrickslabs.tempo.rangeStats._
import com.databrickslabs.tempo.resample._
import org.apache.spark.sql.expressions.{Window, WindowSpec}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Column, DataFrame}

/**
 * The main abstraction of the tempo project is the time series data frame (abbreviated TSDF) which contains methods to transform the existing Spark data frame based on partitions columns and an event timestamp column. Additional methods are present for joining 2 TSDFs
 */
sealed trait TSDF
	extends TSSchema
{
	// core backing values

	/**
	 * The [[DataFrame]] backing this [[TSDF]]
	 */
	val df: DataFrame

	/**
	 * The schema of the timeseries DataFrame
	 */
	val schema: TSStructType

	// transformation functions

	def asofJoin(rightTSDF: TSDF,
		leftPrefix: String,
		rightPrefix: String = "right_",
							 maxLookback : Int = 0,
		tsPartitionVal: Int = 0,
		fraction: Double = 0.1) : TSDF

	def vwap(frequency : String = "m", volume_col : String = "volume", price_col : String = "price") : TSDF

	def rangeStats(colsToSummarise: Seq[String] = Seq(), rangeBackWindowSecs: Int = 1000): TSDF

	def EMA(colName: String, window: Int, exp_factor: Double = 0.2): TSDF

	def resample(freq : String, func : String) : TSDF

	def write(tabName: String, tabPath : String = "", optimizationCols: Option[Seq[String]] = None) : Unit

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
	 * Construct a base window for this [[TSDF]]
	 * @return a base [[WindowSpec]] from which other windows can be constructed
	 */
	def baseWindow(): WindowSpec

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
private[tempo] sealed class BaseTSDF(val df: DataFrame, val schema: TSStructType )
	extends TSDF
{
	/**
	 * @constructor
	 * @param df
	 * @param tsColumn
	 * @param partitionCols
	 */
	def this(df: DataFrame, tsColumn: StructField, partitionCols: StructField* ) =
		this(df,new TSStructType(df.schema.fields, tsColumn, partitionCols :_*))

	/**
	 * The timeseries column
	 */
	val tsColumn: StructField = schema.tsColumn

	/**
	 * The timeseries column name
	 */
	val tsColumnName: String = schema.tsColumnName

	/**
	 * Columns that partition this timeseries
	 * (they separate one series from another)
	 */
	val partitionCols: Seq[StructField] = schema.partitionCols

	/**
	 * Indicates whether or not this timeseries is partitioned
	 */
	val isPartitioned: Boolean = schema.isPartitioned

	/**
	 * Columns that define the structure of the [[TSDF]].
	 * These should be protected from arbitrary modification.
	 */
	val structuralColumns: Seq[StructField] = schema.structuralColumns

	/**
	 * Observation columns
	 */
	val observationColumns: Seq[StructField] = schema.observationColumns

	/**
	 * Measure columns (numeric observations)
	 */
	val measureColumns: Seq[StructField] = schema.measureColumns

	// Transformations

	/**
	 * Add or modify partition columns
	 *
	 * @param partitionCols the names of columns used to partition the SeqDF
	 * @return a new SeqDF instance, partitioned according to the given columns
	 */
	def partitionedBy(partitionCols: String*): TSDF =
		TSDF(this.df, this.tsColumnName, partitionCols :_*)

	def withColumn(colName: String, col: Column): TSDF = {
		TSDF(this.df.withColumn(colName, col),
		     tsColumnName = this.tsColumnName,
		     partitionColumnNames = this.partitionCols.map(_.name).mkString)
	}

	def select(cols: Column*): TSDF = {

		val colsList = cols.toList

		val timeAndPartitionsPresent = colsList.contains(tsColumnName) && partitionCols.map(x => x.name).toList.forall(colsList.contains)

		//if (!timeAndPartitionsPresent) {throw new RuntimeException("Timestamp column or partition columns are missing")}

		TSDF(df.select(cols:_*), tsColumnName = tsColumnName, partitionColumnNames = partitionCols.map(_.name).mkString)
	}

	def select(col: String,
	                    cols: String*): TSDF = {
		val masterList = Seq(col) ++ cols.toSeq

		val colsList = masterList.toList

		val timeAndPartitionsPresent = colsList.contains(tsColumnName) && partitionCols.map(x => x.name).toList.forall(colsList.contains)

		//if (!timeAndPartitionsPresent) {throw new RuntimeException("Timestamp column or partition columns are missing")}

		TSDF(df.select(masterList.head, masterList.tail:_*), tsColumnName = tsColumnName, partitionColumnNames =  partitionCols.map(_.name).mkString)
	}

	def selectExpr(exprs: String*): TSDF = {
		TSDF(df.selectExpr(exprs:_*), tsColumnName = tsColumnName, partitionColumnNames =  partitionCols.map(_.name).mkString)
	}

	def filter(condition: Column): TSDF = {
		TSDF(df.filter(condition), tsColumnName = tsColumnName, partitionColumnNames = partitionCols.map(_.name).mkString)
	}

	def filter(conditionExpr: String): TSDF = {
		TSDF(df.filter(conditionExpr), tsColumnName = tsColumnName, partitionColumnNames = partitionCols.map(_.name).mkString)
	}

	def where(condition: Column): TSDF = {
		TSDF(df.where(condition), tsColumnName = tsColumnName, partitionColumnNames = partitionCols.map(_.name).mkString)
	}

	def where(conditionExpr: String): TSDF = {
		TSDF(df.where(conditionExpr), tsColumnName = tsColumnName, partitionColumnNames = partitionCols.map(_.name).mkString)
	}

	def limit(n: Int): TSDF = {
		TSDF(df.limit(n), tsColumnName = tsColumnName, partitionColumnNames = partitionCols.map(_.name).mkString)
	}

	def union(other: TSDF): TSDF = {
		TSDF(df.union(other.df), tsColumnName = tsColumnName, partitionColumnNames = partitionCols.map(_.name).mkString)
	}

	def unionAll(other: TSDF): TSDF = {
		TSDF(df.unionAll(other.df), tsColumnName = tsColumnName, partitionColumnNames = partitionCols.map(_.name).mkString)
	}

	def withColumnRenamed(existingName: String,
	                               newName: String): TSDF = {
		TSDF(df.withColumnRenamed(existingName, newName), tsColumnName = tsColumnName, partitionColumnNames = partitionCols.map(_.name).mkString)
	}

	def drop(colName: String): TSDF = {
		TSDF(df.drop(colName), tsColumnName = tsColumnName, partitionColumnNames = partitionCols.map(_.name).mkString)
	}

	def drop(colNames: String*): TSDF = {
		TSDF(df.drop(colNames:_*), tsColumnName = tsColumnName, partitionColumnNames = partitionCols.map(_.name).mkString)
	}

	def drop(col: Column): TSDF = {
		TSDF(df.drop(col), tsColumnName = tsColumnName, partitionColumnNames = partitionCols.map(_.name).mkString)
	}

	// Window builder functions

	/**
	 * Construct a base window for this [[TSDF]]
	 *
	 * @return a base [[WindowSpec]] from which other windows can be constructed
	 */
	def baseWindow(): WindowSpec =
	{
		// order by total ordering column
		val w = Window.orderBy(tsColumnName)

		// partition if needed
		if( this.schema.isPartitioned )
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
		maxLookback : Int = 0,
		tsPartitionVal: Int = 0,
		fraction: Double = 0.1): TSDF = {

		if (tsPartitionVal > 0) {println("WARNING: You are using the skew version of the AS OF join. This may result in null values if there are any values outside of the maximum lookback. For maximum efficiency, choose smaller values of maximum lookback, trading off performance and potential blank AS OF values for sparse keys")}

		if (leftPrefix == "" && tsPartitionVal == 0) {
			asofJoinExec(this,rightTSDF, leftPrefix = None, rightPrefix, maxLookback,  tsPartitionVal = None, fraction)
		}
		else if(leftPrefix == "") {
			asofJoinExec(this, rightTSDF, leftPrefix = None, rightPrefix, maxLookback, Some(tsPartitionVal), fraction)
		}
		else if(tsPartitionVal == 0) {
			asofJoinExec(this, rightTSDF, Some(leftPrefix), rightPrefix, maxLookback, tsPartitionVal = None)
		}
		else {
			asofJoinExec(this, rightTSDF, Some(leftPrefix), rightPrefix, maxLookback, Some(tsPartitionVal), fraction)
		}
	}

	def vwap(frequency : String = "m", volume_col : String = "volume", price_col : String = "price") : TSDF = {
		// set pre_vwap as self or enrich with the frequency
		var pre_vwap = this.df

		if (frequency == "m") {
			pre_vwap = pre_vwap.withColumn("time_group", date_trunc("minute", col(tsColumn.name)))
		} else if (frequency == "H") {
			pre_vwap = pre_vwap.withColumn("time_group", date_trunc("hour", col(tsColumn.name)))
		} else if (frequency == "D") {
			pre_vwap = pre_vwap.withColumn("time_group", date_trunc("day", col(tsColumn.name)))
		}

		var group_cols = List("time_group")

		if (this.partitionCols.size > 0) {
			group_cols = group_cols ++ (this.partitionCols.map(x => x.name))
		}
		var vwapped = ( pre_vwap.withColumn("dllr_value", col(price_col) * col(volume_col)).groupBy(group_cols map col: _*).agg( sum("dllr_value").alias("dllr_value"), sum(volume_col).alias(volume_col), max(price_col).alias("max_" + price_col))
		.withColumn("vwap", col("dllr_value") / col(volume_col)) )

		vwapped = vwapped.withColumnRenamed("time_group", "event_ts")

		TSDF( vwapped, tsColumn.name, partitionColumnNames = partitionCols.map(_.name).mkString )
	}
	//
	def rangeStats(colsToSummarise: Seq[String] = Seq(), rangeBackWindowSecs: Int = 1000): TSDF = {
		rangeStatsExec(this, colsToSummarise = colsToSummarise, rangeBackWindowSecs = rangeBackWindowSecs)
	}

	def EMA(colName: String, window: Int, exp_factor: Double = 0.2): TSDF = {
		emaExec(this, colName, window, exp_factor)
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
		val double_ts_col = this.tsColumnName + "_dbl"

		val this_df = this.df.withColumn(double_ts_col, col(this.tsColumnName).cast("double"))

		// summary missing value percentages
		val missing_vals_pct_cols = List(lit("missing_vals_pct").alias("summary")) ++ (for {c <- this_df.dtypes if (c._2 != "TimestampType")} yield (lit(100) * count(when(col(c._1).isNull, c._1)) / count(lit(1))).alias(c._1)).toList

		val missing_vals = this_df.select(missing_vals_pct_cols:_*)

		// describe stats
		var desc_stats = this_df.describe().union(missing_vals)
		val unique_ts = this_df.select(this.partitionCols.map(x => x.name) map col: _*).distinct().count

		val max_ts = this_df.select(max(col(this.tsColumnName)).alias("max_ts")).collect()(0)(0).toString
		val min_ts = this_df.select(min(col(this.tsColumnName)).alias("max_ts")).collect()(0)(0).toString
		val gran = this_df.selectExpr(s"""min(case when $double_ts_col - cast($double_ts_col as integer) > 0 then '1-millis'
                  when $double_ts_col % 60 != 0 then '2-seconds'
                  when $double_ts_col % 3600 != 0 then '3-minutes'
                  when $double_ts_col % 86400 != 0 then '4-hours'
                  else '5-days' end) granularity""").collect()(0)(0).toString.substring(2)

		val non_summary_cols = for {c <- desc_stats.columns if c != "summary"} yield col(c)
		val non_summary_col_strs = for {c <- desc_stats.columns if c != "summary"} yield c

		val preCols = List(col("summary"), lit(" ").alias("unique_ts_count"), lit(" ").alias("min_ts"),
			lit(" ").alias("max_ts"), lit(" ").alias("granularity")) ++ (non_summary_cols.toList)
		desc_stats = desc_stats.select(preCols:_*)

		val non_summary_cols_blank = List(lit("global").alias("summary"),lit(unique_ts).alias("unique_ts_count"), lit(min_ts).cast("timestamp").alias("min_ts"), lit(max_ts).cast("timestamp").alias("max_ts"), lit(gran).alias("granularity")) ++ (for {c <- non_summary_col_strs} yield lit(" ").alias(c)).toList
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

	def write(tabName: String, tabPath : String = "", optimizationCols: Option[Seq[String]] = None) : Unit = {
       TSDFWriters.write(this, tabName, tabPath, optimizationCols)
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
			return TSDF(lookback_tsdf.where(size(col(featureColName)) === lookbackWindowSize), tsColumnName = tsColumnName, partitionColumnNames = pCols)
		}

		TSDF( lookback_tsdf, tsColumnName = tsColumnName, partitionColumnNames = pCols )

	}
}

/**
 * [[TSDF]] companion object - contains static constructors
 */
object TSDF {
	// Utility functions

	/**
		* @define DEFAULT_SEQ_COLNAME sequence_num
		*/
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
		*
		* @param df      the [[DataFrame]] with the column
		* @param colName the name of the column
		* @return the named column of the [[DataFrame]], if it exists,
		*         otherwise a [[NoSuchElementException]] is thrown
		*/
	private[tempo] def colByName(df: DataFrame)(colName: String): StructField = {
		df.schema.find(_.name.toLowerCase() == colName.toLowerCase()).get
	}

	// TSDF Constructors

	/**
		* @param df                   the source [[DataFrame]]
		* @param tsColumnName         the name of the timeseries column
		* @param partitionColumnNames the names of the paritioning columns
		* @return a [[TSDF]] object representing the given [[DataFrame]]
		*/
	def apply(df: DataFrame,
						tsColumnName: String,
						partitionColumnNames: String*): TSDF = {
		val colFinder = colByName(df) _
		val tsColumn = colFinder(tsColumnName)
		val partitionCols = partitionColumnNames.map(colFinder)

		new BaseTSDF(df, tsColumn, partitionCols: _*)
	}

	/**
		* Construct a [[TSDF]] with multi-column ordering. This method will construct
		* a timeseries column (in the column named by the sequenceColName) based on the
		* ordering of the orderingColumns.
		*
		* @param df              the source [[DataFrame]]
		* @param orderingColumns columns that define ordering on the [[DataFrame]]
		* @param sequenceColName the name for the constructed timeseries column
		* @param partitionCols   partition columns
		* @return a [[TSDF]] object representing the given [[DataFrame]]
		*/
	def apply(df: DataFrame,
						orderingColumns: Seq[String],
						sequenceColName: String,
						partitionCols: Seq[String]): TSDF = {
		// define window for ordering according to the combined sequence columns
		val seq_win =
			if (partitionCols.nonEmpty)
				Window.partitionBy(partitionCols.map(df(_)): _*)
					.orderBy(orderingColumns.map(df(_)): _*)
			else
				Window.orderBy(orderingColumns.map(df(_)): _*)

		// add total ordering column
		val newDF = df.withColumn(sequenceColName, row_number().cast(LongType).over(seq_win))

		// construct our TSDF
		apply(newDF, sequenceColName, partitionCols: _*)
	}

	/**
		* Construct a [[TSDF]] with multi-column ordering. This method will construct
		* a timeseries column named $DEFAULT_SEQ_COLNAME based on the ordering of the
		* orderingColumns.
		*
		* @param df              the source [[DataFrame]]
		* @param orderingColumns columns that define ordering on the [[DataFrame]]
		* @param partitionCols   partition columns
		* @return a [[TSDF]] object representing the given [[DataFrame]]
		*/
	def apply(df: DataFrame,
						orderingColumns: Seq[String],
						partitionCols: Seq[String]): TSDF =
		apply(df, orderingColumns, DEFAULT_SEQ_COLNAME, partitionCols)

	/**
		* Implicit TSDF functions on DataFrames
		*
		* @param df the DataFrame on which to define the implicit functions
		*/
	implicit class TSDFImplicits(df: DataFrame) {
		/**
			* Convert a [[DataFrame]] to [[TSDF]]
			*
			* @param tsColumnName         timeseries column
			* @param partitionColumnNames partition columns
			* @return a [[TSDF]] based on the current [[DataFrame]]
			*/
		def toTSDF(tsColumnName: String,
							 partitionColumnNames: String*): TSDF =
			TSDF(df, tsColumnName, partitionColumnNames: _*)

		/**
			* Convert a [[DataFrame]] to [[TSDF]]
			*
			* @param orderingColumns columns that define ordering on the [[DataFrame]]
			* @param sequenceColName the name for the constructed timeseries column
			* @param partitionCols   partition columns
			* @return a [[TSDF]] based on the current [[DataFrame]]
			*/
		def toTSDF(orderingColumns: Seq[String],
							 sequenceColName: String,
							 partitionCols: Seq[String]): TSDF =
			TSDF(df, orderingColumns, sequenceColName, partitionCols)

		/**
			* Convert a [[DataFrame]] to [[TSDF]]
			*
			* @param orderingColumns columns that define ordering on the [[DataFrame]]
			* @param partitionCols   parition columns
			* @return a [[TSDF]] based on the current [[DataFrame]]
			*/
		def toTSDF(orderingColumns: Seq[String],
							 partitionCols: Seq[String]): TSDF =
			TSDF(df, orderingColumns, partitionCols)
	}
}