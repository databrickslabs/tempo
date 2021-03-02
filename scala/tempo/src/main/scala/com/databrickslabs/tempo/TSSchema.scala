package com.databrickslabs.tempo

import org.apache.spark.sql.types.{NumericType, StructField, StructType}

/**
 * A Timeseries Schema
 */
trait TSSchema
{
	/**
	 * The timeseries column
	 */
	val tsColumn: StructField

	/**
	 * The timeseries column name
	 */
	val tsColumnName: String

	/**
	 * Columns that partition this timeseries
	 * (they separate one series from another)
	 */
	val partitionCols: Seq[StructField]

	/**
	 * Indicates whether or not this timeseries is partitioned
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
}

/**
 * Timeseries Schema
 * @param fields
 * @param tsColumn
 * @param partitionCols
 */
class TSStructType(fields: Array[StructField],
                   val tsColumn: StructField,
                   val partitionCols: StructField*)
	extends StructType(fields) with TSSchema
{
	// Validate the arguments
	assert( fields.contains(tsColumn),
	        s"The given DataFrame does not contain the given timeseries column ${tsColumn.name}!")
	assert( TSDF.validTSColumnTypes.contains(tsColumn.dataType),
	        s"The given timeseries column ${tsColumn.name} is of a type (${tsColumn.dataType}) that cannot be used as an index!")

	partitionCols.foreach( pCol => assert( fields.contains(pCol),
	                                       s"The given DataFrame does not contain the given partition column ${pCol.name}!" ))

	// Define some other values

	/**
	 * The timeseries column name
	 */
	override val tsColumnName: String = tsColumn.name

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
		fields.filter(!structuralColumns.contains(_))

	/**
	 * Measure columns (numeric observations)
	 */
	val measureColumns: Seq[StructField] =
		observationColumns.filter(_.dataType.isInstanceOf[NumericType])
}
