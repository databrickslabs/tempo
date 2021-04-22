package com.databrickslabs.tempo

import org.apache.spark.sql.functions.{col, to_timestamp}
import org.apache.spark.sql.types.{DataType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

// create Spark session with local mode
trait SparkSessionTestWrapper
{
	lazy val spark: SparkSession = {
		SparkSession.builder()
		            .master("local")
		            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
		            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
		            .config("spark.sql.shuffle.partitions", 1)
		            .config("spark.driver.bindAddress", "127.0.0.1")
		            .appName("spark session")
		            .getOrCreate()
	}

	/**
	 *
	 * @param fields
	 * @return
	 */
	def buildSchema( fields: Seq[(String,DataType)] ): StructType =
		StructType( fields.map( f => StructField(f._1, f._2)) )

	/**
	 * Constructs a Spark Dataframe from the given components
	 * :param schema: the schema to use for the Dataframe
	 * :param data: values to use for the Dataframe
	 * :param ts_cols: list of column names to be converted to Timestamp values
	 * :return: a Spark Dataframe, constructed from the given schema and values
	 */
	def buildTestDF(schema : StructType, data : Seq[Row], ts_cols : String*) : DataFrame =
	{
		// build dataframe
		var df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)

		// convert all timestamp columns
		for (tsc <- ts_cols) {
			df = df.withColumn(tsc, to_timestamp(col(tsc)))
		}
		return df
	}

}
