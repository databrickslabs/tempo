package com.databrickslabs.tempo

import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.scalatest.FunSpec

class DeltaWriterTests
	extends FunSpec with SparkSessionTestWrapper
{
	val schema = buildSchema(Seq(("symbol", StringType),
	                             ("date", StringType),
	                             ("event_ts", StringType),
	                             ("trade_pr", DoubleType),
	                             ("trade_pr_2", DoubleType)))

	val expectedSchema = buildSchema(Seq(("symbol", StringType),
	                                     ("event_ts", StringType),
	                                     ("date", StringType),
	                                     ("trade_pr_2", DoubleType),
	                                     ("trade_pr", DoubleType)))

	lazy val df = buildTestDF(schema,
	                          Seq(Row("S1", "SAME_DT", "2020-08-01 00:00:10", 349.21, 10.0),
	                              Row("S1", "SAME_DT", "2020-08-01 00:00:11", 340.21, 9.0),
	                              Row("S1", "SAME_DT", "2020-08-01 00:01:12", 353.32, 8.0),
	                              Row("S1", "SAME_DT", "2020-08-01 00:01:13", 351.32, 7.0),
	                              Row("S1", "SAME_DT", "2020-08-01 00:01:14", 350.32, 6.0),
	                              Row("S1", "SAME_DT", "2020-09-01 00:01:12", 361.1, 5.0),
	                              Row("S1", "SAME_DT", "2020-09-01 00:19:12", 362.1, 4.0)),
	                          "event_ts")

	lazy val dfExpected = buildTestDF(expectedSchema,
	                                  Seq(Row("S1", "2020-08-01 00:00:00", "SAME_DT", 10.0, 349.21),
	                                      Row("S1", "2020-08-01 00:01:00", "SAME_DT", 8.0, 353.32),
	                                      Row("S1", "2020-09-01 00:01:00", "SAME_DT", 5.0, 361.1),
	                                      Row("S1", "2020-09-01 00:19:00", "SAME_DT", 4.0, 362.1)),
	                                  "event_ts")

	it("Delta writer test") {

		// convert to TSDF
		val tsdf_left = TSDF(df, tsColumnName = "event_ts", partitionColumnNames = Seq("symbol"))

		import scala.reflect.io.Directory
		import java.io.File

		val spark_warehouse_dir = spark.conf.get("spark.sql.warehouse.dir")
		val directory = new Directory(new File(spark_warehouse_dir + "my_table/"))
		directory.deleteRecursively()

		tsdf_left.write("my_table")
		println("delta table count" + spark.table("my_table").count())

		// should be equal to the expected dataframe
		assert(spark.table("my_table").count() == 7)
		// self.assertDataFramesEqual(tsdf_left.df, tsdf_left.df)
	}
}
