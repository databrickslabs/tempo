package com.databrickslabs.tempo

import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.{col, max}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.scalatest.FunSpec

class DescribeTests
	extends FunSpec with SparkSessionTestWrapper
{
	val leftSchema = buildSchema(List(("symbol", StringType),
	                                  ("event_ts", StringType),
	                                  ("trade_pr", DoubleType)))

	val expectedSchema = buildSchema(List(("symbol", StringType),
	                                      ("left_event_ts", StringType),
	                                      ("left_trade_pr", DoubleType),
	                                      ("right_bid_pr", DoubleType),
	                                      ("right_ask_pr", DoubleType),
	                                      ("right_event_ts", StringType)))

	lazy val dfLeft = buildTestDF( leftSchema,
	                               Seq(Row("S1", "2020-08-01 00:00:10", 349.21),
	                                   Row("S1", "2020-08-01 00:01:12", 351.32),
	                                   Row("S1", "2020-09-01 00:02:10", 361.1),
	                                   Row("S1", "2020-09-01 00:19:12", 362.1)),
	                               "event_ts")

	lazy val dfExpected = buildTestDF( expectedSchema,
	                                   Seq(Row("S1", "2020-08-01 00:00:10", 349.21, "2020-08-01 00:00:01", 345.11, 351.12),
	                                       Row("S1", "2020-08-01 00:01:12", 351.32, "2020-08-01 00:01:05", 348.10, 353.13),
	                                       Row("S1", "2020-09-01 00:02:10", 361.1, "2020-09-01 00:02:01", 358.93, 365.12),
	                                       Row("S1", "2020-09-01 00:19:12", 362.1, "2020-09-01 00:15:01", 359.21, 365.31)),
	                                   "left_event_ts", "right_event_ts")

	it("TSDF Describe Test") {

		// perform the join
		val tsdf_left = TSDF(dfLeft, tsColumnName = "event_ts", partitionColumnNames = Seq("symbol"))
		val res = tsdf_left.describe()

		// joined dataframe should equal the expected dataframe
		// self.assertDataFramesEqual(res, dfExpected)
		assert(res.count == 7)
		assert(res.select(max(col("unique_time_series_count"))).collect()(0)(0) == "1")
		assert(res.select(col("min_ts").cast("string")).collect()(0)(0) == "2020-08-01 00:00:10")
		assert(res.select(col("max_ts").cast("string")).collect()(0)(0) == "2020-09-01 00:19:12")
	}
}
