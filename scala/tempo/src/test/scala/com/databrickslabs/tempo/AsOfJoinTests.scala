package com.databrickslabs.tempo

import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.scalatest.FunSpec

class AsOfJoinTests
	extends FunSpec with SparkSessionTestWrapper
{
	val leftSchema = buildSchema(List(("symbol", StringType),
	                                  ("event_ts", StringType),
	                                  ("trade_pr", DoubleType)))

	val rightSchema = buildSchema(List(("symbol", StringType),
	                                   ("event_ts", StringType),
	                                   ("bid_pr", DoubleType),
	                                   ("ask_pr", DoubleType)))

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
	                               "event_ts" )

	lazy val dfLeft2 = buildTestDF( leftSchema,
	                                Seq(Row("S1", "2020-08-01 00:00:02", 349.21),
	                                    Row("S1", "2020-08-01 00:00:08", 351.32),
	                                    Row("S1", "2020-08-01 00:00:11", 361.12),
	                                    Row("S1", "2020-08-01 00:00:18", 364.31),
	                                    Row("S1", "2020-08-01 00:00:19", 362.94),
	                                    Row("S1", "2020-08-01 00:00:21", 364.27),
	                                    Row("S1", "2020-08-01 00:00:23", 367.36)),
	                                "event_ts" )

	lazy val dfRight = buildTestDF( rightSchema,
	                                Seq(Row("S1", "2020-08-01 00:00:01", 345.11, 351.12),
	                                    Row("S1", "2020-08-01 00:01:05", 348.10, 353.13),
	                                    Row("S1", "2020-09-01 00:02:01", 358.93, 365.12),
	                                    Row("S1", "2020-09-01 00:15:01", 359.21, 365.31)),
	                                "event_ts" )

	lazy val dfRight2 = buildTestDF( rightSchema,
		Seq(Row("S1", "2020-08-01 00:00:01", 345.11, 351.12),
			Row("S1", "2020-08-01 00:00:09", 348.10, 353.13),
			Row("S1", "2020-08-01 00:00:12", 358.93, 365.12),
			Row("S1", "2020-08-01 00:00:19", 359.21, 365.31)),
		"event_ts" )

	lazy val dfExpected1 = buildTestDF( expectedSchema,
	                                        Seq(Row("S1", "2020-08-01 00:00:10", 349.21, 345.11, 351.12, "2020-08-01 00:00:01"),
	                                            Row("S1", "2020-08-01 00:01:12", 351.32, 348.1, 353.13, "2020-08-01 00:01:05"),
	                                            Row("S1", "2020-09-01 00:02:10", 361.1, 358.93, 365.12, "2020-09-01 00:02:01"),
	                                            Row("S1", "2020-09-01 00:19:12", 362.1, 359.21, 365.31, "2020-09-01 00:15:01")),
	                                        "left_event_ts", "right_event_ts" )

	lazy val dfExpected2 = buildTestDF( expectedSchema,
	                                    Seq(Row("S1", "2020-08-01 00:00:02", 349.21, 345.11, 351.12, "2020-08-01 00:00:01"),
	                                        Row("S1", "2020-08-01 00:00:08", 351.32, 345.11, 351.12, "2020-08-01 00:00:01"),
	                                        Row("S1", "2020-08-01 00:00:11", 361.12, 348.1, 353.13, "2020-08-01 00:00:09"),
	                                        Row("S1", "2020-08-01 00:00:18", 364.31, 358.93, 365.12, "2020-08-01 00:00:12"),
	                                        Row("S1", "2020-08-01 00:00:19", 362.94, 359.21, 365.31, "2020-08-01 00:00:19"),
	                                        Row("S1", "2020-08-01 00:00:21", 364.27, 359.21, 365.31, "2020-08-01 00:00:19"),
	                                        Row("S1", "2020-08-01 00:00:23", 367.36, 359.21, 365.31, "2020-08-01 00:00:19")),
	                                    "left_event_ts", "right_event_ts" )

	it("Standard AS OF Join Test") {

		// set up TSDFs
		val tsdf_left = TSDF(dfLeft.withColumn("dummy", lit(10)),
		                     "event_ts",
		                     Seq("symbol", "dummy"))
		val tsdf_right = TSDF(dfRight.withColumn("dummy", lit(10)),
		                      "event_ts",
		                      Seq("symbol", "dummy"))

		// perform join
		val joined_df = tsdf_left.asofJoin(tsdf_right, "left_")

	}

	it("Time-partitioned as-of join") {

		// perform the join
		val tsdf_left = TSDF(dfLeft2, "event_ts", Seq("symbol"))
		val tsdf_right = TSDF(dfRight2, "event_ts", Seq("symbol"))
		val joined_df = tsdf_left.asofJoin(tsdf_right, "left_", "right_", tsPartitionVal = 10, fraction = 0.1)
		
		assert(joined_df.df.collect().sameElements(dfExpected2.collect()))

		// this will execute the block printing out a message that values are being missing given the small tsPartitionVal window
		val missing_vals_joined_df = tsdf_left.asofJoin(tsdf_right, "left_", "right_", tsPartitionVal = 1, fraction = 0.1)
		val missing_vals_df_ct = missing_vals_joined_df.df.count()
		assert(missing_vals_df_ct == 7)
	}
}
