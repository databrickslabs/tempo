package com.databrickslabs.tempo

import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{DoubleType, IntegerType, StringType, StructField, StructType}
import org.scalatest.FunSpec

class RangeStatsTests
	extends FunSpec with SparkSessionTestWrapper
{
	val schema = buildSchema(Seq(("symbol", StringType),
	                             ("event_ts", StringType),
	                             ("trade_pr", DoubleType)))

	val expectedSchema = StructType(List(StructField("symbol", StringType),
	                                     StructField("event_ts", StringType),
	                                     StructField("mean_trade_pr", DoubleType),
	                                     StructField("count_trade_pr", IntegerType, nullable = false),
	                                     StructField("min_trade_pr", DoubleType),
	                                     StructField("max_trade_pr", DoubleType),
	                                     StructField("sum_trade_pr", DoubleType),
	                                     StructField("stddev_trade_pr", DoubleType),
	                                     StructField("zscore_trade_pr", DoubleType)))

	lazy val df = buildTestDF(schema,
	                          Seq(Row("S1", "2020-08-01 00:00:10", 349.21),
		                            Row("S1", "2020-08-01 00:01:12", 351.32),
		                            Row("S2", "2020-09-01 00:02:10", 361.1),
		                            Row("S2", "2020-09-01 00:19:12", 362.1)),
	                          "event_ts" )

	lazy val dfExpected = buildTestDF(expectedSchema,
	                                  Seq(Row("S1", "2020-08-01 00:00:10", 349.21, 1, 349.21, 349.21, 349.21, null, null),
		                                    Row("S1", "2020-08-01 00:01:12", 350.27, 2, 349.21, 351.32, 700.53, 1.49, 0.71),
		                                    Row("S2", "2020-09-01 00:02:10", 361.1, 1, 361.1, 361.1, 361.1, null, null),
		                                    Row("S2", "2020-09-01 00:19:12", 361.6, 2, 361.1, 362.1, 723.2, 0.71, 0.71)),
	                                  "event_ts" ).select(col("symbol"),
	                                                              col("event_ts"),
	                                                              col("mean_trade_pr").cast("decimal(5,2)"),
	                                                              col("count_trade_pr"),
	                                                              col("min_trade_pr").cast("decimal(5,2)"),
	                                                              col("max_trade_pr").cast("decimal(5,2)"),
	                                                              col("stddev_trade_pr").cast("decimal(5,2)"),
	                                                              col("zscore_trade_pr").cast("decimal(5,2)"))

	it("range-stats test") {

		val tsdf = TSDF(df, "event_ts", Seq("symbol"))

		val featuredDf = tsdf.rangeStats(rangeBackWindowSecs = 1200)
		                     .df
		                     .select(col("symbol"),
																 col("event_ts"),
																 col("mean_trade_pr").cast("decimal(5,2)"),
																 col("count_trade_pr").cast("integer"),
																 col("min_trade_pr").cast("decimal(5,2)"),
																 col("max_trade_pr").cast("decimal(5,2)"),
																 col("stddev_trade_pr").cast("decimal(5,2)"),
																 col("zscore_trade_pr").cast("decimal(5,2)"))

		assert(featuredDf.collect().sameElements(dfExpected.collect()))
	}
}
