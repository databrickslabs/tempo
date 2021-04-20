package com.databrickslabs.tempo

import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{DoubleType, StringType}
import org.scalatest.FunSpec

class ResampleTests
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
	                                     ("trade_pr", DoubleType),
	                                     ("trade_pr_2", DoubleType)))

	lazy val df = buildTestDF(schema,
	                          Seq(Row("S1", "SAME_DT", "2020-08-01 00:00:10", 10.0, 349.21),
	                              Row("S1", "SAME_DT", "2020-08-01 00:00:11", 9.0, 340.21),
	                              Row("S1", "SAME_DT", "2020-08-01 00:01:12", 8.0, 353.32),
	                              Row("S1", "SAME_DT", "2020-08-01 00:01:13", 7.0, 351.32),
	                              Row("S1", "SAME_DT", "2020-08-01 00:01:14", 6.0, 350.32),
	                              Row("S1", "SAME_DT", "2020-09-01 00:01:12", 5.0, 361.1),
	                              Row("S1", "SAME_DT", "2020-09-01 00:19:12", 4.0, 362.1)),
	                          "event_ts")

	lazy val dfExpected = buildTestDF(expectedSchema,
	                                  Seq(Row("S1", "2020-08-01 00:00:00", "SAME_DT", 10.0, 349.21),
	                                      Row("S1", "2020-08-01 00:01:00", "SAME_DT", 8.0, 353.32),
	                                      Row("S1", "2020-09-01 00:01:00", "SAME_DT", 5.0, 361.1),
	                                      Row("S1", "2020-09-01 00:19:00", "SAME_DT", 4.0, 362.1)),
	                                  "event_ts")

	it("Resample test") {
		// convert to TSDF
		val tsdf_left = TSDF(df, tsColumnName = "event_ts", partitionColumnNames = Seq("symbol"))

		// using lookback of 20 minutes
		val featured_df = tsdf_left.resample(freq = "min", func = "closest_lead").df

		// should be equal to the expected dataframe
		assert(featured_df.collect().sameElements(dfExpected.collect()))
	}
}
