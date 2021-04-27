package com.databrickslabs.tempo

import com.databrickslabs.tempo.ml.stat.SMA
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{DoubleType, StringType}
import org.scalatest.FunSpec

class SMATests
	extends FunSpec with SparkSessionTestWrapper
{
	val schema = buildSchema(Seq(("symbol", StringType),
	                             ("event_ts", StringType),
	                             ("trade_pr", DoubleType)))

	val expectedSchema = buildSchema(Seq(("symbol", StringType),
	                                     ("event_ts", StringType),
	                                     ("trade_pr", DoubleType),
	                                     ("sma_trade_pr", DoubleType)))

	lazy val df = buildTestDF(schema,
	                          Seq(Row("S1", "2020-08-01 00:00:10", 8.0),
	                              Row("S1", "2020-08-01 00:01:12", 4.0),
	                              Row("S1", "2020-08-01 00:02:23", 2.0),
	                              Row("S2", "2020-09-01 00:02:10", 8.0),
	                              Row("S2", "2020-09-01 00:19:12", 16.0),
	                              Row("S2", "2020-09-01 00:19:12", 32.0)),
	                          "event_ts")

	lazy val dfExpected = buildTestDF(expectedSchema,
	                                  Seq(Row("S1", "2020-08-01 00:00:10", 8.0, 8.0),
	                                      Row("S1", "2020-08-01 00:01:12", 4.0, 6.0),
	                                      Row("S1", "2020-08-01 00:02:23", 2.0, 4.0 + (2.0/3.0)),
	                                      Row("S2", "2020-09-01 00:02:10", 8.0, 8.0),
	                                      Row("S2", "2020-09-01 00:19:12", 16.0, 12.0),
	                                      Row("S2", "2020-09-01 00:19:12", 32.0, 18.0 + (2.0/3.0))),
	                                  "event_ts" )

	it("SMA Transformer"){
		val tsdf = TSDF(df, tsColumnName = "event_ts", partitionColumnNames = Seq("symbol"))

		val sma_tx = SMA("trade_pr","ema_trade_pr",2)
		val smaDf = sma_tx.transform(tsdf).df

		assert(smaDf.collect().sameElements(dfExpected.collect()))
	}
}
