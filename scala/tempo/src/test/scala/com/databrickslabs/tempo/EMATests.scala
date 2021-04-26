package com.databrickslabs.tempo

import com.databrickslabs.tempo.ml.stat.{EMA => EMATX}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{DoubleType, StringType}
import org.scalatest.FunSpec

class EMATests
	extends FunSpec with SparkSessionTestWrapper
{
	val schema = buildSchema(Seq(("symbol", StringType),
	                             ("event_ts", StringType),
	                             ("trade_pr", DoubleType)))

	val expectedSchema = buildSchema(Seq(("symbol", StringType),
	                                     ("event_ts", StringType),
	                                     ("trade_pr", DoubleType),
	                                     ("ema_trade_pr", DoubleType)))

	lazy val df = buildTestDF(schema,
	                          Seq(Row("S1", "2020-08-01 00:00:10", 8.0),
		                            Row("S1", "2020-08-01 00:01:12", 4.0),
		                            Row("S1", "2020-08-01 00:02:23", 2.0),
		                            Row("S2", "2020-09-01 00:02:10", 8.0),
		                            Row("S2", "2020-09-01 00:19:12", 16.0),
		                            Row("S2", "2020-09-01 00:19:12", 32.0)),
	                          "event_ts")

	lazy val dfExpected = buildTestDF(expectedSchema,
	                                  Seq(Row("S1", "2020-08-01 00:00:10", 8.0, 4.0),
		                                    Row("S1", "2020-08-01 00:01:12", 4.0, 4.0),
		                                    Row("S1", "2020-08-01 00:02:23", 2.0, 3.0),
		                                    Row("S2", "2020-09-01 00:02:10", 8.0, 4.0),
		                                    Row("S2", "2020-09-01 00:19:12", 16.0, 10.0),
		                                    Row("S2", "2020-09-01 00:19:12", 32.0, 21.0)),
	                                  "event_ts" )

	it("EMA method") {

		val tsdf = TSDF(df, tsColumnName = "event_ts", partitionColumnNames = Seq("symbol"))
		val emaDf = tsdf.EMA("trade_pr", window = 2, exp_factor = 0.5).df

		assert(emaDf.collect().sameElements(dfExpected.collect()))
	}

	it("EMA Transformer"){
		val tsdf = TSDF(df, tsColumnName = "event_ts", partitionColumnNames = Seq("symbol"))

		val ema_tx = EMATX(measureCol = "trade_pr", emaCol = "ema_trade_pr", windowLength = 2, alpha = 0.5)
		val emaDf = ema_tx.transform(tsdf).df

		assert(emaDf.collect().sameElements(dfExpected.collect()))
	}

}
