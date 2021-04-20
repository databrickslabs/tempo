package com.databrickslabs.tempo

import org.apache.spark.sql.{Row, SparkSession, DataFrame}
import org.apache.spark.sql.types._
import org.scalatest.{FunSpec, BeforeAndAfterAll, FeatureSpec, Matchers}
import org.apache.spark.sql.functions._


// test class for all functionality in tempo. no special functionality from Scalatest other than managing the tests with an ID. Basic assert statements to match data frames for now
class MirroredDataTests
  extends FunSpec
    with SparkSessionTestWrapper
{

  val schema = buildSchema(Seq(("symbol", StringType),
                               ("date", StringType),
                               ("event_ts", StringType),
                               ("trade_pr", DoubleType),
                               ("trade_pr_2", DoubleType)))

  val df = buildTestDF(schema,
                       Seq(Row("S1", "SAME_DT", "2020-08-01 00:00:10", 10.0, 349.21),
                           Row("S1", "SAME_DT", "2020-08-01 00:00:11", 9.0, 340.21),
                           Row("S1", "SAME_DT", "2020-08-01 00:01:12", 8.0, 353.32),
                           Row("S1", "SAME_DT", "2020-08-01 00:01:13", 7.0, 351.32),
                           Row("S1", "SAME_DT", "2020-08-01 00:01:14", 6.0, 350.32),
                           Row("S1", "SAME_DT", "2020-09-01 00:01:12", 5.0, 361.1),
                           Row("S1", "SAME_DT", "2020-09-01 00:19:12", 4.0, 362.1)),
                       "event_ts")

  it("Mirrored Data Frame Operations on TSDF")
  {
    // convert to TSDF
    val tsdf_left = TSDF(df, tsColumnName = "event_ts", partitionColumnNames = Seq("symbol"))
    val tsdf_right = TSDF(df, tsColumnName = "event_ts", partitionColumnNames = Seq("symbol"))

    val left_of_union = tsdf_left.select("event_ts", "symbol", "date", "trade_pr", "trade_pr_2").selectExpr("*").filter("1=1").where("2=2")
    val right_of_union = tsdf_right.select(cols= col("event_ts"), col("symbol"), col("date"), col("trade_pr"), col("trade_pr_2")).filter(col("trade_pr") === 4.0).where(col("trade_pr") === 4.0).limit(1)

    val result = left_of_union.union(right_of_union).unionAll(right_of_union).withColumn("extra", lit("Extra")).withColumn("to_be_dropped_1", lit(2)).withColumn("to_be_dropped_2", lit(3)).withColumnRenamed("trade_pr_2", "better_trade_pr").drop("date").drop("to_be_dropped_1", "to_be_dropped_2")

    assert(result.df.count == 9)
  }
}