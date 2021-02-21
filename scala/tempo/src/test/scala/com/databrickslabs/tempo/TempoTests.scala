package com.databrickslabs.tempo

import org.apache.spark.sql.{Row, SparkSession, DataFrame}
import org.apache.spark.sql.types._
import org.scalatest.{FunSpec, BeforeAndAfterAll, FeatureSpec, Matchers}
import org.apache.spark.sql.functions._


// create Spark session with local mode
trait SparkSessionTestWrapper {

  lazy val spark: SparkSession = {
    SparkSession.builder().master("local").config("spark.sql.shuffle.partitions", 1).config("spark.driver.bindAddress", "127.0.0.1").appName("spark session").getOrCreate()
  }

  import spark.implicits._

  /**
    * Constructs a Spark Dataframe from the given components
    * :param schema: the schema to use for the Dataframe
    * :param data: values to use for the Dataframe
    * :param ts_cols: list of column names to be converted to Timestamp values
    * :return: a Spark Dataframe, constructed from the given schema and values
    */
  def buildTestDF(schema : StructType, data : Seq[Row], ts_cols : List[String]) : DataFrame = {
    // build dataframe
    var df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)

    // convert all timestamp columns
    for (tsc <- ts_cols) {
      df = df.withColumn(tsc, to_timestamp(col(tsc)))
    }
    return df
  }

}

// test class for all functionality in tempo. no special functionality from Scalatest other than managing the tests with an ID. Basic assert statements to match data frames for now
class TempoTestSpec
  extends FunSpec
    with SparkSessionTestWrapper {

  it("Standard AS OF Join Test") {

    val leftSchema = StructType(List(StructField("symbol", StringType),
      StructField("event_ts", StringType),
      StructField("trade_pr", DoubleType)))

    val rightSchema = StructType(List(StructField("symbol", StringType),
      StructField("event_ts", StringType),
      StructField("bid_pr", DoubleType),
      StructField("ask_pr", DoubleType)))

    val expectedSchema = StructType(List(StructField("symbol", StringType),
      StructField("left_event_ts", StringType),
      StructField("left_trade_pr", DoubleType),
      StructField("right_event_ts", StringType),
      StructField("right_bid_pr", DoubleType),
      StructField("right_ask_pr", DoubleType)))

    val left_data = Seq(Row("S1", "2020-08-01 00:00:10", 349.21),
      Row("S1", "2020-08-01 00:01:12", 351.32),
      Row("S1", "2020-09-01 00:02:10", 361.1),
      Row("S1", "2020-09-01 00:19:12", 362.1))

    val right_data = Seq(Row("S1", "2020-08-01 00:00:01", 345.11, 351.12),
      Row("S1", "2020-08-01 00:01:05", 348.10, 353.13),
      Row("S1", "2020-09-01 00:02:01", 358.93, 365.12),
      Row("S1", "2020-09-01 00:15:01", 359.21, 365.31))

    val expected_data = Seq(Row("S1", "2020-08-01 00:00:10", 349.21, "2020-08-01 00:00:01", 345.11, 351.12),
      Row("S1", "2020-08-01 00:01:12", 351.32, "2020-08-01 00:01:05", 348.10, 353.13),
      Row("S1", "2020-09-01 00:02:10", 361.1, "2020-09-01 00:02:01", 358.93, 365.12),
      Row("S1", "2020-09-01 00:19:12", 362.1, "2020-09-01 00:15:01", 359.21, 365.31))

    // Construct dataframes
    var dfLeft = spark.createDataFrame(spark.sparkContext.parallelize(left_data), leftSchema)
    var dfRight = spark.createDataFrame(spark.sparkContext.parallelize(right_data), rightSchema)
    var dfExpected = spark.createDataFrame(spark.sparkContext.parallelize(expected_data), expectedSchema)

    dfLeft = dfLeft.withColumn("event_ts", col("event_ts").cast("timestamp"))
    dfRight = dfRight.withColumn("event_ts", col("event_ts").cast("timestamp"))
    dfExpected = dfExpected.withColumn("left_event_ts", col("left_event_ts").cast("timestamp"))
    dfExpected = dfExpected.withColumn("right_event_ts", col("right_event_ts").cast("timestamp"))

    // perform the join
    val tsdf_left = TSDF(dfLeft, tsColumnName = "event_ts", partitionColumnNames = "symbol")
    val tsdf_right = TSDF(dfRight, tsColumnName = "event_ts", partitionColumnNames = "symbol")


    val joined_df = tsdf_left.asofJoin(tsdf_right)

    // run basic assertion of the elements being the same
    assert(joined_df.df.collect().sameElements(tsdf_right.df.collect()))

  }

  it("Resample test") {
    val schema = StructType(List(StructField("symbol", StringType),
    StructField("date", StringType),
    StructField("event_ts", StringType),
    StructField("trade_pr", DoubleType),
    StructField("trade_pr_2", DoubleType)) )

    val expectedSchema = StructType(List(StructField("symbol", StringType),
    StructField("event_ts", StringType),
    StructField("date", StringType),
    StructField("trade_pr", DoubleType),
    StructField("trade_pr_2", DoubleType)))

    val data =
    Seq(Row("S1", "SAME_DT", "2020-08-01 00:00:10", 10.0, 349.21),
    Row("S1", "SAME_DT", "2020-08-01 00:00:11", 9.0, 340.21),
    Row("S1", "SAME_DT", "2020-08-01 00:01:12", 8.0, 353.32),
    Row("S1", "SAME_DT", "2020-08-01 00:01:13", 7.0, 351.32),
    Row("S1", "SAME_DT", "2020-08-01 00:01:14", 6.0, 350.32),
    Row("S1", "SAME_DT", "2020-09-01 00:01:12", 5.0, 361.1),
    Row("S1", "SAME_DT", "2020-09-01 00:19:12", 4.0, 362.1))

    val expected_data =
    Seq(Row("S1", "2020-08-01 00:00:00", "SAME_DT", 10.0, 349.21),
    Row("S1", "2020-08-01 00:01:00", "SAME_DT", 8.0, 353.32),
    Row("S1", "2020-09-01 00:01:00", "SAME_DT", 5.0, 361.1 ),
    Row("S1", "2020-09-01 00:19:00", "SAME_DT", 4.0, 362.1))

    //construct dataframes
    val df = buildTestDF(schema, data, List("event_ts"))
    val dfExpected = buildTestDF(expectedSchema, expected_data, List("event_ts"))

    // convert to TSDF
    val tsdf_left = TSDF(df, tsColumnName = "event_ts", partitionColumnNames = "symbol")

    // using lookback of 20 minutes
    val featured_df = tsdf_left.resample(freq = "min", func = "closest_lead").df

    // should be equal to the expected dataframe
    featured_df.show(100, false)
    println(featured_df.schema)
    dfExpected.show(100, false)
    assert(featured_df.collect().sameElements(dfExpected.collect()))
  }
}