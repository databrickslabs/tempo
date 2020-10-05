import unittest

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import *

from tempo.tsdf import TSDF

class SparkTest(unittest.TestCase):
    ##
    ## Fixtures
    ##
    def setUp(self):
        self.spark = (SparkSession.builder
                      .master("local")
                      .getOrCreate())

    def tearDown(self) -> None:
        self.spark.stop()

    ##
    ## Utility Functions
    ##

    def buildTestDF(self, schema, data, ts_cols = ["event_ts"]):
      # build dataframe
      df = self.spark.createDataFrame(data, schema)

      # convert all timestamp columns
      for tsc in ts_cols:
        df = df.withColumn(tsc, F.to_timestamp(F.col(tsc)))

      return df

    ##
    ## DataFrame Assert Functions
    ##

    def assertFieldsEqual(self, fieldA, fieldB):
        self.assertEqual(fieldA.name.lower(), fieldB.name.lower())
        self.assertEqual(fieldA.dataType, fieldB.dataType)
        self.assertEqual(fieldA.nullable, fieldB.nullable)

    def assertSchemaContainsField(self, schema, field):
        # the schema must contain a field with the right name
        lc_fieldNames = [fc.lower() for fc in schema.fieldNames()]
        self.assertTrue( field.name.lower() in lc_fieldNames )
        # the attributes of the fields must be equal
        self.assertFieldsEqual( field, schema[field.name] )

    def assertSchemasEqual(self, schemaA, schemaB):
        # both schemas must have the same length
        self.assertEqual( len(schemaA.fields), len(schemaB.fields) )
        # schemaA must contain every field in schemaB
        for field in schemaB.fields:
            self.assertSchemaContainsField( schemaA, field )

    def assertHasSchema(self,df,expectedSchema):
        self.assertSchemasEqual(df.schema,expectedSchema)

    def assertDataFramesEqual(self,dfA,dfB):
        # must have the same schemas
        self.assertSchemasEqual(dfA.schema,dfB.schema)
        # enforce a common column ordering
        colOrder = sorted(dfA.columns)
        sortedA = dfA.select(colOrder)
        sortedB = dfB.select(colOrder)
        # must have identical data
        # that is all rows in A must be in B, and vice-versa
        self.assertEqual( sortedA.subtract(sortedB).count(), 0 )
        self.assertEqual( sortedB.subtract(sortedA).count(), 0 )


class AsOfJoinTest(SparkTest):

    def test_asof_join(self):
        leftSchema = StructType([StructField("symbol", StringType()),
                                 StructField("event_ts", StringType()),
                                 StructField("trade_pr", FloatType())])

        rightSchema = StructType([StructField("symbol", StringType()),
                                  StructField("event_ts", StringType()),
                                  StructField("bid_pr", FloatType()),
                                  StructField("ask_pr", FloatType())])

        expectedSchema = StructType([StructField("symbol", StringType()),
                                     StructField("EVENT_TS_left", StringType()),
                                     StructField("trade_pr", FloatType()),
                                     StructField("EVENT_TS_right", StringType()),
                                     StructField("bid_pr", FloatType()),
                                     StructField("ask_pr", FloatType())])

        left_data = [["S1", "2020-08-01 00:00:10", 349.21],
                     ["S1", "2020-08-01 00:01:12", 351.32],
                     ["S1", "2020-09-01 00:02:10", 361.1],
                     ["S1", "2020-09-01 00:19:12", 362.1]]

        right_data = [["S1", "2020-08-01 00:00:01",  345.11, 351.12],
                      ["S1", "2020-08-01 00:01:05",  348.10, 353.13],
                      ["S1", "2020-09-01 00:02:01",  358.93, 365.12],
                      ["S1", "2020-09-01 00:15:01",  359.21, 365.31]]

        expected_data = [
            ["S1", "2020-08-01 00:00:10", 349.21, "2020-08-01 00:00:01",  345.11, 351.12],
            ["S1", "2020-08-01 00:01:12", 351.32, "2020-08-01 00:01:05",  348.10, 353.13],
            ["S1", "2020-09-01 00:02:10", 361.1, "2020-09-01 00:02:01",  358.93, 365.12],
            ["S1", "2020-09-01 00:19:12", 362.1, "2020-09-01 00:15:01",  359.21, 365.31]]


        # construct dataframes
        dfLeft = self.buildTestDF(leftSchema, left_data)
        dfRight = self.buildTestDF(rightSchema, right_data)
        dfExpected = self.buildTestDF(expectedSchema, expected_data, ["EVENT_TS_right", "EVENT_TS_left"])

        # perform the join
        tsdf_left = TSDF(dfLeft)
        joined_df = tsdf_left.asofJoin(dfRight, partitionCols=["symbol"]).df

        # joined dataframe should equal the expected dataframe
        self.assertDataFramesEqual( joined_df, dfExpected )

class RangStatsTest(SparkTest):

    def test_range_stats(self):
        """
        This method tests the lookback stats for one numeric column (trade_pr).
        :param - debug is used for testing only - switch to True to see data frame output
        input parameters to the stats which are unique to this test are 1200 for 20 minute lookback
        """
        schema = StructType([StructField("symbol", StringType()),
                             StructField("event_ts", StringType()),
                             StructField("trade_pr", FloatType())])

        expectedSchema = StructType([StructField("symbol", StringType()),
                                     StructField("event_ts", StringType()),
                                     StructField("mean_trade_pr", FloatType()),
                                     StructField("count_trade_pr", LongType(), nullable=False),
                                     StructField("min_trade_pr", FloatType()),
                                     StructField("max_trade_pr", FloatType()),
                                     StructField("sum_trade_pr", FloatType()),
                                     StructField("stddev_trade_pr", FloatType()),
                                     StructField("zscore_trade_pr", FloatType())])

        data = [["S1", "2020-08-01 00:00:10", 349.21],
                                  ["S1", "2020-08-01 00:01:12", 351.32],
                                  ["S1", "2020-09-01 00:02:10", 361.1],
                                  ["S1", "2020-09-01 00:19:12", 362.1]]

        expected_data = [
                ["S1", "2020-08-01 00:00:10", 349.21, 1, 349.21, 349.21, 349.21, None, None],
                ["S1", "2020-08-01 00:01:12", 350.26, 2, 349.21, 351.32, 700.53, 1.49, 0.71],
                ["S1", "2020-09-01 00:02:10", 361.1, 1, 361.1, 361.1, 361.1, None, None],
                ["S1", "2020-09-01 00:19:12", 361.6, 2, 361.1, 362.1, 723.2, 0.71, 0.71]]

        # construct dataframes
        df = self.buildTestDF(schema,data)
        dfExpected = self.buildTestDF(expectedSchema,expected_data)

        # convert to TSDF
        tsdf_left = TSDF(df)

        # using lookback of 20 minutes
        featured_df = tsdf_left.withRangeStats(partitionCols=["symbol"], rangeBackWindowSecs=1200).df

        # cast to decimal with precision in cents for simplicity
        featured_df = featured_df.select(F.col("symbol"), F.col("event_ts"),
                                          F.col("mean_trade_pr").cast("decimal(5, 2)"),
                                          F.col("count_trade_pr"),
                                          F.col("min_trade_pr").cast("decimal(5,2)"),
                                          F.col("max_trade_pr").cast("decimal(5,2)"),
                                          F.col("sum_trade_pr").cast("decimal(5,2)"),
                                          F.col("stddev_trade_pr").cast("decimal(5,2)"),
                                          F.col("zscore_trade_pr").cast("decimal(5,2)"))

        # cast to decimal with precision in cents for simplicity
        dfExpected = dfExpected.select(F.col("symbol"), F.col("event_ts"),
                                        F.col("mean_trade_pr").cast("decimal(5, 2)"),
                                        F.col("count_trade_pr"),
                                        F.col("min_trade_pr").cast("decimal(5,2)"),
                                        F.col("max_trade_pr").cast("decimal(5,2)"),
                                        F.col("sum_trade_pr").cast("decimal(5,2)"),
                                        F.col("stddev_trade_pr").cast("decimal(5,2)"),
                                        F.col("zscore_trade_pr").cast("decimal(5,2)"))

        # should be equal to the expected dataframe
        self.assertDataFramesEqual(featured_df, dfExpected)

## MAIN
if __name__ == '__main__':
    unittest.main()
