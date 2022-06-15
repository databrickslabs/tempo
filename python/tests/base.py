import re
import unittest
import warnings
import json

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from tempo.tsdf import TSDF


class SparkTest(unittest.TestCase):
    #
    # Fixtures
    #

    # Spark Session object
    spark = None
    test_data = None

    @classmethod
    def setUpClass(cls) -> None:
        # create and configure PySpark Session
        cls.spark = (
            SparkSession.builder.appName("myapp")
            .config("spark.jars.packages", "io.delta:delta-core_2.12:1.1.0")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
            .config(
                "spark.driver.extraJavaOptions",
                "-Dio.netty.tryReflectionSetAccessible=true",
            )
            .config(
                "spark.executor.extraJavaOptions",
                "-Dio.netty.tryReflectionSetAccessible=true",
            )
            .master("local")
            .getOrCreate()
        )
        cls.spark.conf.set("spark.sql.shuffle.partitions", 1)
        cls.spark.sparkContext.setLogLevel("ERROR")
        # filter out ResourceWarning messages
        warnings.filterwarnings("ignore", category=ResourceWarning)

    @classmethod
    def tearDownClass(cls) -> None:
        # shut down Spark
        cls.spark.stop()

    def setUp(self) -> None:
        self.test_data = self.loadTestData(self.id())

    def tearDown(self) -> None:
        del self.test_data

    #
    # Utility Functions
    #

    def get_data_as_sdf(self, name: str):
        schema = self.test_data[name]["schema"]
        data = self.test_data[name]["data"]
        ts_cols = self.test_data[name].get("ts_cols", None)
        return self.buildTestDF(schema, data, ts_cols)

    def get_data_as_tsdf(self, name: str):
        df = self.get_data_as_sdf(name)
        tsdf = TSDF(
            df,
            ts_col=self.test_data[name]["ts_col"],
            partition_cols=self.test_data[name]["partition_cols"],
            sequence_col=self.test_data[name]["sequence_col"],
        )
        return tsdf

    TEST_DATA_FOLDER = "unit_test_data"

    def loadTestData(self, test_case_path: str) -> dict:
        """
        This function reads our unit test data config json and returns the required metadata to create the correct
        format of test data (Spark DataFrames, Pandas DataFrames and Tempo TSDFs)
        :param test_case_path: string representation of the data path e.g. : "tsdf_tests.BasicTests.test_describe"
        :type test_case_path: str
        """
        file_name, class_name, func_name = test_case_path.split(".")
        test_data_file = f"{self.TEST_DATA_FOLDER}/{file_name}.json"
        with open(test_data_file) as f:
            data_metadata_from_json = json.load(f)
            return data_metadata_from_json[class_name][func_name]

    def buildTestDF(self, schema, data, ts_cols=["event_ts"]):
        """
        Constructs a Spark Dataframe from the given components
        :param schema: the schema to use for the Dataframe
        :param data: values to use for the Dataframe
        :param ts_cols: list of column names to be converted to Timestamp values
        :return: a Spark Dataframe, constructed from the given schema and values
        """
        # build dataframe
        df = self.spark.createDataFrame(data, schema)

        # check if ts_col follows standard timestamp format, then check if timestamp has micro/nanoseconds
        for tsc in ts_cols:
            ts_value = str(df.select(ts_cols).limit(1).collect()[0][0])
            ts_pattern = "^\d{4}-\d{2}-\d{2}| \d{2}:\d{2}:\d{2}\.\d*$"
            decimal_pattern = "[.]\d+"
            if re.match(ts_pattern, str(ts_value)) is not None:
                if (
                    re.search(decimal_pattern, ts_value) is None
                    or len(re.search(decimal_pattern, ts_value)[0]) <= 4
                ):
                    df = df.withColumn(tsc, F.to_timestamp(F.col(tsc)))
        return df

    #
    # DataFrame Assert Functions
    #

    def assertFieldsEqual(self, fieldA, fieldB):
        """
        Test that two fields are equivalent
        """
        self.assertEqual(fieldA.name.lower(), fieldB.name.lower())
        self.assertEqual(fieldA.dataType, fieldB.dataType)
        # self.assertEqual(fieldA.nullable, fieldB.nullable)

    def assertSchemaContainsField(self, schema, field):
        """
        Test that the given schema contains the given field
        """
        # the schema must contain a field with the right name
        lc_fieldNames = [fc.lower() for fc in schema.fieldNames()]
        self.assertTrue(field.name.lower() in lc_fieldNames)
        # the attributes of the fields must be equal
        self.assertFieldsEqual(field, schema[field.name])

    def assertSchemasEqual(self, schemaA, schemaB):
        """
        Test that the two given schemas are equivalent (column ordering ignored)
        """
        # both schemas must have the same length
        self.assertEqual(len(schemaA.fields), len(schemaB.fields))
        # schemaA must contain every field in schemaB
        for field in schemaB.fields:
            self.assertSchemaContainsField(schemaA, field)

    def assertHasSchema(self, df, expectedSchema):
        """
        Test that the given Dataframe conforms to the expected schema
        """
        self.assertSchemasEqual(df.schema, expectedSchema)

    def assertDataFramesEqual(self, dfA, dfB):
        """
        Test that the two given Dataframes are equivalent.
        That is, they have equivalent schemas, and both contain the same values
        """
        # must have the same schemas
        self.assertSchemasEqual(dfA.schema, dfB.schema)
        # enforce a common column ordering
        colOrder = sorted(dfA.columns)
        sortedA = dfA.select(colOrder)
        sortedB = dfB.select(colOrder)
        # must have identical data
        # that is all rows in A must be in B, and vice-versa

        self.assertEqual(sortedA.subtract(sortedB).count(), 0)
        self.assertEqual(sortedB.subtract(sortedA).count(), 0)
