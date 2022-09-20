import re
import os
import unittest
import warnings
from typing import Union

import jsonref

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from tempo.tsdf import TSDF
from tempo.intervals import IntervalsDF
from chispa import assert_df_equality
from pyspark.sql.dataframe import DataFrame


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
            SparkSession.builder.appName("unit-tests")
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
            .config("spark.sql.session.timeZone", "UTC")
            .master("local")
            .getOrCreate()
        )
        cls.spark.conf.set("spark.sql.shuffle.partitions", "1")
        cls.spark.sparkContext.setLogLevel("ERROR")
        # filter out ResourceWarning messages
        warnings.filterwarnings("ignore", category=ResourceWarning)

    @classmethod
    def tearDownClass(cls) -> None:
        # shut down Spark
        cls.spark.stop()

    def setUp(self) -> None:
        self.test_data = self.__loadTestData(self.id())

    def tearDown(self) -> None:
        del self.test_data

    #
    # Utility Functions
    #

    def get_data_as_sdf(self, name: str, convert_ts_col=True):
        td = self.test_data[name]
        ts_cols = []
        if convert_ts_col and (td.get("ts_col", None) or td.get("other_ts_cols", [])):
            ts_cols = [td["ts_col"]] if "ts_col" in td else []
            ts_cols.extend(td.get("other_ts_cols", []))
        return self.buildTestDF(td["schema"], td["data"], ts_cols)

    def get_data_as_tsdf(self, name: str, convert_ts_col=True):
        df = self.get_data_as_sdf(name, convert_ts_col)
        td = self.test_data[name]
        tsdf = TSDF(
            df,
            ts_col=td["ts_col"],
            partition_cols=td.get("partition_cols", None),
            sequence_col=td.get("sequence_col", None),
        )
        return tsdf

    def get_data_as_idf(self, name: str, convert_ts_col=True):
        df = self.get_data_as_sdf(name, convert_ts_col)
        td = self.test_data[name]
        idf = IntervalsDF(
            df,
            start_ts=td["start_ts"],
            end_ts=td["end_ts"],
            series_ids=td.get("series", None),
        )
        return idf

    TEST_DATA_FOLDER = "unit_test_data"

    def __getTestDataFilePath(self, test_file_name: str) -> str:
        # what folder are we running from?
        cwd = os.path.basename(os.getcwd())

        # build path based on what folder we're in
        dir_path = "./"
        if cwd == "tempo":
            dir_path = "./python/tests"
        elif cwd == "python":
            dir_path = "./tests"
        elif cwd != "tests":
            raise RuntimeError(
                f"Cannot locate test data file {test_file_name}, running from dir {os.getcwd()}"
            )

        # return appropriate path
        return f"{dir_path}/{self.TEST_DATA_FOLDER}/{test_file_name}.json"

    def __loadTestData(self, test_case_path: str) -> dict:
        """
        This function reads our unit test data config json and returns the required metadata to create the correct
        format of test data (Spark DataFrames, Pandas DataFrames and Tempo TSDFs)
        :param test_case_path: string representation of the data path e.g. : "tsdf_tests.BasicTests.test_describe"
        :type test_case_path: str
        """
        file_name, class_name, func_name = test_case_path.split(".")

        # find our test data file
        test_data_file = self.__getTestDataFilePath(file_name)
        if not os.path.isfile(test_data_file):
            warnings.warn(f"Could not load test data file {test_data_file}")
            return {}

        # proces the data file
        with open(test_data_file, "r") as f:
            data_metadata_from_json = jsonref.load(f)
            # warn if data not present
            if class_name not in data_metadata_from_json:
                warnings.warn(f"Could not load test data for {file_name}.{class_name}")
                return {}
            if func_name not in data_metadata_from_json[class_name]:
                warnings.warn(
                    f"Could not load test data for {file_name}.{class_name}.{func_name}"
                )
                return {}
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
            ts_pattern = r"^\d{4}-\d{2}-\d{2}| \d{2}:\d{2}:\d{2}\.\d*$"
            decimal_pattern = r"[.]\d+"
            if re.match(ts_pattern, str(ts_value)) is not None:
                if (
                    re.search(decimal_pattern, ts_value) is None
                    or len(re.search(decimal_pattern, ts_value)[0]) <= 4
                ):
                    df = df.withColumn(tsc, F.to_timestamp(F.col(tsc)))
        return df

    #
    # Assertion Functions
    #

    def assertFieldsEqual(self, fieldA, fieldB):
        """
        Test that two fields are equivalent
        """
        self.assertEqual(
            fieldA.name.lower(),
            fieldB.name.lower(),
            msg=f"Field {fieldA} has different name from {fieldB}",
        )
        self.assertEqual(
            fieldA.dataType,
            fieldB.dataType,
            msg=f"Field {fieldA} has different type from {fieldB}",
        )
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

    @staticmethod
    def assertDataFrameEquality(
        df1: Union[IntervalsDF, TSDF, DataFrame],
        df2: Union[IntervalsDF, TSDF, DataFrame],
        from_tsdf: bool = False,
        from_idf: bool = False,
        ignore_row_order: bool = False,
        ignore_column_order: bool = True,
        ignore_nullable: bool = True,
    ):
        """
        Test that the two given Dataframes are equivalent.
        That is, they have equivalent schemas, and both contain the same values
        """

        if from_tsdf or from_idf:
            df1 = df1.df
            df2 = df2.df

        assert_df_equality(
            df1,
            df2,
            ignore_row_order=ignore_row_order,
            ignore_column_order=ignore_column_order,
            ignore_nullable=ignore_nullable,
        )
