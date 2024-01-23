import os
import re
import unittest
import warnings
from typing import Union, Optional

import jsonref
from chispa import assert_df_equality

import pyspark.sql.functions as sfn
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame

from tempo.intervals import IntervalsDF
from tempo.tsdf import TSDF


class TestDataFrame:
    """
    A class to hold metadata about a Spark DataFrame
    """

    def __init__(self, spark: SparkSession, test_data: dict):
        """
        :param spark: the SparkSession to use
        :param test_data: a dictionary containing the test data & metadata
        """
        self.spark = spark
        self.__test_data = test_data

    @property
    def df(self):
        """
        :return: the DataFrame component of the test data
        """
        return self.__test_data["df"]

    @property
    def schema(self):
        """
        :return: the schema component of the test data
        """
        return self.df["schema"]

    def data(self):
        """
        :return: the data component of the test data
        """
        return self.df["data"]

    @property
    def ts_idx(self):
        """
        :return: the timestamp index metadata component of the test data
        """
        return self.__test_data["ts_idx"]

    @property
    def tsdf_constructor(self) -> Optional[str]:
        """
        :return: the name of the TSDF constructor to use
        """
        return self.__test_data.get("tsdf_constructor", None)

    def as_sdf(self) -> DataFrame:
        """
        Constructs a Spark Dataframe from the test data
        """
        # build dataframe
        df = self.spark.createDataFrame(self.data(), self.schema)

        # convert timestamp columns
        if "ts_convert" in self.df:
            for ts_col in self.df["ts_convert"]:
                df = df.withColumn(ts_col, sfn.to_timestamp(ts_col))
        # convert date columns
        if "date_convert" in self.df:
            for date_col in self.df["date_convert"]:
                df = df.withColumn(date_col, sfn.to_date(date_col))

        return df

    def as_tsdf(self) -> TSDF:
        """
        Constructs a TSDF from the test data
        """
        sdf = self.as_sdf()
        if self.tsdf_constructor is not None:
            return getattr(TSDF, self.tsdf_constructor)(sdf, **self.ts_idx)
        else:
            return TSDF(sdf, **self.ts_idx)


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
        if self.test_data is None:
            self.test_data = self.__loadTestData(self.id())

    #
    # Test Data Loading Functions
    #

    # def get_data_as_sdf(self, name: str) -> DataFrame:
    #     td = self.test_data[name]
    #     return self.buildTestDF(td["df"])

    # def get_ts_idx_metadata(self, name: str) -> dict:
    #     td = self.test_data[name]
    #     return td["ts_idx"]
    #
    # def get_tsdf_constructor_fn(self, name: str) -> str:
    #     td = self.test_data[name]
    #     return td.get("constructor", None)
    #
    # def get_data_as_tsdf(self, name: str) -> TSDF:
    #     sdf = self.get_data_as_sdf(name)
    #     ts_idx_meta = self.get_ts_idx_metadata(name)
    #     tsdf_constructor = self.get_tsdf_constructor_fn(name)
    #     if tsdf_constructor is not None:
    #         return getattr(TSDF, tsdf_constructor)(sdf, **ts_idx_meta)
    #     else:
    #         return TSDF(sdf, **ts_idx_meta)
    #
    # def get_data_as_idf(self, name: str, convert_ts_col=True):
    #     df = self.get_data_as_sdf(name, convert_ts_col)
    #     td = self.test_data[name]
    #     idf = IntervalsDF(
    #         df,
    #         start_ts=td["start_ts"],
    #         end_ts=td["end_ts"],
    #         series_ids=td.get("series", None),
    #     )
    #     return idf

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
                f"Cannot locate test data file {test_file_name}, running from dir"
                f" {os.getcwd()}"
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
            # return the data
            return data_metadata_from_json

    def get_test_data(self, name: str) -> TestDataFrame:
        return TestDataFrame(self.spark, self.test_data[name])

    # def buildTestDF(self, df_spec) -> DataFrame:
    #     """
    #     Constructs a Spark Dataframe from the given components
    #     :param df_spec: a dictionary containing the following keys: schema, data, ts_convert
    #     :return: a Spark Dataframe, constructed from the given schema and values
    #     """
    #     # build dataframe
    #     df = self.spark.createDataFrame(df_spec['data'], df_spec['schema'])
    #
    #     # convert timestamp columns
    #     if 'ts_convert' in df_spec:
    #         for ts_col in df_spec['ts_convert']:
    #             df = df.withColumn(ts_col, sfn.to_timestamp(ts_col))
    #     return df

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


    def assertDataFrameEquality(
        self,
        df1: Union[TSDF, DataFrame],
        df2: Union[TSDF, DataFrame],
        ignore_row_order: bool = False,
        ignore_column_order: bool = True,
        ignore_nullable: bool = True,
    ):
        """
        Test that the two given Dataframes are equivalent.
        That is, they have equivalent schemas, and both contain the same values
        """

        # handle TSDFs
        if isinstance(df1, TSDF):
            # df2 must also be a TSDF
            self.assertIsInstance(df2, TSDF)
            # should have the same schemas
            self.assertEqual(df1.ts_schema, df2.ts_schema)
            # get the underlying Spark DataFrames
            df1 = df1.df
            df2 = df2.df

        # handle DataFrames
        assert_df_equality(
            df1,
            df2,
            ignore_row_order=ignore_row_order,
            ignore_column_order=ignore_column_order,
            ignore_nullable=ignore_nullable,
        )
