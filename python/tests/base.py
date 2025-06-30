import os
import shutil
import unittest
import warnings
from typing import Union, Optional

import jsonref
import pandas as pd
import pyspark.sql.functions as sfn
from chispa import assert_df_equality
from delta.pip_utils import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame

from tempo.intervals import IntervalsDF
from tempo.tsdf import TSDF


class TestDataFrameBuilder:
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

    # Spark DataFrame metadata

    @property
    def df(self) -> dict:
        """
        :return: the DataFrame component of the test data
        """
        return self.__test_data["df"]

    @property
    def df_schema(self) -> str:
        """
        :return: the schema component of the test data
        """
        return self.df["schema"]

    def df_data(self) -> Union[list, pd.DataFrame]:
        """
        :return: the data component of the test data
        """
        data = self.df["data"]
        # return data literals (list of rows)
        if isinstance(data, list):
            return data
        # load data from a csv file
        elif isinstance(data, str):
            csv_path = SparkTest.getTestDataFilePath(data, extension='')
            return pd.read_csv(csv_path)
        else:
            raise ValueError(f"Invalid data type {type(data)}")

    # TSDF metadata

    @property
    def tsdf_constructor(self) -> Optional[str]:
        """
        :return: the name of the TSDF constructor to use
        """
        return self.__test_data.get("tsdf_constructor", None)

    @property
    def idf_construct(self) -> Optional[str]:
        """
        :return: the name of the IntervalsDF constructor to use
        """
        return self.__test_data.get("idf_constructor", None)

    @property
    def tsdf(self) -> dict:
        """
        :return: the timestamp index metadata component of the test data
        """
        return self.__test_data["tsdf"]

    @property
    def idf(self) -> dict:
        """
        :return: the start and end timestamp index metadata component of the test data
        """
        return self.__test_data["idf"]

    @property
    def ts_schema(self) -> Optional[dict]:
        """
        :return: the timestamp index schema component of the test data
        """
        return self.tsdf.get("ts_schema", None)

    @property
    def ts_idx_class(self) -> str:
        """
        :return: the timestamp index class component of the test data
        """
        return self.ts_schema["ts_idx_class"]

    @property
    def ts_col(self) -> str:
        """
        :return: the timestamp column component of the test data
        """
        return self.ts_schema["ts_col"]

    @property
    def ts_idx(self) -> dict:
        """
        :return: the timestamp index data component of the test data
        """
        return self.ts_schema["ts_idx"]

    # Builder functions

    def to_timestamp_ntz_compat(self, col):
        """
        Try to use sfn.to_timestamp_ntz if available, otherwise fall back to sfn.to_timestamp.
        """
        try:
            return sfn.to_timestamp_ntz(col)
        except AttributeError:
            # For older PySpark versions where to_timestamp_ntz does not exist
            return sfn.to_timestamp(col)

    def as_sdf(self) -> DataFrame:
        """
        Constructs a Spark Dataframe from the test data
        """
        # build dataframe
        df = self.spark.createDataFrame(self.df_data(), self.df_schema)

        # convert timestamp columns
        if "ts_convert" in self.df:
            for ts_col in self.df["ts_convert"]:
                # handle nested columns
                if "." in ts_col:
                    col, field = ts_col.split(".")
                    convert_field_expr = sfn.to_timestamp(sfn.col(col).getField(field))
                    df = df.withColumn(
                        col, sfn.col(col).withField(field, convert_field_expr)
                    )
                else:
                    df = df.withColumn(ts_col, sfn.to_timestamp(ts_col))
        if "ts_convert_ntz" in self.df:
            for ts_col in self.df["ts_convert_ntz"]:
                # handle nested columns
                if "." in ts_col:
                    col, field = ts_col.split(".")
                    convert_field_expr = self.to_timestamp_ntz_compat(sfn.col(col).getField(field))
                    df = df.withColumn(
                        col, sfn.col(col).withField(field, convert_field_expr)
                    )
                else:
                    df = df.withColumn(ts_col, self.to_timestamp_ntz_compat(ts_col))
        # convert date columns
        if "date_convert" in self.df:
            for date_col in self.df["date_convert"]:
                # handle nested columns
                if "." in date_col:
                    col, field = date_col.split(".")
                    convert_field_expr = sfn.to_timestamp(sfn.col(col).getField(field))
                    df = df.withColumn(
                        col, sfn.col(col).withField(field, convert_field_expr)
                    )
                else:
                    df = df.withColumn(date_col, sfn.to_date(date_col))

        if "decimal_convert" in self.df:
            for decimal_col in self.df["decimal_convert"]:
                if "." in date_col:
                    col, field = decimal_col.split(".")
                    convert_field_expr = sfn.col(col).getField(field).cast("decimal")
                    df = df.withColumn(
                        col, sfn.col(col).withField(field, convert_field_expr)
                    )
                else:
                    df = df.withColumn(decimal_col, sfn.col(decimal_col).cast("decimal"))

        return df

    def as_tsdf(self) -> TSDF:
        """
        Constructs a TSDF from the test data
        """
        sdf = self.as_sdf()
        if self.tsdf_constructor is not None:
            return getattr(TSDF, self.tsdf_constructor)(sdf, **self.tsdf)
        else:
            return TSDF(sdf, **self.tsdf)

    def as_idf(self) -> IntervalsDF:
        """
        Constructs a IntervalsDF from the test data
        """
        sdf = self.as_sdf()
        if self.idf_construct is not None:
            return getattr(IntervalsDF, self.idf_construct)(sdf, **self.idf)
        else:
            return IntervalsDF(self.as_sdf(), **self.idf)


class SparkTest(unittest.TestCase):
    #
    # Fixtures
    #

    # Spark Session object
    spark = None

    # test data
    test_data_file = None
    test_case_data = None

    @classmethod
    def _cleanup_delta_warehouse(cls) -> None:
        """
        Clean up Delta warehouse directories and metastore files that may interfere with tests
        """
        try:
            # List of paths to clean up
            cleanup_paths = [
                "spark-warehouse",
                "metastore_db",
                "derby.log"
            ]

            for path in cleanup_paths:
                if os.path.exists(path):
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
        except Exception as e:
            # Don't fail tests if cleanup fails, just warn
            warnings.warn(f"Failed to clean up Delta warehouse: {e}")

    @classmethod
    def setUpClass(cls) -> None:
        # Clean up any existing Delta tables and warehouse before starting tests
        cls._cleanup_delta_warehouse()
        
        # create and configure PySpark Session
        cls.spark = (
            configure_spark_with_delta_pip(SparkSession.builder.appName("unit-tests"))
            .config(
                "spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension",
            )
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
        # Clean up Delta tables and warehouse after tests complete
        cls._cleanup_delta_warehouse()

    def setUp(self) -> None:
        self.test_case_data = self.__loadTestData(self.id())

    def tearDown(self) -> None:
        del self.test_case_data

    #
    # Utility Functions
    #

    def get_data_as_idf(self, name: str, convert_ts_col=True):
        df = self.get_data_as_sdf(name, convert_ts_col)
        td = self.test_case_data[name]
        idf = IntervalsDF(
            df,
            start_ts=td["start_ts"],
            end_ts=td["end_ts"],
            series_ids=td.get("series", None),
        )
        return idf

    TEST_DATA_FOLDER = "unit_test_data"

    @classmethod
    def getTestDataFilePath(cls, test_file_name: str, extension: str = '.json') -> str:
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
        return f"{dir_path}/{cls.TEST_DATA_FOLDER}/{test_file_name}{extension}"

    def __loadTestData(self, test_case_path: str) -> dict:
        """
        This function reads our unit test data config json and returns the required metadata to create the correct
        format of test data (Spark DataFrames, Pandas DataFrames and Tempo TSDFs)
        :param test_case_path: string representation of the data path e.g. : "tsdf_tests.BasicTests.test_describe"
        :type test_case_path: str
        """
        file_name, class_name, func_name = test_case_path.split(".")[-3:]

        # load the test data file if it hasn't been loaded yet
        if self.test_data_file is None:
            # find our test data file
            test_data_filename = self.getTestDataFilePath(file_name)
            if not os.path.isfile(test_data_filename):
                warnings.warn(f"Could not load test data file {test_data_filename}")
                self.test_data_file = {}

            # proces the data file
            with open(test_data_filename, "r") as f:
                self.test_data_file = jsonref.load(f)

        # return the data if it exists
        if class_name in self.test_data_file:
            if func_name in self.test_data_file[class_name]:
                return self.test_data_file[class_name][func_name]

        # return empty dictionary if no data found
        return {}

    def get_test_df_builder(self, name: str) -> TestDataFrameBuilder:
        return TestDataFrameBuilder(self.spark, self.test_case_data[name])

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
        df1: Union[TSDF, DataFrame, IntervalsDF],
        df2: Union[TSDF, DataFrame, IntervalsDF],
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
            # get the underlying Spark DataFrames
            df1 = df1.df
            df2 = df2.df

        # Handle IDFs
        if isinstance(df1, IntervalsDF):
            # df2 must also be a IntervalsDF
            self.assertIsInstance(df2, IntervalsDF)
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
            ignore_metadata=True,
        )
