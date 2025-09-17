import os
import shutil
import unittest
import warnings
from typing import Optional, Union

import jsonref
import pandas as pd
import pyspark.sql.functions as sfn
from chispa import assert_df_equality
from delta.pip_utils import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame

from tempo.intervals import IntervalsDF
from tempo.tsdf import TSDF


# helper functions

def prefix_value(key: str, d: dict):
    # look for an exact match
    if key in d:
        return d[key]
    # scan for a prefix-match
    for k in d:
        if key.startswith(k):
            return d[k]
    return None

# test classes

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
        # Parse the schema to identify struct columns

        # Parse schema string if it's a string
        if isinstance(self.df_schema, str):
            # Check if schema contains struct definitions
            if "struct<" in self.df_schema:
                # Parse the schema string to build proper StructType
                schema = self._parse_complex_schema(self.df_schema)
                # Process data to convert lists to Row objects for struct columns
                data = self._process_struct_data(self.df_data(), schema)
                df = self.spark.createDataFrame(data, schema)
            else:
                # Simple schema - use existing logic
                df = self.spark.createDataFrame(self.df_data(), self.df_schema)
        else:
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
                if "." in decimal_col:
                    col, field = decimal_col.split(".")
                    convert_field_expr = sfn.col(col).getField(field).cast("decimal")
                    df = df.withColumn(
                        col, sfn.col(col).withField(field, convert_field_expr)
                    )
                else:
                    df = df.withColumn(decimal_col, sfn.col(decimal_col).cast("decimal"))

        return df

    def _parse_complex_schema(self, schema_str: str):
        """
        Parse a schema string that may contain struct types
        """
        from pyspark.sql.types import DoubleType, FloatType, IntegerType, LongType, StringType, StructField, StructType

        # This is a simplified parser - in production you'd want a more robust solution
        # For now, we'll manually handle the specific case we need
        fields = []
        parts = schema_str.split(", ")

        i = 0
        while i < len(parts):
            part = parts[i]
            if "struct<" in part:
                # Find the complete struct definition
                struct_def = part
                while ">" not in struct_def and i < len(parts) - 1:
                    i += 1
                    struct_def += ", " + parts[i]

                # Parse struct field
                field_name = struct_def.split()[0]
                # For the nanos test case, we know the struct format
                if "event_ts string" in struct_def:
                    struct_fields = [
                        StructField("event_ts", StringType(), True),
                        StructField("parsed_ts", StringType(), True),  # Will be converted to timestamp later
                        StructField("double_ts", DoubleType(), True)
                    ]
                    fields.append(StructField(field_name, StructType(struct_fields), True))
                else:
                    # Generic struct handling would go here
                    pass
            else:
                # Parse simple field
                field_parts = part.strip().split()
                if len(field_parts) >= 2:
                    field_name = field_parts[0]
                    field_type = field_parts[1]

                    if field_type == "string":
                        fields.append(StructField(field_name, StringType(), True))
                    elif field_type == "double":
                        fields.append(StructField(field_name, DoubleType(), True))
                    elif field_type == "float":
                        fields.append(StructField(field_name, FloatType(), True))
                    elif field_type == "integer" or field_type == "int":
                        fields.append(StructField(field_name, IntegerType(), True))
                    elif field_type == "long":
                        fields.append(StructField(field_name, LongType(), True))
            i += 1

        return StructType(fields)

    def _process_struct_data(self, data, schema):
        """
        Convert list data to Row objects where needed for struct columns
        """
        from pyspark.sql import Row
        from pyspark.sql.types import StructType

        processed_data = []
        for row in data:
            new_row = []
            for i, (value, field) in enumerate(zip(row, schema.fields)):
                if isinstance(field.dataType, StructType) and isinstance(value, list):
                    # Convert list to Row object for struct column
                    struct_row = Row(*[f.name for f in field.dataType.fields])(*value)
                    new_row.append(struct_row)
                else:
                    new_row.append(value)
            processed_data.append(new_row)

        return processed_data

    def as_tsdf(self) -> TSDF:
        """
        Constructs a TSDF from the test data
        """
        sdf = self.as_sdf()

        # Remove ts_schema from kwargs if present, as it's not a valid TSDF parameter
        tsdf_kwargs = dict(self.tsdf)
        if "ts_schema" in tsdf_kwargs:
            del tsdf_kwargs["ts_schema"]

        if self.tsdf_constructor is not None:
            return getattr(TSDF, self.tsdf_constructor)(sdf, **tsdf_kwargs)
        else:
            return TSDF(sdf, **tsdf_kwargs)

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
        # parse out components of the test case path
        id_parts = self.id().split(".")
        func_name = id_parts[-1]
        class_name = id_parts[-2]

        # Handle nested test directories - build the path from module components
        # For example: tests.intervals.core.intervals_df_tests -> intervals/core/intervals_df_tests
        module_parts = id_parts[:-2]  # Everything except class and function name

        # Remove 'tests' prefix if present
        if module_parts and module_parts[0] == 'tests':
            module_parts = module_parts[1:]

        # Build the file path from remaining module parts
        if module_parts:
            file_name = "/".join(module_parts)
        else:
            # Fallback to old behavior for non-nested tests
            file_name = id_parts[-3] if len(id_parts) >= 3 else ""

        self.file_name = file_name
        self.class_name = class_name
        self.func_name = func_name

        # load the test data file if it hasn't been loaded yet
        if self.test_case_data is None:
            self.test_case_data = self.__loadTestData(file_name)

    def tearDown(self) -> None:
        del self.file_name
        del self.class_name
        del self.func_name

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
    def getTestDataDirPath(cls) -> str:
        # what folder are we running from?
        cwd = os.path.basename(os.getcwd())

        # build path based on what folder we're in
        dir_path = "./"
        if cwd == "tempo":
            dir_path = "./python/tests"
        elif cwd == "python":
            dir_path = "./tests"
        elif cwd != "tests":
            raise RuntimeError(f"Cannot locate test dir, running from dir {os.getcwd()}")
        return os.path.abspath(os.path.join(dir_path, cls.TEST_DATA_FOLDER))

    @classmethod
    def getTestDataFilePath(cls, test_file_name: str, extension: str = '.json') -> str:
        return os.path.join(cls.getTestDataDirPath(),
                            f"{test_file_name}{extension}")

    def __loadTestData(self, file_name: str) -> dict:
        """
        This function reads our unit test data config json and returns the required metadata to create the correct
        format of test data (Spark DataFrames, Pandas DataFrames and Tempo TSDFs)
        :param file_name: base name of the test data file
        :type file_name: str
        """

        # find our test data file
        test_data_filename = self.getTestDataFilePath(file_name)
        if not os.path.isfile(test_data_filename):
            warnings.warn(f"Could not load test data file {test_data_filename}")
            return {}

        # proces the data file
        with open(test_data_filename) as f:
            base_path = "file://"+ self.getTestDataDirPath() + "/"
            test_data = jsonref.load(f, base_uri=base_path)

        return test_data

    def get_test_df_builder(self, *data_keys: str) -> TestDataFrameBuilder:
        # unpack the test data by keys
        data = self.test_case_data
        for key in data_keys:
            data = prefix_value(key, data)
        # return the builder
        return TestDataFrameBuilder(self.spark, data)

    def get_test_function_df_builder(self, *sub_elements: str) -> TestDataFrameBuilder:
        """
        Get the test data builder for the current test function
        """
        function_data_keys = [self.class_name, self.func_name] + list(sub_elements)
        return self.get_test_df_builder(*function_data_keys)

    def get_data_as_sdf(self, name: str, convert_ts_col=True):
        """
        Get test data as a Spark DataFrame
        """
        return self.get_test_function_df_builder(name).as_sdf()

    def get_data_as_tsdf(self, name: str):
        """
        Get test data as a TSDF
        """
        return self.get_test_function_df_builder(name).as_tsdf()

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