from __future__ import annotations

import copy
import logging
import operator
from abc import ABCMeta, abstractmethod
from collections.abc import Collection, Iterable, Mapping, Sequence
from datetime import datetime as dt
from datetime import timedelta as td
from functools import cached_property
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast, overload

import numpy as np
import pandas as pd
import pyspark.sql.functions as sfn
from IPython.core.display import HTML  # type: ignore[import-not-found]
from IPython.display import display as ipydisplay  # type: ignore[import-not-found]
from pandas.core.frame import DataFrame as PandasDataFrame
from pyspark import RDD
from pyspark.sql import GroupedData, SparkSession
from pyspark.sql.column import Column
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import AtomicType, DataType, StructType
from pyspark.sql.window import Window, WindowSpec
from scipy.fft import fft, fftfreq

import tempo.io as t_io
import tempo.resample as t_resample
import tempo.resample_utils as t_resample_utils
import tempo.utils as t_utils
from tempo.intervals import IntervalsDF
from tempo.tsschema import (
    DEFAULT_TIMESTAMP_FORMAT,
    CompositeTSIndex,
    OrdinalTSIndex,
    ParsedTSIndex,
    SimpleTSIndex,
    TSIndex,
    TSSchema,
    WindowBuilder,
    identify_fractional_second_separator,
    is_time_format,
    sub_seconds_precision_digits,
)
from tempo.typing import ColumnOrName, PandasGroupedMapFunction, PandasMapIterFunction

logger = logging.getLogger(__name__)


# Helper functions


def make_struct_from_cols(
    df: DataFrame, struct_col_name: str, cols_to_move: List[str]
) -> DataFrame:
    """
    Transform a :class:`DataFrame` by moving certain columns into a named struct

    :param df: the :class:`DataFrame` to transform
    :param struct_col_name: name of the struct column to create
    :param cols_to_move: name of the columns to move into the struct

    :return: the transformed :class:`DataFrame`
    """
    return df.withColumn(struct_col_name, sfn.struct(*cols_to_move)).drop(*cols_to_move)


def time_str_to_double(
    df: DataFrame,
    ts_str_col: str,
    ts_dbl_col: str,
    ts_fmt: str = DEFAULT_TIMESTAMP_FORMAT,
) -> DataFrame:
    """
    Convert a string timestamp column to a double timestamp column

    :param df: the :class:`DataFrame` to transform
    :param ts_str_col: name of the string timestamp column
    :param ts_dbl_col: name of the double timestamp column to create
    :param fractional_seconds_split_char: the character to split fractional seconds on

    :return: the transformed :class:`DataFrame`
    """
    tmp_int_ts_col = "__tmp_int_ts"
    tmp_frac_ts_col = "__tmp_fract_ts"
    fract_secs_sep = identify_fractional_second_separator(ts_fmt)
    double_ts_df = (
        # get the interger part of the timestamp
        df.withColumn(tmp_int_ts_col, sfn.to_timestamp(ts_str_col, ts_fmt).cast("long"))
        # get the fractional part of the timestamp
        .withColumn(
            tmp_frac_ts_col,
            sfn.when(
                sfn.col(ts_str_col).contains(fract_secs_sep),
                sfn.concat(
                    sfn.lit("0."),
                    sfn.split(sfn.col(ts_str_col), f"\\{fract_secs_sep}")[1],
                ),
            )
            .otherwise(0.0)
            .cast("double"),
        )
        # combine them together
        .withColumn(ts_dbl_col, sfn.col(tmp_int_ts_col) + sfn.col(tmp_frac_ts_col))
        # clean up
        .drop(tmp_int_ts_col, tmp_frac_ts_col)
    )
    return double_ts_df


# The TSDF class


class TSDF(WindowBuilder):
    """
    This class represents a time series DataFrame (TSDF) - a DataFrame with a
    time series index. It can represent multiple logical time series,
    each identified by a unique set of series IDs.
    """

    summarizable_types = ["int", "bigint", "float", "double"]

    def __init__(
        self,
        df: DataFrame,
        ts_schema: Optional[TSSchema] = None,
        ts_col: Optional[str] = None,
        series_ids: Optional[Collection[str]] = None,
        resample_freq: Optional[str] = None,
        resample_func: Optional[Union[Callable, str]] = None,
    ) -> None:
        self.df = df
        # construct schema if we don't already have one
        if ts_schema:
            self.ts_schema = ts_schema
        else:
            assert ts_col is not None
            self.ts_schema = TSSchema.fromDFSchema(self.df.schema, ts_col, series_ids)
        # validate that this schema works for this DataFrame
        self.ts_schema.validate(df.schema)

        # Optional resample metadata (used when this TSDF is created from resample())
        self.resample_freq = resample_freq
        self.resample_func = resample_func

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(df={self.df}, ts_schema={self.ts_schema})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TSDF):
            return False
        return self.ts_schema == other.ts_schema and self.df == other.df

    def __withTransformedDF(self, new_df: DataFrame) -> TSDF:
        """
        This helper function will create a new :class:`TSDF` using the current schema, but a new / transformed :class:`DataFrame`

        :param new_df: the new / transformed :class:`DataFrame` to

        :return: a new TSDF object with the transformed DataFrame
        """
        return TSDF(
            new_df,
            ts_schema=copy.deepcopy(self.ts_schema),
            resample_freq=self.resample_freq,
            resample_func=self.resample_func,
        )

    def __withStandardizedColOrder(self) -> TSDF:
        """
        Standardizes the column ordering as such:
        * series_ids,
        * ts_index,
        * observation columns

        :return: a :class:`TSDF` with the columns reordered into
        "standard order" (as described above)
        """
        std_ordered_cols = (
            list(self.series_ids)
            + [self.ts_index.colname]
            + list(self.observational_cols)
        )

        return self.__withTransformedDF(self.df.select(std_ordered_cols))

    # default column name for constructed timeseries index struct columns
    __DEFAULT_TS_IDX_COL = "ts_idx"

    @classmethod
    def buildEmptyLattice(
        cls,
        spark: SparkSession,
        start_time: dt,
        end_time: Optional[dt] = None,
        step_size: Optional[td] = None,
        num_intervals: Optional[int] = None,
        ts_col: Optional[str] = None,
        series_ids: Optional[Any] = None,
        series_schema: Optional[Union[AtomicType, StructType, str]] = None,
        observation_cols: Optional[Union[Mapping[str, str], Iterable[str]]] = None,
        num_partitions: Optional[int] = None,
    ) -> TSDF:
        """
        Construct an empty "lattice", i.e. a :class:`TSDF` with a time range
        for each unique series and a set of observational columns (initialized to Nulls)

        :param spark: the Spark session to use
        :param start_time: the start time of the lattice
        :param end_time: the end time of the lattice (optional)
        :param step_size: the step size between each time interval (optional)
        :param num_intervals: the number of intervals to create (optional)
        :param ts_col: the name of the timestamp column (optional)
        :param series_ids: the unique series identifiers (optional)
        :param series_schema: the schema of the series identifiers (optional)
        :param observation_cols: the observational columns to include (optional)
        :param num_partitions: the number of partitions to create (optional)

        :return: a :class:`TSDF` representing the empty lattice
        """

        # set a default timestamp column if not provided
        if ts_col is None:
            ts_col = cls.__DEFAULT_TS_IDX_COL

        # initialize the lattice as a time range
        lattice_df = t_utils.time_range(
            spark, start_time, end_time, step_size, num_intervals, ts_colname=ts_col
        )
        select_exprs = [sfn.col(ts_col)]

        # handle construction of the series_ids DataFrame
        series_df = None
        if series_ids:
            if isinstance(series_ids, DataFrame):
                series_df = series_ids
            elif isinstance(series_ids, (RDD, PandasDataFrame)):
                series_df = spark.createDataFrame(series_ids)
            elif isinstance(series_ids, dict):
                series_df = spark.createDataFrame(pd.DataFrame(series_ids))
            else:
                series_df = spark.createDataFrame(data=series_ids, schema=series_schema)
            # add the series columns to the select expressions
            select_exprs += [sfn.col(c) for c in series_df.columns]
            # lattice is the cross join of the time range and the series identifiers
            lattice_df = lattice_df.crossJoin(series_df)

        # set up select expressions for the observation columns
        if observation_cols:
            # convert to a dict if not already, mapping all columns to "double" types
            if not isinstance(observation_cols, dict):
                observation_cols = {col: "double" for col in observation_cols}
            select_exprs += [
                sfn.lit(None).cast(coltype).alias(colname)
                for colname, coltype in observation_cols.items()
            ]
            lattice_df = lattice_df.select(*select_exprs)

        # repartition the lattice in a more optimal way
        if num_partitions is None:
            num_partitions = lattice_df.rdd.getNumPartitions()
        if series_df:
            sort_cols = series_df.columns + [ts_col]
            lattice_df = lattice_df.repartition(
                num_partitions, *(series_df.columns)
            ).sortWithinPartitions(*sort_cols)
        else:
            lattice_df = lattice_df.repartitionByRange(num_partitions, ts_col)

        # construct the appropriate TSDF
        return TSDF(lattice_df, ts_col=ts_col, series_ids=series_df.columns if series_df else None)

    @classmethod
    def fromSubsequenceCol(
        cls,
        df: DataFrame,
        ts_col: str,
        subsequence_col: str,
        series_ids: Optional[Collection[str]] = None,
    ) -> TSDF:
        # construct a struct with the ts_col and subsequence_col
        struct_col_name = cls.__DEFAULT_TS_IDX_COL
        with_subseq_struct_df = make_struct_from_cols(
            df, struct_col_name, [ts_col, subsequence_col]
        )
        # construct an appropriate TSIndex
        subseq_struct = with_subseq_struct_df.schema[struct_col_name]
        # For now, we'll use an OrdinalTSIndex with the struct column
        # This is a limitation that should be addressed in a future refactor
        subseq_idx = OrdinalTSIndex(subseq_struct)
        # construct & return the TSDF with appropriate schema
        return TSDF(with_subseq_struct_df, ts_schema=TSSchema(subseq_idx, series_ids))

    # default column name for parsed timeseries column
    __DEFAULT_PARSED_TS_COL = "parsed_ts"
    __DEFAULT_DOUBLE_TS_COL = "double_ts"

    @classmethod
    def fromStringTimestamp(
        cls,
        df: DataFrame,
        ts_col: str,
        series_ids: Optional[Collection[str]] = None,
        ts_fmt: str = DEFAULT_TIMESTAMP_FORMAT,
    ) -> TSDF:
        # TODO (v0.2 refactor): Fix timezone handling for nanosecond precision timestamps
        # When using composite timestamp indexes for nanosecond precision, there can be
        # timezone inconsistencies between different join strategies (broadcast vs union).
        # This should be addressed in the v0.2 refactor to ensure consistent behavior.

        # parse the ts_col based on the pattern
        is_sub_ms = False
        sub_ms_digits = 0
        if is_time_format(ts_fmt):
            # is this a sub-microsecond precision timestamp?
            sub_ms_digits = sub_seconds_precision_digits(ts_fmt)
            is_sub_ms = sub_ms_digits > 6
            # if the ts_fmt is a time format, we can use to_timestamp
            ts_expr = sfn.to_timestamp(sfn.col(ts_col), ts_fmt)
        else:
            # otherwise, we'll use to_date
            ts_expr = sfn.to_date(sfn.col(ts_col), ts_fmt)
        # parse the ts_col give the expression
        parsed_ts_col = cls.__DEFAULT_PARSED_TS_COL
        parsed_df = df.withColumn(parsed_ts_col, ts_expr)
        # parse a sub-microsecond precision timestamp to a double
        if is_sub_ms:
            # get the integer part of the timestamp
            parsed_df = time_str_to_double(
                parsed_df, ts_col, cls.__DEFAULT_DOUBLE_TS_COL, ts_fmt
            )
        # move the ts cols into a struct
        struct_col_name = cls.__DEFAULT_TS_IDX_COL
        cols_to_move = [ts_col, parsed_ts_col]
        if is_sub_ms:
            cols_to_move.append(cls.__DEFAULT_DOUBLE_TS_COL)
        with_parsed_struct_df = make_struct_from_cols(
            parsed_df, struct_col_name, cols_to_move
        )
        # construct an appropriate TSIndex
        parsed_struct = with_parsed_struct_df.schema[struct_col_name]
        if is_sub_ms:
            parsed_ts_idx = ParsedTSIndex.fromParsedTimestamp(
                parsed_struct,
                parsed_ts_col,
                ts_col,
                cls.__DEFAULT_DOUBLE_TS_COL,
                sub_ms_digits,
            )
        else:
            parsed_ts_idx = ParsedTSIndex.fromParsedTimestamp(
                parsed_struct, parsed_ts_col, ts_col
            )
        # construct & return the TSDF with appropriate schema
        return TSDF(
            with_parsed_struct_df, ts_schema=TSSchema(parsed_ts_idx, series_ids)
        )

    @property
    def ts_index(self) -> TSIndex:
        return self.ts_schema.ts_idx

    @property
    def ts_col(self) -> str:
        # TODO - this should be replaced TSIndex expressions
        return self.ts_schema.ts_idx.colname

    @property
    def columns(self) -> List[str]:
        return self.df.columns

    @property
    def series_ids(self) -> List[str]:
        return self.ts_schema.series_ids

    @property
    def structural_cols(self) -> List[str]:
        return self.ts_schema.structural_columns

    @cached_property
    def observational_cols(self) -> List[str]:
        return self.ts_schema.find_observational_columns(self.df.schema)

    @cached_property
    def metric_cols(self) -> List[str]:
        return self.ts_schema.find_metric_columns(self.df.schema)

    #
    # Helper functions
    #

    def __checkPartitionCols(self, tsdf_right: "TSDF") -> None:
        for left_col, right_col in zip(self.series_ids, tsdf_right.series_ids):
            if left_col != right_col:
                raise ValueError(
                    "left and right dataframe partition columns should have same name in same order"
                )

    def __validateTsColMatch(self, right_tsdf: "TSDF") -> None:
        # TODO - can simplify this to get types from schema object
        left_ts_datatype = self.df.select(self.ts_col).dtypes[0][1]
        right_ts_datatype = right_tsdf.df.select(right_tsdf.ts_col).dtypes[0][1]
        if left_ts_datatype != right_ts_datatype:
            raise ValueError(
                "left and right dataframe timestamp index columns should have same type"
            )

    def __addPrefixToColumns(self, col_list: list[str], prefix: str) -> TSDF:
        """
        Add prefix to all specified columns.
        """
        # no-op if no prefix
        if not prefix:
            return self

        # build a column rename map
        col_map = {col: "_".join([prefix, col]) for col in col_list}
        # TODO - In the future (when Spark 3.4+ is standard) we should implement batch rename using:
        # df = self.df.withColumnsRenamed(col_map)

        # build a list of column expressions to rename columns in a select
        select_exprs = [
            sfn.col(col).alias(col_map[col]) if col in col_map else sfn.col(col)
            for col in self.df.columns
        ]
        # select the renamed columns
        renamed_df = self.df.select(*select_exprs)

        # find the structural columns
        ts_col = col_map.get(self.ts_col, self.ts_col)
        partition_cols = [col_map.get(c, c) for c in self.series_ids]
        # sequence_col = col_map.get(self.sequence_col, self.sequence_col)
        # TODO: Handle sequence_col in the refactored version
        return TSDF(renamed_df, ts_col=ts_col, series_ids=partition_cols)

    def __addColumnsFromOtherDF(self, other_cols: Sequence[str]) -> TSDF:
        """
        Add columns from some other DF as lit(None), as pre-step before union.
        """

        # build a list of column expressions to rename columns in a select
        current_cols = [sfn.col(col) for col in self.df.columns]
        new_cols = [sfn.lit(None).alias(col) for col in other_cols]
        new_df = self.df.select(current_cols + new_cols)

        return self.__withTransformedDF(new_df)

    def __combineTSDF(self, ts_df_right: TSDF, combined_ts_col: str) -> TSDF:
        combined_df = self.df.unionByName(ts_df_right.df).withColumn(
            combined_ts_col, sfn.coalesce(self.ts_col, ts_df_right.ts_col)
        )

        return TSDF(combined_df, ts_col=combined_ts_col, series_ids=self.series_ids)

    def __getLastRightRow(
        self,
        left_ts_col: str,
        right_cols: list[str],
        sequence_col: Optional[str],
        tsPartitionVal: Optional[int],
        ignoreNulls: bool,
        suppress_null_warning: bool,
    ) -> TSDF:
        """Get last right value of each right column (inc. right timestamp) for each self.ts_col value

        self.ts_col, which is the combined time-stamp column of both left and right dataframe, is dropped at the end
        since it is no longer used in subsequent methods.
        """
        ptntl_sort_keys = [self.ts_col, "rec_ind", sequence_col]
        sort_keys = [sfn.col(col_name) for col_name in ptntl_sort_keys if col_name]

        window_spec = (
            Window.partitionBy(self.series_ids)
            .orderBy(sort_keys)
            .rowsBetween(Window.unboundedPreceding, Window.currentRow)
        )

        # generate expressions to find the last value of each right-hand column
        if ignoreNulls is False:
            if tsPartitionVal is not None:
                raise ValueError(
                    "Disabling null skipping with a partition value is not supported yet."
                )
            mod_right_cols = [
                sfn.last(
                    sfn.when(sfn.col("rec_ind") == -1, sfn.struct(col)).otherwise(None),
                    True,
                )
                .over(window_spec)[col]
                .alias(col)
                for col in right_cols
            ]
        elif tsPartitionVal is None:
            mod_right_cols = [
                sfn.last(col, ignoreNulls).over(window_spec).alias(col)
                for col in right_cols
            ]
        else:
            mod_right_cols = [
                sfn.last(col, ignoreNulls).over(window_spec).alias(col)
                for col in right_cols
            ]
            # non-null count columns, these will be dropped below
            mod_right_cols += [
                sfn.count(col).over(window_spec).alias("non_null_ct" + col)
                for col in right_cols
            ]

        # select the left-hand side columns, and the modified right-hand side columns
        non_right_cols = list(set(self.df.columns) - set(right_cols))
        df = self.df.select(non_right_cols + mod_right_cols)
        # drop the null left-hand side rows
        df = (df.filter(sfn.col(left_ts_col).isNotNull()).drop(self.ts_col)).drop(
            "rec_ind"
        )

        # remove the null_ct stats used to record missing values in partitioned as of join
        if tsPartitionVal is not None:
            for column in df.columns:
                if column.startswith("non_null"):
                    # Avoid collect() calls when explicitly ignoring the warnings about null values due to lookback
                    # window. if setting suppress_null_warning to True and warning logger is enabled for other part
                    # of the code, it would make sense to not log warning in this function while allowing other part
                    # of the code to continue to log warning. So it makes more sense for and than or on this line
                    if not suppress_null_warning and logger.isEnabledFor(
                        logging.WARNING
                    ):
                        any_blank_vals = df.agg({column: "min"}).collect()[0][0] == 0
                        newCol = column.replace("non_null_ct", "")
                        if any_blank_vals:
                            logger.warning(
                                "Column "
                                + newCol
                                + " had no values within the lookback window. Consider using a larger window to avoid missing values. If this is the first record in the data frame, this warning can be ignored."
                            )
                    df = df.drop(column)

        return TSDF(df, ts_col=left_ts_col, series_ids=self.series_ids)

    def __getTimePartitions(self, tsPartitionVal: int, fraction: float = 0.1) -> TSDF:
        """
        Create time-partitions for our data-set. We put our time-stamps into brackets of <tsPartitionVal>. Timestamps
        are rounded down to the nearest <tsPartitionVal> seconds.

        We cast our timestamp column to double instead of using f.unix_timestamp, since it provides more precision.

        Additionally, we make these partitions overlapping by adding a remainder df. This way when calculating the
        last right timestamp we will not end up with nulls for the first left timestamp in each partition.

        TODO: change ts_partition to accommodate for higher precision than seconds.
        """
        partition_df = (
            self.df.withColumn(
                "ts_col_double", sfn.col(self.ts_col).cast("double")
            )  # double is preferred over unix_timestamp
            .withColumn(
                "ts_partition",
                sfn.lit(tsPartitionVal)
                * (sfn.col("ts_col_double") / sfn.lit(tsPartitionVal)).cast("integer"),
            )
            .withColumn(
                "partition_remainder",
                (sfn.col("ts_col_double") - sfn.col("ts_partition"))
                / sfn.lit(tsPartitionVal),
            )
            .withColumn("is_original", sfn.lit(1))
        ).cache()  # cache it because it's used twice.

        # add [1 - fraction] of previous time partition to the next partition.
        remainder_df = (
            partition_df.filter(sfn.col("partition_remainder") >= sfn.lit(1 - fraction))
            .withColumn(
                "ts_partition", sfn.col("ts_partition") + sfn.lit(tsPartitionVal)
            )
            .withColumn("is_original", sfn.lit(0))
        )

        df = partition_df.union(remainder_df).drop(
            "partition_remainder", "ts_col_double"
        )
        return TSDF(
            df, ts_col=self.ts_col, series_ids=self.series_ids + ["ts_partition"]
        )

    #
    # Slicing & Selection
    #

    def select(self, *cols: Union[str, Column]) -> TSDF:
        """
        pyspark.sql.DataFrame.select() method's equivalent for TSDF objects
        Parameters
        ----------
        cols : str or list of strs
        column names (string).
        If one of the column names is '*', that column is expanded to include all columns
        in the current :class:`TSDF`.

        Examples
        --------
        tsdf.select('*').collect()
        [Row(age=2, name='Alice'), Row(age=5, name='Bob')]
        tsdf.select('name', 'age').collect()
        [Row(name='Alice', age=2), Row(name='Bob', age=5)]

        """
        # The columns which will be a mandatory requirement while selecting from TSDFs
        selected_df = self.df.select(*cols)
        return self.__withTransformedDF(selected_df)

    def where(self, condition: Union[Column, str]) -> TSDF:
        """
        Selects rows using the given condition.

        :param condition: a :class:`Column` of :class:`types.BooleanType` or a string of SQL expression.

        :return: a new :class:`TSDF` object
        :rtype: :class:`TSDF`
        """
        where_df = self.df.where(condition)
        return self.__withTransformedDF(where_df)

    def at(self, ts: Any) -> TSDF:
        """
        Select only records at a given time

        :param ts: timestamp of the records to select

        :return: a :class:`~tsdf.TSDF` object containing just the records at the given time
        """
        return self.where(self.ts_index == ts)

    def before(self, ts: Any) -> TSDF:
        """
        Select only records before a given time

        :param ts: timestamp on which to filter records

        :return: a :class:`~tsdf.TSDF` object containing just the records before the given time
        """
        return self.where(self.ts_index < ts)

    def atOrBefore(self, ts: Any) -> TSDF:
        """
        Select only records at or before a given time

        :param ts: timestamp on which to filter records

        :return: a :class:`~tsdf.TSDF` object containing just the records at or before the given time
        """
        return self.where(self.ts_index <= ts)

    def after(self, ts: Any) -> TSDF:
        """
        Select only records after a given time

        :param ts: timestamp on which to filter records

        :return: a :class:`~tsdf.TSDF` object containing just the records after the given time
        """
        return self.where(self.ts_index > ts)

    def atOrAfter(self, ts: Any) -> TSDF:
        """
        Select only records at or after a given time

        :param ts: timestamp on which to filter records

        :return: a :class:`~tsdf.TSDF` object containing just the records at or after the given time
        """
        return self.where(self.ts_index >= ts)

    def between(self, start_ts: Any, end_ts: Any, inclusive: bool = True) -> TSDF:
        """
        Select only records in a given range

        :param start_ts: starting time of the range to select
        :param end_ts: ending time of the range to select
        :param inclusive: whether the range is inclusive of the endpoints or not, defaults to True
        :type inclusive: bool

        :return: a :class:`~tsdf.TSDF` object containing just the records within the range specified
        """
        if inclusive:
            return self.atOrAfter(start_ts).atOrBefore(end_ts)
        return self.after(start_ts).before(end_ts)

    def __top_rows_per_series(self, win: WindowSpec, n: int) -> TSDF:
        """
        Private method to select just the top n rows per series (as defined by a window ordering)

        :param win: the window on which we order the rows in each series
        :param n: the number of rows to return

        :return: a :class:`~tsdf.TSDF` object containing just the top n rows in each series
        """
        row_num_col = "__row_num"
        prev_records_df = (
            self.df.withColumn(row_num_col, sfn.row_number().over(win))
            .where(sfn.col(row_num_col) <= sfn.lit(n))
            .drop(row_num_col)
        )
        return self.__withTransformedDF(prev_records_df)

    def earliest(self, n: int = 1) -> TSDF:
        """
        Select the earliest n records for each series

        :param n: number of records to select (default is 1)

        :return: a :class:`~tsdf.TSDF` object containing the earliest n records for each series
        """
        prev_window = self.baseWindow(reverse=False)
        return self.__top_rows_per_series(prev_window, n)

    def latest(self, n: int = 1) -> TSDF:
        """
        Select the latest n records for each series

        :param n: number of records to select (default is 1)

        :return: a :class:`~tsdf.TSDF` object containing the latest n records for each series
        """
        next_window = self.baseWindow(reverse=True)
        return self.__top_rows_per_series(next_window, n)

    def priorTo(self, ts: Any, n: int = 1) -> TSDF:
        """
        Select the n most recent records prior to a given time
        You can think of this like an 'asOf' select - it selects the records as of a particular time

        :param ts: timestamp on which to filter records
        :param n: number of records to select (default is 1)

        :return: a :class:`~tsdf.TSDF` object containing the n records prior to the given time
        """
        return self.atOrBefore(ts).latest(n)

    def subsequentTo(self, ts: Any, n: int = 1) -> TSDF:
        """
        Select the n records subsequent to a give time

        :param ts: timestamp on which to filter records
        :param n: number of records to select (default is 1)

        :return: a :class:`~tsdf.TSDF` object containing the n records subsequent to the given time
        """
        return self.atOrAfter(ts).earliest(n)

    #
    # Display functions
    #

    def show(
        self, n: int = 20, k: int = 5, truncate: bool = True, vertical: bool = False
    ) -> None:
        """
        pyspark.sql.DataFrame.show() method's equivalent for TSDF objects

        Parameters
        ----------
        n : int, optional
        Number of rows to show.
        truncate : bool or int, optional
        If set to ``True``, truncate strings longer than 20 chars by default.
        If set to a number greater than one, truncates long strings to length ``truncate``
        and align cells right.
        vertical : bool, optional
        If set to ``True``, print output rows vertically (one line
        per column value).

        Example to show usage
        ---------------------
        from pyspark.sql.functions import *

        phone_accel_df = spark.read.format("csv").option("header", "true").load("dbfs:/home/tempo/Phones_accelerometer").withColumn("event_ts", (col("Arrival_Time").cast("double")/1000).cast("timestamp")).withColumn("x", col("x").cast("double")).withColumn("y", col("y").cast("double")).withColumn("z", col("z").cast("double")).withColumn("event_ts_dbl", col("event_ts").cast("double"))

        from tempo import *

        phone_accel_tsdf = TSDF(phone_accel_df, ts_col="event_ts", partition_cols = ["User"])

        # Call show method here
        phone_accel_tsdf.show()

        """
        # validate k <= n
        if k > n:
            raise ValueError(f"Parameter k {k} cannot be greater than parameter n {n}")

        if not t_utils.IS_DATABRICKS and t_utils.ENV_CAN_RENDER_HTML:
            # In Jupyter notebooks, for wide dataframes the below line will enable
            # rendering the output in a scrollable format.
            ipydisplay(
                HTML("<style>pre { white-space: pre !important; }</style>")
            )  # pragma: no cover
        # t_utils.get_display_df(self, k).show(n, truncate, vertical)
        self.df.show(n, truncate, vertical)

    # def describe(self) -> DataFrame:
    #     """
    #     Describe a TSDF object using a global summary across all time series (anywhere from 10 to millions) as well as the standard Spark data frame stats. Missing vals
    #     Summary
    #     global - unique time series based on partition columns, min/max times, granularity - lowest precision in the time series timestamp column
    #     count / mean / stddev / min / max - standard Spark data frame describe() output
    #     missing_vals_pct - percentage (from 0 to 100) of missing values.
    #     """
    #     # extract the double version of the timestamp column to summarize
    #     double_ts_col = self.ts_col + "_dbl"
    #
    #     this_df = self.df.withColumn(double_ts_col, sfn.col(self.ts_col).cast("double"))
    #
    #     # summary missing value percentages
    #     missing_vals = this_df.select(
    #         [
    #             (
    #                 100
    #                 * sfn.count(sfn.when(sfn.col(c[0]).isNull(), c[0]))
    #                 / sfn.count(sfn.lit(1))
    #             ).alias(c[0])
    #             for c in this_df.dtypes
    #             if c[1] != "timestamp"
    #         ]
    #     ).select(sfn.lit("missing_vals_pct").alias("summary"), "*")
    #
    #     # describe stats
    #     desc_stats = this_df.describe().union(missing_vals)
    #     unique_ts = this_df.select(*self.series_ids).distinct().count()
    #
    #     max_ts = this_df.select(
    #         sfn.max(sfn.col(self.ts_col)).alias("max_ts")
    #     ).collect()[0][0]
    #     min_ts = this_df.select(
    #         sfn.min(sfn.col(self.ts_col)).alias("max_ts")
    #     ).collect()[0][0]
    #     gran = this_df.selectExpr(
    #         """min(case when {0} - cast({0} as integer) > 0 then '1-millis'
    #               when {0} % 60 != 0 then '2-seconds'
    #               when {0} % 3600 != 0 then '3-minutes'
    #               when {0} % 86400 != 0 then '4-hours'
    #               else '5-days' end) granularity""".format(
    #             double_ts_col
    #         )
    #     ).collect()[0][0][2:]
    #
    #     non_summary_cols = [c for c in desc_stats.columns if c != "summary"]
    #
    #     desc_stats = desc_stats.select(
    #         sfn.col("summary"),
    #         sfn.lit(" ").alias("unique_ts_count"),
    #         sfn.lit(" ").alias("min_ts"),
    #         sfn.lit(" ").alias("max_ts"),
    #         sfn.lit(" ").alias("granularity"),
    #         *non_summary_cols,
    #     )
    #
    #     # add in single record with global summary attributes and the previously computed missing value and Spark data frame describe stats
    #     global_smry_rec = desc_stats.limit(1).select(
    #         sfn.lit("global").alias("summary"),
    #         sfn.lit(unique_ts).alias("unique_ts_count"),
    #         sfn.lit(min_ts).alias("min_ts"),
    #         sfn.lit(max_ts).alias("max_ts"),
    #         sfn.lit(gran).alias("granularity"),
    #         *[sfn.lit(" ").alias(c) for c in non_summary_cols],
    #     )
    #
    #     full_smry = global_smry_rec.union(desc_stats)
    #     full_smry = full_smry.withColumnRenamed(
    #         "unique_ts_count", "unique_time_series_count"
    #     )
    #
    #     try:  # pragma: no cover
    #         dbutils.fs.ls("/")  # type: ignore
    #         return full_smry
    #     # TODO: Can we raise something other than generic Exception?
    #     #  perhaps refactor to check for IS_DATABRICKS
    #     except Exception:
    #         return full_smry

    def __getSparkPlan(self, df: DataFrame, spark: SparkSession) -> str:
        """
        Internal helper function to obtain the Spark plan for the input data frame

        Parameters
        :param df - input Spark data frame - the AS OF join has 2 data frames; this will be called for each
        :param spark - Spark session which is used to query the view obtained from the Spark data frame
        """

        df.createOrReplaceTempView("view")
        plan = spark.sql("explain cost select * from view").collect()[0][0]

        return plan

    def __getBytesFromPlan(self, df: DataFrame, spark: SparkSession) -> float:
        """
        Internal helper function to obtain how many bytes in memory the Spark data
        frame is likely to take up. This is an upper bound and is obtained from the
        plan details in Spark

        Parameters
        :param df - input Spark data frame - the AS OF join has 2 data frames; this will be called for each
        :param spark - Spark session which is used to query the view obtained from the Spark data frame
        """

        plan = self.__getSparkPlan(df, spark)

        import re

        search_result = re.search(r"sizeInBytes=.*(['\)])", plan, re.MULTILINE)
        if search_result is not None:
            result = search_result.group(0).replace(")", "")
        else:
            raise ValueError("Unable to obtain sizeInBytes from Spark plan")

        size = result.split("=")[1].split(" ")[0]
        units = result.split("=")[1].split(" ")[1]

        # perform to MB for threshold check
        if units == "GiB":
            plan_bytes = float(size) * 1024 * 1024 * 1024
        elif units == "MiB":
            plan_bytes = float(size) * 1024 * 1024
        elif units == "KiB":
            plan_bytes = float(size) * 1024
        else:
            plan_bytes = float(size)

        return plan_bytes

    def asofJoin(
        self,
        right_tsdf: TSDF,
        left_prefix: Optional[str] = None,
        right_prefix: str = "right",
        tsPartitionVal: Optional[int] = None,
        fraction: float = 0.5,
        skipNulls: bool = True,
        sql_join_opt: bool = False,
        suppress_null_warning: bool = False,
        tolerance: Optional[int] = None,
    ) -> TSDF:
        """
        Performs an as-of join between two time-series. If a tsPartitionVal is
        specified, it will do this partitioned by time brackets, which can help alleviate skew.

        NOTE: partition cols have to be the same for both Dataframes. We are
        collecting stats when the WARNING level is enabled also.

        Parameters
        :param right_tsdf - right-hand data frame containing columns to merge in
        :param left_prefix - optional prefix for base data frame
        :param right_prefix - optional prefix for right-hand data frame
        :param tsPartitionVal - value to break up each partition into time brackets
        :param fraction - overlap fraction
        :param skipNulls - whether to skip nulls when joining in values
        :param sql_join_opt - if set to True, will use standard Spark SQL join if it is estimated to be efficient
        :param suppress_null_warning - when tsPartitionVal is specified, will collect min of each column and raise warnings about null values, set to True to avoid
        :param tolerance - only join values within this tolerance range (inclusive), expressed in number of seconds as a double
        """

        # first block of logic checks whether a standard range join will suffice
        left_df = self.df
        right_df = right_tsdf.df

        # test if the broadcast join will be efficient
        if sql_join_opt:
            spark = SparkSession.builder.getOrCreate()
            left_bytes = self.__getBytesFromPlan(left_df, spark)
            right_bytes = self.__getBytesFromPlan(right_df, spark)

            # choose 30MB as the cutoff for the broadcast
            bytes_threshold = 30 * 1024 * 1024
            if (left_bytes < bytes_threshold) or (right_bytes < bytes_threshold):
                spark.conf.set("spark.databricks.optimizer.rangeJoin.binSize", 60)
                partition_cols = right_tsdf.series_ids
                left_cols = list(set(left_df.columns) - set(self.series_ids))
                right_cols = list(set(right_df.columns) - set(right_tsdf.series_ids))

                left_prefix = left_prefix + "_" if left_prefix else ""
                right_prefix = right_prefix + "_" if right_prefix else ""

                w = Window.partitionBy(*partition_cols).orderBy(
                    right_prefix + right_tsdf.ts_col
                )

                new_left_ts_col = left_prefix + self.ts_col
                series_cols = [sfn.col(c) for c in self.series_ids]
                new_left_cols = [
                    sfn.col(c).alias(left_prefix + c) for c in left_cols
                ] + series_cols
                new_right_cols = [
                    sfn.col(c).alias(right_prefix + c) for c in right_cols
                ] + series_cols
                quotes_df_w_lag = right_df.select(*new_right_cols).withColumn(
                    "lead_" + right_tsdf.ts_col,
                    sfn.lead(right_prefix + right_tsdf.ts_col).over(w),
                )
                left_df = left_df.select(*new_left_cols)
                res = (
                    left_df.join(quotes_df_w_lag, partition_cols)
                    .where(
                        left_df[new_left_ts_col].between(
                            sfn.col(right_prefix + right_tsdf.ts_col),
                            sfn.coalesce(
                                sfn.col("lead_" + right_tsdf.ts_col),
                                sfn.lit("2099-01-01").cast("timestamp"),
                            ),
                        )
                    )
                    .drop("lead_" + right_tsdf.ts_col)
                )
                return TSDF(res, ts_col=new_left_ts_col, series_ids=self.series_ids)

        # end of block checking to see if standard Spark SQL join will work

        if tsPartitionVal is not None:
            logger.warning(
                "You are using the skew version of the AS OF join. This may result in null values if there are any "
                "values outside of the maximum lookback. For maximum efficiency, choose smaller values of maximum "
                "lookback, trading off performance and potential blank AS OF values for sparse keys"
            )

        # Check whether partition columns have same name in both dataframes
        self.__checkPartitionCols(right_tsdf)

        # prefix non-partition columns, to avoid duplicated columns.
        left_df = self.df
        right_df = right_tsdf.df

        # validate timestamp datatypes match
        self.__validateTsColMatch(right_tsdf)

        orig_left_col_diff = list(set(left_df.columns) - set(self.series_ids))
        orig_right_col_diff = list(set(right_df.columns) - set(self.series_ids))

        left_tsdf = (
            (self.__addPrefixToColumns([self.ts_col] + orig_left_col_diff, left_prefix))
            if left_prefix is not None
            else self
        )
        right_tsdf = right_tsdf.__addPrefixToColumns(
            [right_tsdf.ts_col] + orig_right_col_diff, right_prefix
        )

        left_columns = list(set(left_tsdf.df.columns).difference(set(self.series_ids)))
        right_columns = list(
            set(right_tsdf.df.columns).difference(set(self.series_ids))
        )

        # Union both dataframes, and create a combined TS column
        combined_ts_col = "combined_ts"
        combined_df = left_tsdf.__addColumnsFromOtherDF(right_columns).__combineTSDF(
            right_tsdf.__addColumnsFromOtherDF(left_columns), combined_ts_col
        )
        combined_df.df = combined_df.df.withColumn(
            "rec_ind",
            sfn.when(sfn.col(left_tsdf.ts_col).isNotNull(), 1).otherwise(-1),
        )

        # perform asof join.
        if tsPartitionVal is None:
            seq_col = None
            if isinstance(combined_df.ts_index, CompositeTSIndex):
                comp_idx = combined_df.ts_index
                if len(comp_idx.component_fields) > 1:
                    seq_col = comp_idx.fieldPath(comp_idx.component_fields[1])
            asofDF = combined_df.__getLastRightRow(
                left_tsdf.ts_col,
                right_columns,
                seq_col,
                tsPartitionVal,
                skipNulls,
                suppress_null_warning,
            )
        else:
            tsPartitionDF = combined_df.__getTimePartitions(
                tsPartitionVal, fraction=fraction
            )
            seq_col = None
            if isinstance(tsPartitionDF.ts_index, CompositeTSIndex):
                comp_idx = tsPartitionDF.ts_index
                if len(comp_idx.component_fields) > 1:
                    seq_col = comp_idx.fieldPath(comp_idx.component_fields[1])
            asofDF = tsPartitionDF.__getLastRightRow(
                left_tsdf.ts_col,
                right_columns,
                seq_col,
                tsPartitionVal,
                skipNulls,
                suppress_null_warning,
            )

            # Get rid of overlapped data and the extra columns generated from timePartitions
            df = asofDF.df.filter(sfn.col("is_original") == 1).drop(
                "ts_partition", "is_original"
            )

            asofDF = TSDF(df, ts_col=asofDF.ts_col, series_ids=combined_df.series_ids)

        if tolerance is not None:
            df = asofDF.df
            left_ts_col = left_tsdf.ts_col
            right_ts_col = right_tsdf.ts_col
            tolerance_condition = (
                df[left_ts_col].cast("double") - df[right_ts_col].cast("double")
                > tolerance
            )

            for right_col in right_columns:
                # First set right non-timestamp columns to null for rows outside of tolerance band
                if right_col != right_ts_col:
                    df = df.withColumn(
                        right_col,
                        sfn.when(tolerance_condition, sfn.lit(None)).otherwise(
                            df[right_col]
                        ),
                    )

            # Finally, set right timestamp column to null for rows outside of tolerance band
            df = df.withColumn(
                right_ts_col,
                sfn.when(tolerance_condition, sfn.lit(None)).otherwise(
                    df[right_ts_col]
                ),
            )
            asofDF.df = df

        return asofDF

    def baseWindow(self, reverse: bool = False) -> WindowSpec:
        return self.ts_schema.baseWindow(reverse=reverse)

    def rowsBetweenWindow(
        self, start: int, end: int, reverse: bool = False
    ) -> WindowSpec:
        return self.ts_schema.rowsBetweenWindow(start, end, reverse=reverse)

    def rangeBetweenWindow(
        self, start: int, end: int, reverse: bool = False
    ) -> WindowSpec:
        return self.ts_schema.rangeBetweenWindow(start, end, reverse=reverse)

    #
    # Re-Partitioning
    #

    def repartitionBySeries(self, numPartitions: Optional[int] = None) -> TSDF:
        """
        Repartition the data frame by series id(s) into a given number of partitions.

        :param numPartitions: number of partitions to repartition the data frame into

        :return: a new :class:`~tsdf.TSDF` object with the data frame repartitioned
        """

        # only makes sense if we have series ids
        assert (
            self.series_ids and len(self.series_ids) > 0
        ), "No series ids to repartition by"

        # keep same number of partitions if not specified
        if numPartitions is None:
            numPartitions = self.df.rdd.getNumPartitions()

        # repartition by series ids, ordering by time
        repartitioned_df = self.df.repartition(
            numPartitions, *self.series_ids
        ).sortWithinPartitions(*[self.series_ids + [self.ts_index.orderByExpr()]])
        return self.__withTransformedDF(repartitioned_df)

    def repartitionByTime(self, numPartitions: Optional[int] = None) -> TSDF:
        """
        Repartition the data frame by time into a given number of partitions.

        :param numPartitions: number of partitions to repartition the data frame into

        :return: a new :class:`~tsdf.TSDF` object with the data frame repartitioned
        """
        if numPartitions is None:
            numPartitions = self.df.rdd.getNumPartitions()

        repartitioned_df = self.df.repartitionByRange(
            numPartitions, self.ts_index.orderByExpr()
        )
        return self.__withTransformedDF(repartitioned_df)

    #
    # Core Transformations
    #

    def withNaturalOrdering(self, reverse: bool = False) -> TSDF:
        order_expr = [sfn.col(c) for c in self.series_ids]
        ts_idx_expr = self.ts_index.orderByExpr(reverse)
        if isinstance(ts_idx_expr, list):
            order_expr.extend(ts_idx_expr)
        else:
            order_expr.append(ts_idx_expr)

        return self.__withTransformedDF(self.df.orderBy(order_expr))

    def withColumn(self, colName: str, col: Column) -> TSDF:
        """
        Returns a new :class:`TSDF` by adding a column or replacing the
        existing column that has the same name.

        :param colName: the name of the new column (or existing column to be replaced)
        :param col: a :class:`Column` expression for the new column definition
        """
        new_df = self.df.withColumn(colName, col)
        return self.__withTransformedDF(new_df)

    def withColumnRenamed(self, existing: str, new: str) -> TSDF:
        """
        Returns a new :class:`TSDF` with the given column renamed.

        :param existing: name of the existing column to renmame
        :param new: new name for the column
        """

        # create new TSIndex
        new_ts_index = copy.deepcopy(self.ts_index)
        if existing == self.ts_index.colname:
            new_ts_index = new_ts_index.renamed(new)

        # and for series ids
        new_series_ids = self.series_ids
        if existing in self.series_ids:
            # replace column name in series
            new_series_ids = self.series_ids
            new_series_ids[new_series_ids.index(existing)] = new

        # rename the column in the underlying DF
        new_df = self.df.withColumnRenamed(existing, new)

        # return new TSDF
        new_schema = TSSchema(new_ts_index, new_series_ids)
        return TSDF(new_df, ts_schema=new_schema)

    def withColumnTypeChanged(self, colName: str, newType: Union[DataType, str]) -> TSDF:
        """

        :param colName:
        :param newType:
        :return:
        """
        new_df = self.df.withColumn(colName, sfn.col(colName).cast(newType))
        return self.__withTransformedDF(new_df)

    def drop(self, *cols: ColumnOrName) -> TSDF:
        """
        Returns a new :class:`TSDF` that drops the specified column.

        :param cols: name of the column to drop

        :return: new :class:`TSDF` with the column dropped
        :rtype: TSDF
        """
        dropped_df = self.df.drop(*cols)
        return self.__withTransformedDF(dropped_df)

    def mapInPandas(
        self, func: PandasMapIterFunction, schema: Union[StructType, str]
    ) -> TSDF:
        """

        :param func:
        :param schema:
        :return:
        """
        mapped_df = self.df.mapInPandas(func, schema)
        return self.__withTransformedDF(mapped_df)

    def union(self, other: TSDF) -> TSDF:
        # union of the underlying DataFrames
        union_df = self.df.union(other.df)
        return self.__withTransformedDF(union_df)

    def unionByName(self, other: TSDF, allowMissingColumns: bool = False) -> TSDF:
        # union of the underlying DataFrames
        union_df = self.df.unionByName(
            other.df, allowMissingColumns=allowMissingColumns
        )
        return self.__withTransformedDF(union_df)

    #
    # Rolling (Windowed) Transformations
    #

    def rollingAgg(
        self, window: WindowSpec, *exprs: Union[Column, Dict[str, str]]
    ) -> TSDF:
        """

        :param window:
        :param exprs:
        :return:
        """
        roll_agg_tsdf = self
        if len(exprs) == 1 and isinstance(exprs[0], dict):
            # dict
            for input_col in exprs[0].keys():
                expr_str = exprs[0][input_col]
                new_col_name = f"{expr_str}({input_col})"
                roll_agg_tsdf = roll_agg_tsdf.withColumn(
                    new_col_name, sfn.expr(expr_str).over(window)
                )
        else:
            # Columns
            assert all(
                isinstance(c, Column) for c in exprs
            ), "all exprs should be Column"
            for expr in exprs:
                if isinstance(expr, Column):
                    new_col_name = f"{expr}"
                    roll_agg_tsdf = roll_agg_tsdf.withColumn(
                        new_col_name, expr.over(window)
                    )

        return roll_agg_tsdf

    def rollingApply(
        self,
        outputCol: str,
        window: WindowSpec,
        func: PandasGroupedMapFunction,
        schema: Union[StructType, str],
        *inputCols: Union[str, Column],
    ) -> TSDF:
        """

        :param outputCol:
        :param window:
        :param func:
        :param schema:
        :param inputCols:
        :return:
        """
        inputCols_list = [sfn.col(col) if not isinstance(col, Column) else col for col in inputCols]
        pd_udf = sfn.pandas_udf(func, schema)
        return self.withColumn(outputCol, pd_udf(*inputCols_list).over(window))

    #
    # Aggregations
    #

    # Aggregations across series and time

    def summarize(self, *cols: Union[str, List[str]]) -> GroupedData:
        """
        Groups the underlying :class:`DataFrame` such that the user can compute
        aggregations over the given columns.
        If no columns are specified, all metric columns will be assumed.

        :param cols: columns to summarize. If none are given,
        then all the `metric_cols` will be used
        :type cols: str or List[str]
        :return: a :class:`GroupedData` object that can be used for
        summarizing columns/metrics across all observations from all series
        :rtype: :class:`GroupedData`
        """
        if len(cols) < 1:
            cols_list = self.metric_cols
        else:
            # Flatten the cols tuple which can contain strings and lists
            cols_list = []
            for col in cols:
                if isinstance(col, list):
                    cols_list.extend(col)
                else:
                    cols_list.append(col)
        return self.df.select(cols_list).groupBy()

    def agg(self, *exprs: Union[Column, Dict[str, str]]) -> DataFrame:
        """

        :param exprs:
        :return:
        """
        return self.df.agg(exprs)

    def describe(self, *cols: Union[str, List[str]]) -> DataFrame:
        """

        :param cols:
        :return:
        """
        if len(cols) < 1:
            cols_list = self.metric_cols
        else:
            # Flatten the cols tuple which can contain strings and lists
            cols_list = []
            for col in cols:
                if isinstance(col, list):
                    cols_list.extend(col)
                else:
                    cols_list.append(col)
        return self.df.describe(cols_list)

    def metricSummary(self, *statistics: str) -> DataFrame:
        """

        :param statistics:
        :return:
        """
        return self.df.select(self.metric_cols).summary(statistics)

    # Aggregations by series

    def groupBySeries(self) -> GroupedData:
        """
        Groups the underlying :class:`DataFrame` by the series IDs

        :return: a :class:`GroupedData` object that can be used for
        aggregating within Series
        :rtype: :class:`GroupedData`
        """
        return self.df.groupBy(self.series_ids)

    def aggBySeries(self, *exprs: Union[Column, Dict[str, str]]) -> DataFrame:
        """
        Compute aggregates of each series.

        :param exprs: a dict mapping from column name (string) to aggregate functions (string),
        or a list of :class:`Column`.
        :return: a :class:`DataFrame` of the resulting aggregates
        :rtype: :class:`DataFrame`
        """
        return self.groupBySeries().agg(exprs)

    def applyToSeries(
        self, func: PandasGroupedMapFunction, schema: Union[StructType, str]
    ) -> DataFrame:
        """
        Maps each series using a pandas udf and returns the result as a `DataFrame`.

        The function should take a `pandas.DataFrame` and return another
        `pandas.DataFrame`. Alternatively, the user can pass a function that takes
        a tuple of the grouping key(s) and a `pandas.DataFrame`.
        For each group, all columns are passed together as a `pandas.DataFrame`
        to the user-function and the returned `pandas.DataFrame` are combined as a
        :class:`DataFrame`.

        The `schema` should be a :class:`StructType` describing the schema of the returned
        `pandas.DataFrame`. The column labels of the returned `pandas.DataFrame` must either match
        the field names in the defined schema if specified as strings, or match the
        field data types by position if not strings, e.g. integer indices.
        The length of the returned `pandas.DataFrame` can be arbitrary.

        :param func: a Python native function that takes a `pandas.DataFrame` and outputs a
        `pandas.DataFrame`, or that takes one tuple (grouping keys) and a
        `pandas.DataFrame` and outputs a `pandas.DataFrame`.
        :type func: function
        :param schema: the return type of the `func` in PySpark. The value can be either a
        :class:`pyspark.sql.types.DataType` object or a DDL-formatted type string.
        :type schema: :class:`pyspark.sql.types.DataType` or str
        :return: a :class:`pyspark.sql.DataFrame` (of the given schema)
        containing the results of applying the given function per series
        :rtype: :class:`pyspark.sql.DataFrame`
        """
        return self.groupBySeries().applyInPandas(func, schema)

    # Cyclical Aggregtion

    def groupByCycles(
        self,
        length: str,
        period: Optional[str] = None,
        offset: Optional[str] = None,
        bySeries: bool = True,
    ) -> GroupedData:
        """

        :param length:
        :param period:
        :param offset:
        :param bySeries:
        :return:
        """
        # build our set of grouping columns
        if bySeries:
            grouping_cols = [sfn.col(series_col) for series_col in self.series_ids]
        else:
            grouping_cols = []
        grouping_cols.append(
            sfn.window(
                timeColumn=self.ts_col,
                windowDuration=length,
                slideDuration=period,
                startTime=offset,
            )
        )

        # return the DataFrame grouped accordingly
        return self.df.groupBy(grouping_cols)

    def aggByCycles(
        self,
        length: str,
        *exprs: Union[Column, Dict[str, str]],
        period: Optional[str] = None,
        offset: Optional[str] = None,
        bySeries: bool = True,
    ) -> IntervalsDF:
        """

        :param length:
        :param exprs:
        :param period:
        :param offset:
        :param bySeries:
        :return:
        """
        # build aggregated DataFrame
        agged_df = self.groupByCycles(length, period, offset, bySeries).agg(exprs)

        # if we have aggregated over series, we return a TSDF without series
        if bySeries:
            return IntervalsDF.fromNestedBoundariesDF(
                agged_df, "window", self.series_ids
            )
        else:
            return IntervalsDF.fromNestedBoundariesDF(agged_df, "window")

    def applyToCycles(
        self,
        length: str,
        func: PandasGroupedMapFunction,
        schema: Union[StructType, str],
        period: Optional[str] = None,
        offset: Optional[str] = None,
        bySeries: bool = True,
    ) -> IntervalsDF:
        """

        :param length:
        :param func:
        :param schema:
        :param period:
        :param offset:
        :param bySeries:
        :return:
        """
        # apply function to get DataFrame of results
        applied_df = self.groupByCycles(length, period, offset, bySeries).applyInPandas(
            func, schema
        )

        # if we have applied over series, we return a TSDF without series
        if bySeries:
            return IntervalsDF.fromNestedBoundariesDF(
                applied_df, "window", self.series_ids
            )
        else:
            return IntervalsDF.fromNestedBoundariesDF(applied_df, "window")

    #
    # utility functions
    #

    def write(
        self,
        spark: SparkSession,
        tabName: str,
        optimizationCols: Optional[List[str]] = None,
    ) -> None:
        t_io.write(self, spark, tabName, optimizationCols)

    def resample(
        self,
        freq: str,
        func: Union[Callable | str],
        metricCols: Optional[List[str]] = None,
        prefix: Optional[str] = None,
        fill: Optional[bool] = None,
        perform_checks: bool = True,
    ) -> TSDF:
        """
        function to upsample based on frequency and aggregate function similar to pandas
        :param freq: frequency for upsample - valid inputs are "hr", "min", "sec" corresponding to hour, minute, or second
        :param func: function used to aggregate input
        :param metricCols supply a smaller list of numeric columns if the entire set of numeric columns should not be returned for the resample function
        :param prefix - supply a prefix for the newly sampled columns
        :param fill - Boolean - set to True if the desired output should contain filled in gaps (with 0s currently)
        :param perform_checks: calculate time horizon and warnings if True (default is True)
        :return: TSDF object with sample data using aggregate function
        """
        t_resample_utils.validateFuncExists(func)

        # Throw warning for user to validate that the expected number of output rows is valid.
        if fill is True and perform_checks is True:
            t_resample.calculate_time_horizon(self, freq)

        enriched_df: DataFrame = t_resample.aggregate(
            self, freq, func, metricCols, prefix, fill
        )
        return TSDF(
            enriched_df,
            ts_schema=copy.deepcopy(self.ts_schema),
            resample_freq=freq,
            resample_func=func,
        )

    def interpolate(
        self,
        method: str,
        freq: Optional[str] = None,
        func: Optional[Union[Callable | str]] = None,
        target_cols: Optional[List[str]] = None,
        ts_col: Optional[str] = None,
        partition_cols: Optional[List[str]] = None,
        show_interpolated: bool = False,
        perform_checks: bool = True,
    ) -> TSDF:
        """
        Function to interpolate based on frequency, aggregation, and fill similar to pandas. Data will first be aggregated using resample, then missing values
        will be filled based on the fill calculation.

        :param freq: frequency for upsample - valid inputs are "hr", "min", "sec" corresponding to hour, minute, or second
        :param func: function used to aggregate input
        :param method: function used to fill missing values e.g. linear, null, zero, bfill, ffill
        :param target_cols [optional]: columns that should be interpolated, by default interpolates all numeric columns
        :param ts_col [optional]: specify other ts_col, by default this uses the ts_col within the TSDF object
        :param partition_cols [optional]: specify other partition_cols, by default this uses the partition_cols within the TSDF object
        :param show_interpolated [optional]: if true will include an additional column to show which rows have been fully interpolated.
        :param perform_checks: calculate time horizon and warnings if True (default is True)
        :return: new TSDF object containing interpolated data
        """

        # Set defaults for target columns, timestamp column and partition columns when not provided
        if freq is None:
            raise ValueError("freq must be provided")
        if func is None:
            raise ValueError("func must be provided")
        if ts_col is None:
            ts_col = self.ts_col
        if partition_cols is None:
            partition_cols = self.series_ids
        if target_cols is None:
            prohibited_cols: List[str] = partition_cols + [ts_col]
            # Don't filter by data type - allow all columns for PR-421 compatibility
            target_cols = [col for col in self.df.columns if col not in prohibited_cols]

        # First resample the data
        # Don't fill with zeros - let interpolation handle the nulls
        resampled_tsdf = self.resample(
            freq=freq,
            func=func,
            metricCols=target_cols,
            fill=False,  # Don't fill - interpolation will handle nulls
            perform_checks=perform_checks,
        )

        # Import interpolation function and pre-defined fill functions
        from tempo.interpol import backward_fill, forward_fill, zero_fill
        from tempo.interpol import interpolate as interpol_func

        # Map method names to interpolation functions (no lambdas)
        fn: Union[str, Callable[[pd.Series], pd.Series]]
        if method == "linear":
            fn = "linear"  # String method for pandas interpolation
        elif method == "null":
            # For null method, we don't fill - just return the resampled data
            return resampled_tsdf
        elif method == "zero":
            fn = zero_fill
        elif method == "bfill":
            fn = backward_fill
        elif method == "ffill":
            fn = forward_fill
        else:
            # Assume it's a valid pandas interpolation method string
            fn = method

        # Apply interpolation to the resampled data
        interpolated_tsdf = interpol_func(
            tsdf=resampled_tsdf,
            cols=target_cols,
            fn=fn,
            leading_margin=2,
            lagging_margin=2,
        )

        if show_interpolated:
            # Add a column indicating which rows were interpolated
            # This would require tracking which rows had nulls before interpolation
            logger.warning(
                "show_interpolated=True is not yet implemented in the refactored version"
            )

        return interpolated_tsdf

    def calc_bars(
        tsdf,
        freq: str,
        metricCols: Optional[List[str]] = None,
        fill: Optional[bool] = None,
    ) -> TSDF:
        resample_open = tsdf.resample(
            freq=freq, func="floor", metricCols=metricCols, prefix="open", fill=fill
        )
        resample_low = tsdf.resample(
            freq=freq, func="min", metricCols=metricCols, prefix="low", fill=fill
        )
        resample_high = tsdf.resample(
            freq=freq, func="max", metricCols=metricCols, prefix="high", fill=fill
        )
        resample_close = tsdf.resample(
            freq=freq, func="ceil", metricCols=metricCols, prefix="close", fill=fill
        )

        join_cols = resample_open.series_ids + [resample_open.ts_col]
        bars = (
            resample_open.df.join(resample_high.df, join_cols)
            .join(resample_low.df, join_cols)
            .join(resample_close.df, join_cols)
        )
        non_part_cols = set(set(bars.columns) - set(resample_open.series_ids)) - set(
            [resample_open.ts_col]
        )
        sel_and_sort = (
            resample_open.series_ids + [resample_open.ts_col] + sorted(non_part_cols)
        )
        bars = bars.select(sel_and_sort)

        return TSDF(
            bars, ts_col=resample_open.ts_col, series_ids=resample_open.series_ids
        )

    def fourier_transform(
        self, timestep: Union[int, float, complex], valueCol: str
    ) -> TSDF:
        """
        Function to fourier transform the time series to its frequency domain representation.
        :param timestep: timestep value to be used for getting the frequency scale
        :param valueCol: name of the time domain data column which will be transformed
        """

        def tempo_fourier_util(
            pdf: pd.DataFrame,
        ) -> pd.DataFrame:
            """
            This method is a vanilla python logic implementing fourier transform on a numpy array using the scipy module.
            This method is meant to be called from Tempo TSDF as a pandas function API on Spark
            """
            select_cols = list(pdf.columns)
            pdf.sort_values(by=["tpoints"], inplace=True, ascending=True)
            y = np.array(pdf["tdval"])
            tran = fft(y)
            r = tran.real
            i = tran.imag
            pdf["ft_real"] = r
            pdf["ft_imag"] = i
            N = tran.shape
            # fftfreq expects a float for the spacing parameter
            if isinstance(timestep, complex):
                spacing = abs(timestep)  # Use magnitude for complex numbers
            else:
                spacing = float(timestep)
            xf = fftfreq(N[0], spacing)
            pdf["freq"] = xf
            return pdf[select_cols + ["freq", "ft_real", "ft_imag"]]

        # TODO: Implement __validated_column or replace with proper validation
        # valueCol = self.__validated_column(self.df, valueCol)
        data = self.df
        # TODO: Handle sequence_col in refactored version
        if False:  # self.sequence_col:
            if self.series_ids == []:
                data = data.withColumn("dummy_group", sfn.lit("dummy_val"))
                data = (
                    data.select(
                        sfn.col("dummy_group"),
                        self.ts_col,
                        self.sequence_col,
                        sfn.col(valueCol),
                    )
                    .withColumn("tdval", sfn.col(valueCol))
                    .withColumn("tpoints", sfn.col(self.ts_col))
                )
                return_schema = ",".join(
                    [f"{i[0]} {i[1]}" for i in data.dtypes]
                    + ["freq double", "ft_real double", "ft_imag double"]
                )
                result = data.groupBy("dummy_group").applyInPandas(
                    tempo_fourier_util, return_schema
                )
                result = result.drop("dummy_group", "tdval", "tpoints")
            else:
                group_cols = self.series_ids
                data = (
                    data.select(
                        *group_cols,
                        self.ts_col,
                        self.sequence_col,
                        sfn.col(valueCol),
                    )
                    .withColumn("tdval", sfn.col(valueCol))
                    .withColumn("tpoints", sfn.col(self.ts_col))
                )
                return_schema = ",".join(
                    [f"{i[0]} {i[1]}" for i in data.dtypes]
                    + ["freq double", "ft_real double", "ft_imag double"]
                )
                result = data.groupBy(*group_cols).applyInPandas(
                    tempo_fourier_util, return_schema
                )
                result = result.drop("tdval", "tpoints")
        elif self.series_ids == []:
            data = data.withColumn("dummy_group", sfn.lit("dummy_val"))
            data = (
                data.select(sfn.col("dummy_group"), self.ts_col, sfn.col(valueCol))
                .withColumn("tdval", sfn.col(valueCol))
                .withColumn("tpoints", sfn.col(self.ts_col))
            )
            return_schema = ",".join(
                [f"{i[0]} {i[1]}" for i in data.dtypes]
                + ["freq double", "ft_real double", "ft_imag double"]
            )
            result = data.groupBy("dummy_group").applyInPandas(
                tempo_fourier_util, return_schema
            )
            result = result.drop("dummy_group", "tdval", "tpoints")
        else:
            group_cols = self.series_ids
            data = (
                data.select(*group_cols, self.ts_col, sfn.col(valueCol))
                .withColumn("tdval", sfn.col(valueCol))
                .withColumn("tpoints", sfn.col(self.ts_col))
            )
            return_schema = ",".join(
                [f"{i[0]} {i[1]}" for i in data.dtypes]
                + ["freq double", "ft_real double", "ft_imag double"]
            )
            result = data.groupBy(*group_cols).applyInPandas(
                tempo_fourier_util, return_schema
            )
            result = result.drop("tdval", "tpoints")

        # TODO: Handle sequence_col in refactored version
        return TSDF(result, ts_col=self.ts_col, series_ids=self.series_ids)

    def extractStateIntervals(
        self,
        *metric_cols: str,
        state_definition: Union[str, Callable[[Column, Column], Column]] = "=",
    ) -> DataFrame:
        """
        Extracts intervals from a :class:`~tsdf.TSDF` based on some notion of "state", as defined by the :param
        state_definition: parameter. The state definition consists of a comparison operation between the current and
        previous values of a metric. If the comparison operation evaluates to true across all metric columns,
        then we consider both points to be in the same "state". Changes of state occur when the comparison operator
        returns false for any given metric column. So, the default state definition ('=') entails that intervals of
        time wherein the metrics all remained constant. A state definition of '>=' would extract intervals wherein
        the metrics were all monotonically increasing.

        :param: metric_cols: the set of metric columns to evaluate for state changes
        :param: state_definition: the comparison function used to evaluate individual metrics for state changes.

        Either a string, giving a standard PySpark column comparison operation, or a binary function with the
        signature: `(x1: Column, x2: Column) -> Column` where the returned column expression evaluates to a
        :class:`~pyspark.sql.types.BooleanType`

        :return: a :class:`~pyspark.sql.DataFrame` object containing the resulting intervals
        """

        # https://spark.apache.org/docs/latest/sql-ref-null-semantics.html#comparison-operators-
        def null_safe_equals(col1: Column, col2: Column) -> Column:
            return (
                sfn.when(col1.isNull() & col2.isNull(), True)
                .when(col1.isNull() | col2.isNull(), False)
                .otherwise(operator.eq(col1, col2))
            )

        operator_dict = {
            # https://spark.apache.org/docs/latest/api/sql/#_2
            "!=": operator.ne,
            # https://spark.apache.org/docs/latest/api/sql/#_11
            "<>": operator.ne,
            # https://spark.apache.org/docs/latest/api/sql/#_8
            "<": operator.lt,
            # https://spark.apache.org/docs/latest/api/sql/#_9
            "<=": operator.le,
            # https://spark.apache.org/docs/latest/api/sql/#_10
            "<=>": null_safe_equals,
            # https://spark.apache.org/docs/latest/api/sql/#_12
            "=": operator.eq,
            # https://spark.apache.org/docs/latest/api/sql/#_13
            "==": operator.eq,
            # https://spark.apache.org/docs/latest/api/sql/#_14
            ">": operator.gt,
            # https://spark.apache.org/docs/latest/api/sql/#_15
            ">=": operator.ge,
        }

        # Validate state definition and construct state comparison function
        if type(state_definition) is str:
            if state_definition not in operator_dict:
                raise ValueError(
                    f"Invalid comparison operator for `state_definition` argument: {state_definition}."
                )

            def state_comparison_fn(a: CT, b: CT) -> Callable[[Column, Column], Column]:
                return operator_dict[state_definition](a, b)

        elif callable(state_definition):
            state_comparison_fn = state_definition  # type: ignore

        else:
            raise TypeError(
                f"The `state_definition` argument can be of type `str` or `callable`, "
                f"but received value of type {type(state_definition)}"
            )

        w = self.baseWindow()

        data = self.df

        # Get previous timestamp to identify start time of the interval
        data = data.withColumn(
            "previous_ts",
            sfn.lag(sfn.col(self.ts_col), offset=1).over(w),
        )

        # Determine state intervals using user-provided the state comparison function
        # The comparison occurs on the current and previous record per metric column
        temp_metric_compare_cols = []
        for mc in metric_cols:
            temp_metric_compare_col = f"__{mc}_compare"
            data = data.withColumn(
                temp_metric_compare_col,
                state_comparison_fn(sfn.col(mc), sfn.lag(sfn.col(mc), 1).over(w)),
            )
            temp_metric_compare_cols.append(temp_metric_compare_col)

        # Remove first record which will have no state change
        # and produces `null` for all state comparisons
        data = data.filter(sfn.col("previous_ts").isNotNull())

        # Each state comparison should return True if state remained constant
        data = data.withColumn(
            "state_change",
            sfn.array_contains(sfn.array(*temp_metric_compare_cols), False),
        )

        # Count the distinct state changes to get the unique intervals
        data = data.withColumn(
            "state_incrementer",
            sfn.sum(sfn.col("state_change").cast("int")).over(w),
        ).filter(~sfn.col("state_change"))

        # Find the start and end timestamp of the interval
        result = (
            data.groupBy(*self.series_ids, "state_incrementer")
            .agg(
                sfn.min("previous_ts").alias("start_ts"),
                sfn.max(self.ts_col).alias("end_ts"),
            )
            .drop("state_incrementer")
        )

        return result


class Comparable(metaclass=ABCMeta):
    """For typing functions generated by operator_dict"""

    @abstractmethod
    def __ne__(self, other: object) -> bool:
        pass

    @abstractmethod
    def __lt__(self, other: Any) -> bool:
        pass

    @abstractmethod
    def __le__(self, other: Any) -> bool:
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass

    @abstractmethod
    def __gt__(self, other: Any) -> bool:
        pass

    @abstractmethod
    def __ge__(self, other: Any) -> bool:
        pass


CT = TypeVar("CT", bound=Comparable)
