from typing import Collection

from pyspark.sql.types import *


class TSIndex:
    # Valid types for time index columns
    __valid_ts_types = (
        DateType(),
        TimestampType(),
        ByteType(),
        ShortType(),
        IntegerType(),
        LongType(),
        DecimalType(),
        FloatType(),
        DoubleType(),
    )

    def __init__(self, name: str, dataType: DataType) -> None:
        if dataType not in self.__valid_ts_types:
            raise TypeError(f"DataType {dataType} is not valid for a Timeseries Index")
        self.name = name
        self.dataType = dataType

    @classmethod
    def fromField(cls, ts_field: StructField) -> "TSIndex":
        return cls(ts_field.name, ts_field.dataType)


class TSSchema:
    """
    Schema type for a :class:`TSDF` class.
    """

    # Valid types for metric columns
    __metric_types = (
        BooleanType(),
        ByteType(),
        ShortType(),
        IntegerType(),
        LongType(),
        DecimalType(),
        FloatType(),
        DoubleType(),
    )

    def __init__(
        self,
        ts_idx: TSIndex,
        series_ids: Collection[str] = None,
        user_ts_col: str = None,
        subsequence_col: str = None,
    ) -> None:
        self.ts_idx = ts_idx
        self.series_ids = list(series_ids)
        self.user_ts_col = user_ts_col
        self.subsequence_col = subsequence_col

    @classmethod
    def fromDFSchema(
        cls, df_schema: StructType, ts_col: str, series_ids: Collection[str] = None
    ) -> "TSSchema":
        # construct a TSIndex for the given ts_col
        ts_idx = TSIndex.fromField(df_schema[ts_col])
        return cls(ts_idx, series_ids)

    @property
    def ts_index(self) -> str:
        return self.ts_idx.name

    @property
    def structural_columns(self) -> set[str]:
        """
        Structural columns are those that define the structure of the :class:`TSDF`. This includes the timeseries column,
        a timeseries index (if different), any subsequence column (if present), and the series ID columns.

        :return: a set of column names corresponding the structural columns of a :class:`TSDF`
        """
        struct_cols = {self.ts_index, self.user_ts_col, self.subsequence_col}.union(
            self.series_ids
        )
        struct_cols.discard(None)
        return struct_cols

    def validate(self, df_schema: StructType) -> None:
        pass

    def find_observational_columns(self, df_schema: StructType) -> list[StructField]:
        return [
            col for col in df_schema.fields if col.name not in self.structural_columns
        ]

    def find_metric_columns(self, df_schema: StructType) -> list[StructField]:
        return [
            col
            for col in self.find_observational_columns(df_schema)
            if col.dataType in self.__metric_types
        ]
