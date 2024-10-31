from typing import Any, List, Tuple
from functools import reduce

from pyspark.sql import DataFrame
from pyspark.sql.window import Window, WindowSpec
from pyspark.sql import functions as sfn

from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.tuning import CrossValidator


TMP_SPLIT_COL = "__tmp_split_col"
TMP_GAP_COL = "__tmp_gap_row"


class TimeSeriesCrossValidator(CrossValidator):
    # some additional parameters
    timeSeriesCol: Param[str] = Param(
        Params._dummy(),
        "timeSeriesCol",
        "The name of the time series column",
        typeConverter=TypeConverters.toString,
    )
    seriesIdCols: Param[List[str]] = Param(
        Params._dummy(),
        "seriesIdCols",
        "The name of the series id columns",
        typeConverter=TypeConverters.toListString,
    )
    gap: Param[int] = Param(
        Params._dummy(),
        "gap",
        "The gap between training and test set",
        typeConverter=TypeConverters.toInt,
    )

    def __init__(
        self,
        timeSeriesCol: str = "event_ts",
        seriesIdCols: List[str] = [],
        gap: int = 0,
        **other_kwargs: Any
    ) -> None:
        super(TimeSeriesCrossValidator, self).__init__(**other_kwargs)
        self._setDefault(timeSeriesCol="event_ts", seriesIdCols=[], gap=0)
        self._set(timeSeriesCol=timeSeriesCol, seriesIdCols=seriesIdCols, gap=gap)

    def getTimeSeriesCol(self) -> str:
        return self.getOrDefault(self.timeSeriesCol)

    def getSeriesIdCols(self) -> List[str]:
        return self.getOrDefault(self.seriesIdCols)

    def getGap(self) -> int:
        return self.getOrDefault(self.gap)

    def setTimeSeriesCol(self, value: str) -> "TimeSeriesCrossValidator":
        return self._set(timeSeriesCol=value)

    def setSeriesIdCols(self, value: List[str]) -> "TimeSeriesCrossValidator":
        return self._set(seriesIdCols=value)

    def setGap(self, value: int) -> "TimeSeriesCrossValidator":
        return self._set(gap=value)

    def _get_split_win(self, desc: bool = False) -> WindowSpec:
        ts_col_expr = sfn.col(self.getTimeSeriesCol())
        if desc:
            ts_col_expr = ts_col_expr.desc()
        win = Window.orderBy(ts_col_expr)
        series_id_cols = self.getSeriesIdCols()
        if series_id_cols and len(series_id_cols) > 0:
            win = win.partitionBy(*series_id_cols)
        return win

    def _kFold(self, dataset: DataFrame) -> List[Tuple[DataFrame, DataFrame]]:
        nFolds = self.getOrDefault(self.numFolds)
        nSplits = nFolds + 1

        # split the data into nSplits subsets by timeseries order
        split_df = dataset.withColumn(
            TMP_SPLIT_COL, sfn.ntile(nSplits).over(self._get_split_win())
        )
        all_splits = [
            split_df.filter(sfn.col(TMP_SPLIT_COL) == i).drop(TMP_SPLIT_COL)
            for i in range(1, nSplits + 1)
        ]
        assert len(all_splits) == nSplits

        # compose the k folds by including all previous splits in the training set,
        # and the next split in the test set
        kFolds = [
            (reduce(lambda a, b: a.union(b), all_splits[: i + 1]), all_splits[i + 1])
            for i in range(nFolds)
        ]
        assert len(kFolds) == nFolds
        for tv in kFolds:
            assert len(tv) == 2

        # trim out a gap from the training datasets, if specified
        gap = self.getOrDefault(self.gap)
        if gap > 0:
            order_cols = self.getSeriesIdCols() + [self.getTimeSeriesCol()]
            # trim each training dataset by the specified gap
            kFolds = [
                (
                    (
                        train_df.withColumn(
                            TMP_GAP_COL,
                            sfn.row_number().over(self._get_split_win(desc=True)),
                        )
                        .where(sfn.col(TMP_GAP_COL) > gap)
                        .drop(TMP_GAP_COL)
                        .orderBy(*order_cols)
                    ),
                    test_df,
                )
                for (train_df, test_df) in kFolds
            ]

        # return the k folds (training, test) datasets
        return kFolds
