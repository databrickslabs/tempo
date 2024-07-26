from typing import List, Tuple
from functools import reduce

from pyspark.sql import DataFrame
from pyspark.sql.window import Window, WindowSpec
from pyspark.sql import functions as sfn

from pyspark.ml.param import Param, Params
from pyspark.ml.tuning import CrossValidator


TMP_SPLIT_COL = "__tmp_split_col"

class TimeSeriesCrossValidator(CrossValidator):
    # some additional parameters
    timeSeriesCol: Param[str] = Param(
        Params._dummy(),
        "timeSeriesCol",
        "The name of the time series column"
    )
    seriesIdCols: Param[List[str]] = Param(
        Params._dummy(),
        "seriesIdCols",
        "The name of the series id columns"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def getTimeSeriesCol(self) -> str:
        return self.getOrDefault(self.timeSeriesCol)

    def getSeriesIdCols(self) -> List[str]:
        return self.getOrDefault(self.seriesIdCols)

    def setTimeSeriesCol(self, value: str) -> "TimeSeriesCrossValidator":
        return self._set(timeSeriesCol=value)

    def setSeriesIdCols(self, value: List[str]) -> "TimeSeriesCrossValidator":
        return self._set(seriesIdCols=value)

    def _get_split_win(self) -> WindowSpec:
        win = Window.orderBy(self.getTimeSeriesCol())
        series_id_cols = self.getSeriesIdCols()
        if series_id_cols and len(series_id_cols) > 0:
            win = win.partitionBy(*series_id_cols)
        return win

    def _kFold(self, dataset: DataFrame) -> List[Tuple[DataFrame, DataFrame]]:
        nFolds = self.getOrDefault(self.numFolds)
        nSplits = nFolds+1

        # split the data into nSplits subsets by timeseries order
        split_df = dataset.withColumn(TMP_SPLIT_COL,
                                      sfn.ntile(nSplits).over(self._get_split_win()))
        all_splits = [split_df.filter(sfn.col(TMP_SPLIT_COL) == i).drop(TMP_SPLIT_COL)
                      for i in range(1, nSplits+1)]
        assert len(all_splits) == nSplits

        # compose the k folds by including all previous splits in the training set,
        # and the next split in the test set
        kFolds = [(reduce(lambda a, b: a.union(b), all_splits[:i+1]), all_splits[i+1])
                  for i in range(nFolds)]
        assert len(kFolds) == nFolds
        for tv in kFolds:
            assert len(tv) == 2

        return kFolds
