from typing import List, Tuple

from pyspark.ml.tuning import CrossValidator
from pyspark.sql import DataFrame


class TimeSeriesCrossValidator(CrossValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _kFold(self, dataset: DataFrame) -> List[Tuple[DataFrame, DataFrame]]:
        pass

