from abc import abstractmethod

from base import SparkTest

class PandasBaselineTest(SparkTest):

    @abstractmethod
    def pandasDataFrames(self) -> list:
        pass

    def tempoVsPandas(self):