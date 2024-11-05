import unittest

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

from tempo.ml import TimeSeriesCrossValidator

from tests.base import SparkTest

class TimeSeriesCrossValidatorTests(SparkTest):
    def test_empty_constructor(self):
        # construct with default parameters
        tscv = TimeSeriesCrossValidator()
        # check the object
        self.assertIsNotNone(tscv)
        self.assertIsInstance(tscv, TimeSeriesCrossValidator)
        self.assertIsInstance(tscv, CrossValidator)
        # check the parameters
        # CrossValidator parameters
        self.assertEqual(tscv.getNumFolds(), 3)
        self.assertEqual(tscv.getFoldCol(), "")
        self.assertEqual(tscv.getParallelism(), 1)
        self.assertEqual(tscv.getCollectSubModels(), False)
        # TimeSeriesCrossValidator parameters
        self.assertEqual(tscv.getTimeSeriesCol(), "event_ts")
        self.assertEqual(tscv.getSeriesIdCols(), [])
        self.assertEqual(tscv.getGap(), 0)

    def test_estim_eval_constructor(self):
        # set up estimator and evaluator
        estimator = GBTRegressor(labelCol="close", featuresCol="features")
        evaluator = RegressionEvaluator(labelCol="close",
                                        predictionCol="prediction",
                                        metricName="rmse")
        parm_grid = ParamGridBuilder().build()
        # construct with default parameters
        tscv = TimeSeriesCrossValidator(estimator=estimator,
                                        evaluator=evaluator,
                                        estimatorParamMaps=parm_grid)
        # test the parameters
        self.assertEqual(tscv.getEstimator(), estimator)
        self.assertEqual(tscv.getEvaluator(), evaluator)
        self.assertEqual(tscv.getEstimatorParamMaps(), parm_grid)

    def test_num_folds_param(self):
        # construct with default parameters
        tscv = TimeSeriesCrossValidator()
        # set the number of folds
        tscv.setNumFolds(5)
        # check the number of folds
        self.assertEqual(tscv.getNumFolds(), 5)

    def test_fold_col_param(self):
        # construct with default parameters
        tscv = TimeSeriesCrossValidator()
        # set the fold column
        tscv.setFoldCol("fold")
        # check the fold column
        self.assertEqual(tscv.getFoldCol(), "fold")

    def test_parallelism_param(self):
        # construct with default parameters
        tscv = TimeSeriesCrossValidator()
        # set the parallelism
        tscv.setParallelism(4)
        # check the parallelism
        self.assertEqual(tscv.getParallelism(), 4)

    def test_collect_sub_models_param(self):
        # construct with default parameters
        tscv = TimeSeriesCrossValidator()
        # set the collect sub models
        tscv.setCollectSubModels(True)
        # check the collect sub models
        self.assertEqual(tscv.getCollectSubModels(), True)

    def test_estimator_param(self):
        # set up estimator and evaluator
        estimator = GBTRegressor(labelCol="close", featuresCol="features")
        # construct with default parameters
        tscv = TimeSeriesCrossValidator()
        # set the estimator
        tscv.setEstimator(estimator)
        # check the estimator
        self.assertEqual(tscv.getEstimator(), estimator)

    def test_evaluator_param(self):
        # set up estimator and evaluator
        evaluator = RegressionEvaluator(labelCol="close",
                                        predictionCol="prediction",
                                        metricName="rmse")
        # construct with default parameters
        tscv = TimeSeriesCrossValidator()
        # set the evaluator
        tscv.setEvaluator(evaluator)
        # check the evaluator
        self.assertEqual(tscv.getEvaluator(), evaluator)

    def test_estimator_param_maps_param(self):
        # set up estimator and evaluator
        parm_grid = ParamGridBuilder().build()
        # construct with default parameters
        tscv = TimeSeriesCrossValidator()
        # set the estimator parameter maps
        tscv.setEstimatorParamMaps(parm_grid)
        # check the estimator parameter maps
        self.assertEqual(tscv.getEstimatorParamMaps(), parm_grid)

    def test_time_series_col_param(self):
        # construct with default parameters
        tscv = TimeSeriesCrossValidator()
        # set the time series column
        tscv.setTimeSeriesCol("ts")
        # check the time series column
        self.assertEqual(tscv.getTimeSeriesCol(), "ts")

    def test_series_id_cols_param(self):
        # construct with default parameters
        tscv = TimeSeriesCrossValidator()
        # set the series id columns
        tscv.setSeriesIdCols(["id1", "id2"])
        # check the series id columns
        self.assertEqual(tscv.getSeriesIdCols(), ["id1", "id2"])

    def test_gap_param(self):
        # construct with default parameters
        tscv = TimeSeriesCrossValidator()
        # set the gap
        tscv.setGap(2)
        # check the gap
        self.assertEqual(tscv.getGap(), 2)

# MAIN
if __name__ == "__main__":
    unittest.main()
