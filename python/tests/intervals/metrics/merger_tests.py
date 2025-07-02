import pandas as pd
import pytest
from pandas import Series

from tempo.intervals.core.exceptions import ErrorMessages
from tempo.intervals.metrics.merger import MetricMerger, DefaultMetricMerger
from tempo.intervals.metrics.operations import MetricMergeConfig
from tempo.intervals.metrics.strategies import MetricMergeStrategy


class TestMetricStrategy(MetricMergeStrategy):
    """Simple test strategy for merging metrics"""

    def validate(self, value1: Series, value2: Series) -> None:
        pass

    def merge(self, value1: Series, value2: Series) -> Series:
        return value1 + value2


class FailingMetricStrategy(MetricMergeStrategy):
    """Strategy that fails during validation"""

    def validate(self, value1: Series, value2: Series) -> None:
        raise ValueError("Validation failed")

    def merge(self, value1: Series, value2: Series) -> Series:
        return value1


class MockInterval:
    """Mock class for Interval"""

    def __init__(self, data, metric_fields):
        self.data = data
        self.metric_fields = metric_fields


@pytest.fixture
def test_data():
    """Fixture to create test dataframes"""
    data1 = pd.DataFrame({
        'time': [1, 2, 3],
        'metric1': [10, 20, 30],
        'metric2': [100, 200, 300]
    })

    data2 = pd.DataFrame({
        'time': [1, 2, 3],
        'metric1': [1, 2, 3],
        'metric2': [10, 20, 30]
    })

    return data1, data2


@pytest.fixture
def test_intervals(test_data):
    """Fixture to create mock intervals"""
    data1, data2 = test_data
    interval1 = MockInterval(data1, ['metric1', 'metric2'])
    interval2 = MockInterval(data2, ['metric1', 'metric2'])
    return interval1, interval2


@pytest.fixture
def merge_config():
    """Fixture to create a merge config with test strategies"""
    config = MetricMergeConfig()
    strategy = TestMetricStrategy()
    config.set_strategy('metric1', strategy)
    config.set_strategy('metric2', strategy)
    return config


class TestMetricMerger:

    def test_init_with_default_config(self):
        """Test initialization with default config"""
        merger = MetricMerger()
        assert isinstance(merger.merge_config, MetricMergeConfig)

    def test_init_with_custom_config(self, merge_config):
        """Test initialization with custom config"""
        merger = MetricMerger(merge_config)
        assert merger.merge_config is merge_config

    def test_merge_success(self, test_intervals, merge_config):
        """Test successful merge operation"""
        interval1, interval2 = test_intervals
        merger = MetricMerger(merge_config)

        result = merger.merge(interval1, interval2)

        # The test strategy adds values, so we expect these sums
        assert result['metric1'].tolist() == [11, 22, 33]
        assert result['metric2'].tolist() == [110, 220, 330]

    def test_merge_different_metric_fields(self, test_data):
        """Test merging intervals with different metric fields"""
        data1, data2 = test_data
        interval1 = MockInterval(data1, ['metric1', 'metric2'])
        interval2 = MockInterval(data2, ['metric1'])

        merger = MetricMerger()

        with pytest.raises(ValueError) as excinfo:
            merger.merge(interval1, interval2)

        assert str(excinfo.value) == ErrorMessages.METRIC_COLUMNS_LENGTH

    def test_failing_strategy(self, test_intervals):
        """Test when a merge strategy fails during validation"""
        interval1, interval2 = test_intervals

        config = MetricMergeConfig()
        failing_strategy = FailingMetricStrategy()
        config.set_strategy('metric1', failing_strategy)
        config.set_strategy('metric2', TestMetricStrategy())

        merger = MetricMerger(config)

        with pytest.raises(ValueError) as excinfo:
            merger.merge(interval1, interval2)

        assert "Strategy FailingMetricStrategy failed: Validation failed" in str(excinfo.value)

    def test_apply_merge_strategy(self):
        """Test the _apply_merge_strategy static method"""
        value1 = Series([1, 2, 3])
        value2 = Series([4, 5, 6])
        strategy = TestMetricStrategy()

        result = MetricMerger._apply_merge_strategy(value1, value2, strategy)

        assert result.tolist() == [5, 7, 9]

    def test_apply_merge_strategy_failure(self):
        """Test the _apply_merge_strategy method with failing strategy"""
        value1 = Series([1, 2, 3])
        value2 = Series([4, 5, 6])
        strategy = FailingMetricStrategy()

        with pytest.raises(ValueError) as excinfo:
            MetricMerger._apply_merge_strategy(value1, value2, strategy)

        assert "Strategy FailingMetricStrategy failed: Validation failed" in str(excinfo.value)


class TestDefaultMetricMerger:

    def test_inheritance(self):
        """Test that DefaultMetricMerger inherits from MetricMerger"""
        merger = DefaultMetricMerger()
        assert isinstance(merger, MetricMerger)

    def test_merge_functionality(self, test_intervals, merge_config):
        """Test that merge functionality works through inheritance"""
        interval1, interval2 = test_intervals
        merger = DefaultMetricMerger(merge_config)

        result = merger.merge(interval1, interval2)

        # The test strategy adds values, so we expect these sums
        assert result['metric1'].tolist() == [11, 22, 33]
        assert result['metric2'].tolist() == [110, 220, 330]
