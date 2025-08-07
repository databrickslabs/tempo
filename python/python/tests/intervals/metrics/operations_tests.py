import pytest
from pandas import Series

from tempo.intervals.metrics.operations import MetricNormalizer, MetricMergeConfig
from tempo.intervals.metrics.strategies import KeepFirstStrategy, KeepLastStrategy


class TestMetricNormalizer:
    def test_abstract_class(self):
        """Test that MetricNormalizer cannot be instantiated directly"""
        with pytest.raises(TypeError) as excinfo:
            MetricNormalizer()
        assert "abstract" in str(excinfo.value).lower()

    def test_concrete_implementation(self):
        """Test that a concrete implementation of MetricNormalizer can be instantiated"""

        class ConcreteMetricNormalizer(MetricNormalizer):
            def normalize(self, interval):
                return Series({"value": 1.0})

        normalizer = ConcreteMetricNormalizer()
        assert isinstance(normalizer, MetricNormalizer)

        # Test that the normalize method works as expected
        result = normalizer.normalize(None)  # Passing None as we've mocked the Interval dependency
        assert isinstance(result, Series)
        assert result["value"] == 1.0


class TestMetricMergeConfig:
    def test_default_initialization(self):
        """Test initialization with default arguments"""
        config = MetricMergeConfig()
        assert isinstance(config.default_strategy, KeepLastStrategy)
        assert config.column_strategies == {}

    def test_custom_initialization(self):
        """Test initialization with custom arguments"""
        default_strategy = KeepFirstStrategy()
        column_strategies = {
            "col1": KeepFirstStrategy(),
            "col2": KeepLastStrategy()
        }

        config = MetricMergeConfig(
            default_strategy=default_strategy,
            column_strategies=column_strategies
        )

        assert config.default_strategy is default_strategy
        assert config.column_strategies is column_strategies

    def test_validate_strategies_default_strategy(self):
        """Test validation of default_strategy"""
        with pytest.raises(ValueError) as excinfo:
            MetricMergeConfig(default_strategy="not a strategy")
        assert "default_strategy must be an instance of MetricMergeStrategy" in str(excinfo.value)

    def test_validate_strategies_column_strategies(self):
        """Test validation of column_strategies"""
        with pytest.raises(ValueError) as excinfo:
            MetricMergeConfig(
                column_strategies={"col1": "not a strategy"}
            )
        assert "Strategy for column col1 must be an instance of MetricMergeStrategy" in str(excinfo.value)

    def test_get_strategy_existing_column(self):
        """Test getting a strategy for a column that has a specific strategy set"""
        default_strategy = KeepFirstStrategy()
        col1_strategy = KeepLastStrategy()

        config = MetricMergeConfig(
            default_strategy=default_strategy,
            column_strategies={"col1": col1_strategy}
        )

        strategy = config.get_strategy("col1")
        assert strategy is col1_strategy

    def test_get_strategy_nonexistent_column(self):
        """Test getting a strategy for a column that doesn't have a specific strategy set"""
        default_strategy = KeepFirstStrategy()

        config = MetricMergeConfig(default_strategy=default_strategy)

        strategy = config.get_strategy("nonexistent_column")
        assert strategy is default_strategy

    def test_set_strategy_valid(self):
        """Test setting a valid strategy for a column"""
        config = MetricMergeConfig()
        new_strategy = KeepFirstStrategy()

        config.set_strategy("col1", new_strategy)

        assert "col1" in config.column_strategies
        assert config.column_strategies["col1"] is new_strategy

    def test_set_strategy_invalid(self):
        """Test setting an invalid strategy for a column"""
        config = MetricMergeConfig()

        with pytest.raises(ValueError) as excinfo:
            config.set_strategy("col1", "not a strategy")
        assert "The provided strategy must be an instance of MetricMergeStrategy" in str(excinfo.value)

    def test_set_strategy_override(self):
        """Test overriding an existing strategy for a column"""
        initial_strategy = KeepFirstStrategy()
        new_strategy = KeepLastStrategy()

        config = MetricMergeConfig(
            column_strategies={"col1": initial_strategy}
        )

        config.set_strategy("col1", new_strategy)

        assert config.column_strategies["col1"] is new_strategy
