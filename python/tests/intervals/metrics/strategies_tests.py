import pytest
from numpy import nan
from pandas import isna, notna, Series, NA, NaT

from tempo.intervals.metrics.strategies import (
    KeepFirstStrategy,
    KeepLastStrategy,
    SumStrategy,
    MaxStrategy,
    MinStrategy,
    AverageStrategy,
)


class TestMetricMergeStrategies:

    @pytest.fixture
    def numeric_values(self):
        return [
            (10, 20),  # Two integers
            (10.5, 20.5),  # Two floats
            (10, 20.5),  # Mixed int and float
            (0, 0),  # Zeros
            (-10, 10),  # Negative and positive
            (nan, 10),  # NaN and value
            (10, nan),  # Value and NaN
            (nan, nan),  # Both NaN
        ]

    @pytest.fixture
    def non_numeric_values(self):
        return [
            ("string1", "string2"),
            ("string", 10),
            (10, "string"),
            (True, False),
        ]

    @pytest.fixture
    def series_values(self):
        return [
            (Series([1, 2, 3]), Series([4, 5, 6])),  # Numeric Series
            (Series([1, 2, nan]), Series([4, nan, 6])),  # Series with NaN
            (Series([1, 2, 3]), 4),  # Series and scalar
            (5, Series([6, 7, 8])),  # Scalar and Series
            (Series(["a", "b", "c"]), Series(["d", "e", "f"])),  # String Series
        ]

    # Tests for KeepFirstStrategy with scalar values
    def test_keep_first_strategy_scalar(self, numeric_values):
        strategy = KeepFirstStrategy()

        for value1, value2 in numeric_values:
            result = strategy.merge(value1, value2)
            expected = value1 if notna(value1) else value2

            if isna(result) and isna(expected):
                assert True  # Both are NaN
            else:
                assert result == expected

    def test_keep_first_strategy_with_non_numeric(self, non_numeric_values):
        strategy = KeepFirstStrategy()

        for value1, value2 in non_numeric_values:
            result = strategy.merge(value1, value2)
            assert result == value1

    def test_keep_first_strategy_series(self, series_values):
        strategy = KeepFirstStrategy()

        for value1, value2 in series_values:
            result = strategy.merge(value1, value2)

            # Check if result is the expected type
            if isinstance(value1, Series) or isinstance(value2, Series):
                assert isinstance(result, Series)

                # Convert scalar to Series if needed for comparison
                v1 = value1 if isinstance(value1, Series) else Series([value1] * len(result))
                v2 = value2 if isinstance(value2, Series) else Series([value2] * len(result))

                # Apply the expected logic for comparison without using mask indexing
                expected = v1.copy()
                for i in range(len(expected)):
                    if isna(expected.iloc[i]):
                        expected.iloc[i] = v2.iloc[i]

                # Compare each element
                for i in range(len(result)):
                    if isna(result.iloc[i]) and isna(expected.iloc[i]):
                        continue  # Both NaN
                    assert result.iloc[i] == expected.iloc[i]
            else:
                # Scalar case
                expected = value1 if notna(value1) else value2
                assert result == expected

    def test_keep_last_strategy_scalar(self, numeric_values):
        strategy = KeepLastStrategy()

        for value1, value2 in numeric_values:
            result = strategy.merge(value1, value2)
            expected = value2 if notna(value2) else value1

            if isna(result) and isna(expected):
                assert True  # Both are NaN
            else:
                assert result == expected

    def test_keep_last_strategy_with_non_numeric(self, non_numeric_values):
        strategy = KeepLastStrategy()

        for value1, value2 in non_numeric_values:
            result = strategy.merge(value1, value2)
            assert result == value2

    def test_keep_last_strategy_series(self, series_values):
        strategy = KeepLastStrategy()

        for value1, value2 in series_values:
            result = strategy.merge(value1, value2)

            # Check if result is the expected type
            if isinstance(value1, Series) or isinstance(value2, Series):
                assert isinstance(result, Series)

                # Convert scalar to Series if needed for comparison
                v1 = value1 if isinstance(value1, Series) else Series([value1] * len(result))
                v2 = value2 if isinstance(value2, Series) else Series([value2] * len(result))

                # Apply the expected logic for comparison without using mask indexing
                expected = v2.copy()
                for i in range(len(expected)):
                    if isna(expected.iloc[i]):
                        expected.iloc[i] = v1.iloc[i]

                # Compare each element
                for i in range(len(result)):
                    if isna(result.iloc[i]) and isna(expected.iloc[i]):
                        continue  # Both NaN
                    assert result.iloc[i] == expected.iloc[i]
            else:
                # Scalar case
                expected = value2 if notna(value2) else value1
                assert result == expected

    def test_sum_strategy_scalar(self, numeric_values):
        strategy = SumStrategy()

        for value1, value2 in numeric_values:
            result = strategy.merge(value1, value2)
            val1 = 0 if isna(value1) else value1
            val2 = 0 if isna(value2) else value2
            expected = val1 + val2

            assert pytest.approx(result) == expected

    def test_sum_strategy_validation(self, non_numeric_values):
        strategy = SumStrategy()

        for value1, value2 in non_numeric_values:
            if (isinstance(value1, (int, float)) or isna(value1)) and \
                    (isinstance(value2, (int, float)) or isna(value2)):
                continue  # Skip valid numeric combinations

            with pytest.raises(ValueError, match="SumStrategy requires numeric values"):
                strategy.validate(value1, value2)
                strategy.merge(value1, value2)

    def test_sum_strategy_series(self):
        strategy = SumStrategy()

        # Test with numeric Series
        s1 = Series([1, 2, nan])
        s2 = Series([3, nan, 5])

        result = strategy.merge(s1, s2)
        assert isinstance(result, Series)
        expected = Series([4, 2, 5])  # NaNs treated as 0

        for i in range(len(result)):
            assert result.iloc[i] == expected.iloc[i]

        # Test with mixed Series and scalar
        s3 = Series([1, 2, 3])
        scalar = 10

        result = strategy.merge(s3, scalar)
        assert isinstance(result, Series)
        expected = Series([11, 12, 13])

        for i in range(len(result)):
            assert result.iloc[i] == expected.iloc[i]

        # Test with invalid non-numeric Series
        s4 = Series(["a", "b", "c"])

        with pytest.raises(ValueError, match="SumStrategy requires numeric values"):
            strategy.merge(s4, s3)

    def test_scalar_plus_series_sum_strategy(self):
        """Test the scalar + Series code path in SumStrategy."""
        strategy = SumStrategy()

        # Test case 1: Normal scalar + series
        scalar = 5
        s = Series([1, 2, 3])
        result = strategy.merge(scalar, s)

        assert isinstance(result, Series)
        assert len(result) == len(s)
        assert result.iloc[0] == 6  # 5 + 1
        assert result.iloc[1] == 7  # 5 + 2
        assert result.iloc[2] == 8  # 5 + 3

        # Test case 2: NaN scalar + series
        scalar = nan
        s = Series([1, 2, 3])
        result = strategy.merge(scalar, s)

        assert isinstance(result, Series)
        assert len(result) == len(s)
        assert result.iloc[0] == 1  # 0 + 1 (NaN treated as 0)
        assert result.iloc[1] == 2  # 0 + 2 (NaN treated as 0)
        assert result.iloc[2] == 3  # 0 + 3 (NaN treated as 0)

        # Test case 3: Scalar + series with NaN values
        scalar = 5
        s = Series([1, nan, 3])
        result = strategy.merge(scalar, s)

        assert isinstance(result, Series)
        assert len(result) == len(s)
        assert result.iloc[0] == 6  # 5 + 1
        assert result.iloc[1] == 5  # 5 + 0 (NaN treated as 0)
        assert result.iloc[2] == 8  # 5 + 3

    def test_different_length_series_sum_strategy(self):
        """Test handling of Series with different lengths in SumStrategy."""
        strategy = SumStrategy()

        # Test case 1: First series longer than second
        s1 = Series([1, 2, 3, 4])
        s2 = Series([10, 20])
        result = strategy.merge(s1, s2)

        assert isinstance(result, Series)
        assert len(result) == 4  # Should match the max length
        assert result.iloc[0] == 11  # 1 + 10
        assert result.iloc[1] == 22  # 2 + 20
        assert result.iloc[2] == 3  # 3 + 0 (NaN treated as 0)
        assert result.iloc[3] == 4  # 4 + 0 (NaN treated as 0)

        # Test case 2: Second series longer than first
        s1 = Series([1, 2])
        s2 = Series([10, 20, 30, 40])
        result = strategy.merge(s1, s2)

        assert isinstance(result, Series)
        assert len(result) == 4  # Should match the max length
        assert result.iloc[0] == 11  # 1 + 10
        assert result.iloc[1] == 22  # 2 + 20
        assert result.iloc[2] == 30  # 0 + 30 (NaN treated as 0)
        assert result.iloc[3] == 40  # 0 + 40 (NaN treated as 0)

        # Test case 3: Both series have NaN values at different positions
        s1 = Series([1, nan, 3])
        s2 = Series([nan, 2, nan, 4])
        result = strategy.merge(s1, s2)

        assert isinstance(result, Series)
        assert len(result) == 4  # Should match the max length
        assert result.iloc[0] == 1  # 1 + 0 (NaN treated as 0)
        assert result.iloc[1] == 2  # 0 + 2 (NaN treated as 0)
        assert result.iloc[2] == 3  # 3 + 0 (NaN treated as 0)
        assert result.iloc[3] == 4  # 0 + 4 (NaN treated as 0)

    def test_validation_for_non_numeric_values_sum_strategy(self):
        """Test validation for non-numeric values in SumStrategy."""
        strategy = SumStrategy()

        # Test case 1: String in Series
        s1 = Series([1, 2, 3])
        s2 = Series(['a', 2, 3])  # Contains non-numeric value

        with pytest.raises(ValueError, match="SumStrategy requires numeric values"):
            strategy.merge(s1, s2)

        # Test case 2: String as scalar
        s1 = Series([1, 2, 3])
        scalar = 'a'  # Non-numeric scalar

        with pytest.raises(ValueError, match="SumStrategy requires numeric values"):
            strategy.merge(s1, scalar)

        # Test case 3: None values (should be treated as NaN and allowed)
        s1 = Series([1, None, 3])
        s2 = Series([4, 5, 6])
        result = strategy.merge(s1, s2)

        assert isinstance(result, Series)
        assert result.iloc[0] == 5  # 1 + 4
        assert result.iloc[1] == 5  # 0 + 5 (None treated as NaN, then as 0)
        assert result.iloc[2] == 9  # 3 + 6

        # Test case 4: Other pandas NA values (should be treated as NaN and allowed)
        s1 = Series([1, NA, 3])
        s2 = Series([4, 5, NaT])
        result = strategy.merge(s1, s2)

        assert isinstance(result, Series)
        assert result.iloc[0] == 5  # 1 + 4
        assert result.iloc[1] == 5  # 0 + 5 (NA treated as NaN, then as 0)
        assert result.iloc[2] == 3  # 3 + 0 (NaT treated as NaN, then as 0)

        # Test case 5: Boolean values (should be valid as they're numeric)
        s1 = Series([1, True, 3])
        s2 = Series([4, False, 6])
        result = strategy.merge(s1, s2)

        assert isinstance(result, Series)
        assert result.iloc[0] == 5  # 1 + 4
        assert result.iloc[1] == 1  # 1 + 0 (True = 1, False = 0)
        assert result.iloc[2] == 9  # 3 + 6

    def test_max_strategy_scalar(self, numeric_values):
        strategy = MaxStrategy()

        for value1, value2 in numeric_values:
            result = strategy.merge(value1, value2)
            series = Series([value1, value2])
            expected = series.max()

            if isna(result) and isna(expected):
                assert True  # Both are NaN
            else:
                assert result == expected

    def test_max_strategy_series(self):
        strategy = MaxStrategy()

        # Test with numeric Series
        s1 = Series([1, 5, nan])
        s2 = Series([3, 2, 5])

        result = strategy.merge(s1, s2)
        assert isinstance(result, Series)

        # Max should take the max value at each position, ignoring NaN
        expected = Series([3, 5, 5])

        for i in range(len(result)):
            assert result.iloc[i] == expected.iloc[i]

        # Test with mixed Series and scalar
        s3 = Series([1, 7, 3])
        scalar = 2

        result = strategy.merge(s3, scalar)
        assert isinstance(result, Series)
        expected = Series([2, 7, 3])

        for i in range(len(result)):
            assert result.iloc[i] == expected.iloc[i]

    def test_min_strategy_scalar(self, numeric_values):
        strategy = MinStrategy()

        for value1, value2 in numeric_values:
            result = strategy.merge(value1, value2)
            series = Series([value1, value2])
            expected = series.min()

            if isna(result) and isna(expected):
                assert True  # Both are NaN
            else:
                assert result == expected

    def test_min_strategy_series(self):
        strategy = MinStrategy()

        # Test with numeric Series
        s1 = Series([1, 5, nan])
        s2 = Series([3, 2, 5])

        result = strategy.merge(s1, s2)
        assert isinstance(result, Series)

        # Min should take the min value at each position, ignoring NaN
        expected = Series([1, 2, 5])

        for i in range(len(result)):
            assert result.iloc[i] == expected.iloc[i]

        # Test with mixed Series and scalar
        s3 = Series([1, 7, 3])
        scalar = 2

        result = strategy.merge(s3, scalar)
        assert isinstance(result, Series)
        expected = Series([1, 2, 2])

        for i in range(len(result)):
            assert result.iloc[i] == expected.iloc[i]

    def test_average_strategy_scalar(self, numeric_values):
        strategy = AverageStrategy()

        for value1, value2 in numeric_values:
            # Skip pairs where both values are NaN
            if isna(value1) and isna(value2):
                continue

            result = strategy.merge(value1, value2)
            series = Series([value1, value2])
            expected = series.mean()

            if isna(result) and isna(expected):
                assert True  # Both are NaN
            else:
                assert pytest.approx(result) == expected

    def test_average_strategy_validation(self, non_numeric_values):
        strategy = AverageStrategy()

        for value1, value2 in non_numeric_values:
            if (isinstance(value1, (int, float)) or isna(value1)) and \
                    (isinstance(value2, (int, float)) or isna(value2)):
                continue  # Skip valid numeric combinations

            with pytest.raises(ValueError, match="AverageStrategy requires numeric values"):
                strategy.validate(value1, value2)
                strategy.merge(value1, value2)

    def test_average_strategy_series(self):
        strategy = AverageStrategy()

        # Test with numeric Series
        s1 = Series([2, 4, nan])
        s2 = Series([4, nan, 6])

        result = strategy.merge(s1, s2)
        assert isinstance(result, Series)

        # Average should compute mean of non-NaN values at each position
        expected = Series([3, 4, 6])  # (2+4)/2, only 4 present, only 6 present

        for i in range(len(result)):
            assert result.iloc[i] == expected.iloc[i]

        # Test with mixed Series and scalar
        s3 = Series([2, 4, 6])
        scalar = 8

        result = strategy.merge(s3, scalar)
        assert isinstance(result, Series)
        expected = Series([5, 6, 7])  # (2+8)/2, (4+8)/2, (6+8)/2

        for i in range(len(result)):
            assert result.iloc[i] == expected.iloc[i]

        # Test with invalid non-numeric Series
        s4 = Series(["a", "b", "c"])

        with pytest.raises(ValueError, match="AverageStrategy requires numeric values"):
            strategy.merge(s4, s3)

    def test_convert_to_series(self):
        strategy = MaxStrategy()  # Using any strategy that has the method

        # Test with two scalars
        v1, v2, was_series = strategy._convert_to_series(10, 20)
        assert isinstance(v1, Series)
        assert isinstance(v2, Series)
        assert v1.iloc[0] == 10
        assert v2.iloc[0] == 20
        assert was_series is False

        # Test with one Series, one scalar
        s1 = Series([1, 2, 3])
        v1, v2, was_series = strategy._convert_to_series(s1, 20)
        assert isinstance(v1, Series)
        assert isinstance(v2, Series)
        assert (v1 == s1).all()
        assert v2.iloc[0] == 20
        assert was_series is True

        # Test with two Series
        s1 = Series([1, 2, 3])
        s2 = Series([4, 5, 6])
        v1, v2, was_series = strategy._convert_to_series(s1, s2)
        assert isinstance(v1, Series)
        assert isinstance(v2, Series)
        assert (v1 == s1).all()
        assert (v2 == s2).all()
        assert was_series is True
