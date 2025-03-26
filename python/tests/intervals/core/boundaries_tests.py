from datetime import datetime
from types import NoneType

import pytest
from pandas import isna, Series, Timestamp

from python.tempo.intervals.core.boundaries import (
    BoundaryConverter,
    BoundaryValue,
    IntervalBoundaries,
    _BoundaryAccessor as InternalBoundaryAccessor,
)


class TestBoundaryConverter:
    def test_boundary_converter_for_string(self):
        sample = "2023-10-25"
        converter = BoundaryConverter.for_type(sample)

        assert converter.original_type == str
        assert converter.original_format == '%Y-%m-%d'

        timestamp = converter.to_timestamp(sample)
        assert isinstance(timestamp, Timestamp)

        result = converter.from_timestamp(timestamp)
        assert result == sample

    def test_boundary_converter_for_int(self):
        sample = 1698192000  # Equivalent to 2023-10-25 00:00:00 UTC
        converter = BoundaryConverter.for_type(sample)

        assert converter.original_type == int

        timestamp = converter.to_timestamp(sample)
        assert isinstance(timestamp, Timestamp)
        assert timestamp.timestamp() == sample

        restored = converter.from_timestamp(timestamp)
        assert restored == sample

    def test_boundary_converter_for_float(self):
        sample = 1698192000.00  # Equivalent to 2023-10-25 00:00:00 UTC
        converter = BoundaryConverter.for_type(sample)

        assert converter.original_type == float

        timestamp = converter.to_timestamp(sample)
        assert isinstance(timestamp, Timestamp)
        assert timestamp.timestamp() == sample

        restored = converter.from_timestamp(timestamp)
        assert restored == sample

    def test_boundary_converter_for_none(self):
        sample = None
        converter = BoundaryConverter.for_type(sample)

        assert converter.original_type is NoneType

        timestamp = converter.to_timestamp(sample)
        assert isna(timestamp)

        restored = converter.from_timestamp(timestamp)
        assert restored is None

    def test_boundary_converter_for_datetime(self):
        sample = datetime(2023, 10, 25, 15, 30)
        converter = BoundaryConverter.for_type(sample)

        assert converter.original_type == datetime

        timestamp = converter.to_timestamp(sample)
        assert isinstance(timestamp, Timestamp)
        assert timestamp.to_pydatetime() == sample

        restored = converter.from_timestamp(timestamp)
        assert restored == sample

    def test_boundary_converter_for_timestamp(self):
        sample = Timestamp("2023-10-25 15:30:00")
        converter = BoundaryConverter.for_type(sample)

        assert converter.original_type == Timestamp

        timestamp = converter.to_timestamp(sample)
        assert timestamp == sample

        restored = converter.from_timestamp(timestamp)
        assert restored == sample

    def test_boundary_converter_unsupported_type(self):
        sample = [2023, 10, 25]

        with pytest.raises(ValueError, match="Unsupported boundary type: <class 'list'>"):
            BoundaryConverter.for_type(sample)


class TestBoundaryValue:

    @pytest.fixture
    def boundary_converter_string(self):
        return BoundaryConverter.for_type("2023-11-01")

    @pytest.fixture
    def boundary_converter_int(self):
        return BoundaryConverter.for_type(1698883200)

    @pytest.fixture
    def boundary_converter_datetime(self):
        return BoundaryConverter.for_type(datetime(2023, 11, 1))

    def test_boundary_value_from_user_value_string(self, boundary_converter_string):
        value = "2023-11-01"
        boundary = BoundaryValue.from_user_value(value)
        assert isinstance(boundary.internal_value, Timestamp)
        assert boundary.internal_value == Timestamp(value)
        assert boundary.to_user_value() == value

    def test_boundary_value_from_user_value_int(self, boundary_converter_int):
        value = 1698883200
        boundary = BoundaryValue.from_user_value(value)
        assert isinstance(boundary.internal_value, Timestamp)
        assert boundary.internal_value == Timestamp(value, unit="s")
        assert boundary.to_user_value() == value

    def test_boundary_value_from_user_value_datetime(self, boundary_converter_datetime):
        value = datetime(2023, 11, 1)
        boundary = BoundaryValue.from_user_value(value)
        assert isinstance(boundary.internal_value, Timestamp)
        assert boundary.internal_value == Timestamp(value)
        assert boundary.to_user_value() == value

    def test_boundary_value_equality_same(self):
        boundary = BoundaryValue.from_user_value("2023-11-01")
        boundary_other = BoundaryValue.from_user_value("2023-11-01")
        assert boundary == boundary_other

    def test_boundary_value_equality_different(self):
        boundary = BoundaryValue.from_user_value("2023-11-01")
        boundary_other = BoundaryValue.from_user_value("2023-12-01")
        assert boundary != boundary_other

    def test_boundary_value_less_than(self):
        boundary = BoundaryValue.from_user_value("2023-11-01")
        boundary_other = BoundaryValue.from_user_value("2023-12-01")
        assert boundary < boundary_other

    def test_boundary_value_less_than_or_equal(self):
        boundary = BoundaryValue.from_user_value("2023-11-01")
        boundary_other = BoundaryValue.from_user_value("2023-12-01")
        assert boundary <= boundary_other

    def test_boundary_value_greater_than(self):
        boundary = BoundaryValue.from_user_value("2023-11-01")
        boundary_other = BoundaryValue.from_user_value("2023-12-01")
        assert boundary_other > boundary

    def test_boundary_value_greater_than_or_equal(self):
        boundary = BoundaryValue.from_user_value("2023-11-01")
        boundary_other = BoundaryValue.from_user_value("2023-12-01")
        assert boundary_other >= boundary


class TestIntervalBoundaries:

    @pytest.fixture
    def boundary_value_mock(self):
        class MockConverter:
            @staticmethod
            def to_timestamp(value):
                return Timestamp(value)

            @staticmethod
            def from_timestamp(timestamp):
                return str(timestamp)

        return BoundaryValue(
            _timestamp=Timestamp("2023-01-01"),
            _converter=MockConverter()
        )

    @pytest.fixture
    def interval_boundaries(self):
        return IntervalBoundaries.create(
            start="2023-01-01",
            end="2023-12-31"
        )

    def test_create_interval_boundaries(self, interval_boundaries):
        assert isinstance(interval_boundaries, IntervalBoundaries)
        assert interval_boundaries.start == "2023-01-01"
        assert interval_boundaries.end == "2023-12-31"

    def test_interval_boundaries_internal_start(self, interval_boundaries, boundary_value_mock):
        start = interval_boundaries.internal_start
        assert isinstance(start, BoundaryValue)
        assert start.internal_value == Timestamp("2023-01-01")

    def test_interval_boundaries_internal_end(self, interval_boundaries, boundary_value_mock):
        end = interval_boundaries.internal_end
        assert isinstance(end, BoundaryValue)
        assert end.internal_value == Timestamp("2023-12-31")

    def test_boundary_value_equality(self, boundary_value_mock):
        other = BoundaryValue(
            _timestamp=Timestamp("2023-01-01"),
            _converter=boundary_value_mock._converter
        )
        assert boundary_value_mock == other

    def test_boundary_value_comparison_earlier_less_than(self, boundary_value_mock):
        earlier = BoundaryValue(
            _timestamp=Timestamp("2022-01-01"),
            _converter=boundary_value_mock._converter
        )
        assert earlier < boundary_value_mock

    def test_boundary_value_comparison_later_greater_than(self, boundary_value_mock):
        later = BoundaryValue(
            _timestamp=Timestamp("2024-01-01"),
            _converter=boundary_value_mock._converter
        )
        assert later > boundary_value_mock

    def test_boundary_value_comparison_earlier_less_than_or_equal(self, boundary_value_mock):
        earlier = BoundaryValue(
            _timestamp=Timestamp("2022-01-01"),
            _converter=boundary_value_mock._converter
        )
        assert earlier <= boundary_value_mock

    def test_boundary_value_comparison_later_greater_than_or_equal(self, boundary_value_mock):
        later = BoundaryValue(
            _timestamp=Timestamp("2024-01-01"),
            _converter=boundary_value_mock._converter
        )
        assert later >= boundary_value_mock


class TestInternalBoundaryAccessor:

    def test_get_boundaries(self):
        # Arrange
        data = Series({
            "start_time": "2023-01-01",
            "end_time": "2023-01-02",
            "value": 10
        })
        accessor = InternalBoundaryAccessor("start_time", "end_time")

        # Act
        boundaries = accessor.get_boundaries(data)

        # Assert
        assert isinstance(boundaries, IntervalBoundaries)
        assert boundaries.start == "2023-01-01"
        assert boundaries.end == "2023-01-02"

    def test_set_boundaries(self):
        # Arrange
        data = Series({
            "start_time": "2023-01-01",
            "end_time": "2023-01-02",
            "value": 10
        })
        accessor = InternalBoundaryAccessor("start_time", "end_time")

        # Create new boundaries
        new_boundaries = IntervalBoundaries.create(
            start="2023-02-01",
            end="2023-02-05"
        )

        # Act
        updated_data = accessor.set_boundaries(data, new_boundaries)

        # Assert
        assert updated_data["start_time"] == "2023-02-01"
        assert updated_data["end_time"] == "2023-02-05"
        assert updated_data["value"] == 10  # Other fields should be unchanged

        # Original data should be unchanged (confirming we got a copy)
        assert data["start_time"] == "2023-01-01"
        assert data["end_time"] == "2023-01-02"

    def test_set_boundaries_different_field_names(self):
        # Arrange
        data = Series({
            "begin": "2023-01-01T00:00:00",
            "finish": "2023-01-02T00:00:00",
            "metric": 5
        })
        accessor = InternalBoundaryAccessor("begin", "finish")

        # Create new boundaries
        new_boundaries = IntervalBoundaries.create(
            start="2023-03-15T12:00:00",
            end="2023-03-16T12:00:00"
        )

        # Act
        updated_data = accessor.set_boundaries(data, new_boundaries)

        # Assert
        assert updated_data["begin"] == "2023-03-15T12:00:00"
        assert updated_data["finish"] == "2023-03-16T12:00:00"
        assert updated_data["metric"] == 5

    def test_get_boundaries_with_different_types(self):
        # Test with Timestamp objects
        data = Series({
            "start_ts": Timestamp("2023-01-01"),
            "end_ts": Timestamp("2023-01-02"),
            "value": 10
        })
        accessor = InternalBoundaryAccessor("start_ts", "end_ts")

        boundaries = accessor.get_boundaries(data)
        assert isinstance(boundaries.start, Timestamp)
        assert boundaries.start == Timestamp("2023-01-01")
        assert boundaries.end == Timestamp("2023-01-02")

        # Test with epoch timestamps (integers)
        epoch_data = Series({
            "start_epoch": 1672531200,  # 2023-01-01 00:00:00 UTC
            "end_epoch": 1672617600,  # 2023-01-02 00:00:00 UTC
            "value": 10
        })
        epoch_accessor = InternalBoundaryAccessor("start_epoch", "end_epoch")

        epoch_boundaries = epoch_accessor.get_boundaries(epoch_data)
        # Should be converted back to the original type (int)
        assert isinstance(epoch_boundaries.start, (int, float))

    def test_missing_field_error(self):
        # Test that KeyError is raised when field is missing
        data = Series({"start": "2023-01-01", "value": 10})  # Missing 'end'
        accessor = InternalBoundaryAccessor("start", "end")

        with pytest.raises(KeyError):
            accessor.get_boundaries(data)


class TestNegativeTimestampValidation:
    """Tests for negative timestamp validation in BoundaryConverter"""

    def test_negative_timestamp_with_string(self):
        """Test that string inputs resulting in negative timestamps raise ValueError"""
        # A date far in the past that would result in a negative timestamp
        sample = "1800-01-01"
        converter = BoundaryConverter.for_type("2023-01-01")  # Create converter with valid format

        # Using converter directly to test the validation
        with pytest.raises(ValueError, match="Timestamps cannot be negative."):
            converter.to_timestamp(sample)

    def test_negative_timestamp_with_int(self):
        """Test that integer inputs resulting in negative timestamps raise ValueError"""
        # A negative epoch timestamp
        sample = -1000000  # Negative seconds since epoch
        converter = BoundaryConverter.for_type(1698192000)  # Create converter with valid format

        with pytest.raises(ValueError, match="Timestamps cannot be negative."):
            converter.to_timestamp(sample)

    def test_negative_timestamp_with_float(self):
        """Test that float inputs resulting in negative timestamps raise ValueError"""
        # A negative epoch timestamp as float
        sample = -1000000.5  # Negative seconds since epoch
        converter = BoundaryConverter.for_type(1698192000.0)  # Create converter with valid format

        with pytest.raises(ValueError, match="Timestamps cannot be negative."):
            converter.to_timestamp(sample)

    def test_negative_timestamp_with_datetime(self):
        """Test that datetime inputs resulting in negative timestamps raise ValueError"""
        # A date far in the past that would result in a negative timestamp
        sample = datetime(1800, 1, 1)
        converter = BoundaryConverter.for_type(datetime(2023, 1, 1))  # Create converter with valid format

        with pytest.raises(ValueError, match="Timestamps cannot be negative."):
            converter.to_timestamp(sample)

    def test_negative_timestamp_with_pandas_timestamp(self):
        """Test that Timestamp inputs with negative values raise ValueError"""
        # Create a negative timestamp directly (may need to adjust based on how pandas handles this)
        sample = Timestamp('1800-01-01')  # A date that would result in a negative timestamp value
        converter = BoundaryConverter.for_type(Timestamp('2023-01-01'))  # Create converter with valid format

        with pytest.raises(ValueError, match="Timestamps cannot be negative."):
            converter.to_timestamp(sample)

    def test_boundary_value_creation_with_negative_timestamp(self):
        """Test that creating a BoundaryValue with a negative timestamp raises ValueError"""
        from python.tempo.intervals.core.boundaries import BoundaryValue

        # Try to create a BoundaryValue with a string date that would result in a negative timestamp
        with pytest.raises(ValueError, match="Timestamps cannot be negative."):
            BoundaryValue.from_user_value("1800-01-01")

    def test_interval_boundaries_creation_with_negative_start(self):
        """Test that creating IntervalBoundaries with a negative start timestamp raises ValueError"""
        from python.tempo.intervals.core.boundaries import IntervalBoundaries

        # Try to create IntervalBoundaries with a negative start timestamp
        with pytest.raises(ValueError, match="Timestamps cannot be negative."):
            IntervalBoundaries.create(
                start="1800-01-01",
                end="2023-01-01"
            )

    def test_interval_boundaries_creation_with_negative_end(self):
        """Test that creating IntervalBoundaries with a negative end timestamp raises ValueError"""
        from python.tempo.intervals.core.boundaries import IntervalBoundaries

        # Try to create IntervalBoundaries with a negative end timestamp
        with pytest.raises(ValueError, match="Timestamps cannot be negative."):
            IntervalBoundaries.create(
                start="2023-01-01",
                end="1800-01-01"
            )
