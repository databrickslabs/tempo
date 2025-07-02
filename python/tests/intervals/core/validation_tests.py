import pandas as pd
import pytest

from tempo.intervals.core.exceptions import EmptyIntervalError, InvalidDataTypeError, InvalidMetricColumnError
from tempo.intervals.core.validation import IntervalValidator, ValidationResult


@pytest.fixture
def valid_series():
    return pd.Series([1, 2, 3, 4, 5])


@pytest.fixture
def empty_series():
    return pd.Series([])


@pytest.fixture
def valid_column_list():
    return ["col1", "col2", "col3"]


@pytest.fixture
def duplicate_column_list():
    return ["col1", "col2", "col1"]


@pytest.fixture
def non_string_column_list():
    return ["col1", "col2", 3]


@pytest.fixture
def string_instead_of_list():
    return "columns"


@pytest.fixture
def non_sequence():
    return 123


class TestValidationResult:
    def test_validation_result_with_valid_only(self):
        """Test creation of ValidationResult with only is_valid parameter"""
        result = ValidationResult(is_valid=True)
        assert result.is_valid is True
        assert result.message is None

    def test_validation_result_with_message(self):
        """Test creation of ValidationResult with both parameters"""
        result = ValidationResult(is_valid=False, message="Validation failed")
        assert result.is_valid is False
        assert result.message == "Validation failed"


class TestValidateData:
    def test_validate_data_with_valid_series(self, valid_series):
        """Test validation of a valid pandas Series"""
        result = IntervalValidator.validate_data(valid_series)
        assert result.is_valid is True
        assert result.message is None

    def test_validate_data_with_empty_series(self, empty_series):
        """Test validation fails with an empty Series"""
        with pytest.raises(EmptyIntervalError) as excinfo:
            IntervalValidator.validate_data(empty_series)
        assert str(excinfo.value) == "Data must not be empty"

    def test_validate_data_with_non_series(self):
        """Test validation fails with a non-Series object"""
        with pytest.raises(InvalidDataTypeError) as excinfo:
            IntervalValidator.validate_data([1, 2, 3])
        assert str(excinfo.value) == "Expected data to be a Pandas Series"


class TestValidateColumns:
    def test_validate_columns_with_none(self):
        """Test validation of None columns"""
        result = IntervalValidator._validate_columns(None, "test columns")
        assert result.is_valid is True
        assert result.message is None

    def test_validate_columns_with_valid_sequence(self, valid_column_list):
        """Test validation of a valid column sequence"""
        result = IntervalValidator._validate_columns(valid_column_list, "test columns")
        assert result.is_valid is True
        assert result.message is None

    def test_validate_columns_with_non_sequence(self, non_sequence):
        """Test validation fails with a non-sequence object"""
        with pytest.raises(InvalidMetricColumnError) as excinfo:
            IntervalValidator._validate_columns(non_sequence, "test columns")
        assert str(excinfo.value) == "test columns must be a sequence"

    def test_validate_columns_with_string(self, string_instead_of_list):
        """Test validation fails with a string (which is a sequence but not what we want)"""
        with pytest.raises(InvalidMetricColumnError) as excinfo:
            IntervalValidator._validate_columns(string_instead_of_list, "test columns")
        assert str(excinfo.value) == "test columns must be a sequence"

    def test_validate_columns_with_non_string_elements(self, non_string_column_list):
        """Test validation fails with non-string elements in the sequence"""
        with pytest.raises(InvalidMetricColumnError) as excinfo:
            IntervalValidator._validate_columns(non_string_column_list, "test columns")
        assert str(excinfo.value) == "All test columns must be of type str"

    def test_validate_columns_with_duplicate_columns(self, duplicate_column_list):
        """Test validation fails with duplicate column names"""
        with pytest.raises(InvalidMetricColumnError) as excinfo:
            IntervalValidator._validate_columns(duplicate_column_list, "test columns")
        assert str(excinfo.value) == "Duplicate test columns found"


class TestValidateSeriesIdColumns:
    def test_validate_series_id_columns_success(self, valid_column_list):
        """Test successful validation of series ID columns"""
        result = IntervalValidator.validate_series_id_columns(valid_column_list)
        assert result.is_valid is True
        assert result.message is None

    def test_validate_series_id_columns_with_duplicates(self, duplicate_column_list):
        """Test validation fails with duplicate series ID columns"""
        with pytest.raises(InvalidMetricColumnError) as excinfo:
            IntervalValidator.validate_series_id_columns(duplicate_column_list)
        assert str(excinfo.value) == "Duplicate series ID columns found"

    def test_validate_series_id_columns_with_none(self):
        """Test validation with None series ID columns"""
        result = IntervalValidator.validate_series_id_columns(None)
        assert result.is_valid is True
        assert result.message is None

    def test_validate_series_id_columns_with_non_string_elements(self, non_string_column_list):
        """Test validation fails with non-string elements in series ID columns"""
        with pytest.raises(InvalidMetricColumnError) as excinfo:
            IntervalValidator.validate_series_id_columns(non_string_column_list)
        assert str(excinfo.value) == "All series ID columns must be of type str"


class TestValidateMetricColumns:
    def test_validate_metric_columns_success(self, valid_column_list):
        """Test successful validation of metric columns"""
        result = IntervalValidator.validate_metric_columns(valid_column_list)
        assert result.is_valid is True
        assert result.message is None

    def test_validate_metric_columns_with_duplicates(self, duplicate_column_list):
        """Test validation fails with duplicate metric columns"""
        with pytest.raises(InvalidMetricColumnError) as excinfo:
            IntervalValidator.validate_metric_columns(duplicate_column_list)
        assert str(excinfo.value) == "Duplicate metric columns found"

    def test_validate_metric_columns_with_none(self):
        """Test validation with None metric columns"""
        result = IntervalValidator.validate_metric_columns(None)
        assert result.is_valid is True
        assert result.message is None

    def test_validate_metric_columns_with_non_string_elements(self, non_string_column_list):
        """Test validation fails with non-string elements in metric columns"""
        with pytest.raises(InvalidMetricColumnError) as excinfo:
            IntervalValidator.validate_metric_columns(non_string_column_list)
        assert str(excinfo.value) == "All metric columns must be of type str"
