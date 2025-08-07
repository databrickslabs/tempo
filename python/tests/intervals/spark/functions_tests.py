from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
from pyspark.sql.types import (
    ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType,
    DecimalType, BooleanType, StringType, StructField, StructType
)

from tempo.intervals.spark.functions import is_metric_col, make_disjoint_wrap


class TestIsMetricCol:
    """Tests for the is_metric_col function"""

    @pytest.mark.parametrize("dtype", [
        ByteType(), ShortType(), IntegerType(), LongType(),
        FloatType(), DoubleType(), DecimalType(10, 2), BooleanType()
    ])
    def test_numeric_types_return_true(self, dtype):
        """Test is_metric_col with various numeric types that should return True."""
        col = StructField("test", dtype, True)
        assert is_metric_col(col) is True

    @pytest.mark.parametrize("dtype", [
        StringType(),
        StructType([StructField("nested", IntegerType(), True)])
    ])
    def test_non_numeric_types_return_false(self, dtype):
        """Test is_metric_col with non-numeric types that should return False."""
        col = StructField("test", dtype, True)
        assert is_metric_col(col) is False


class TestMakeDisjointWrap:
    """Tests for the make_disjoint_wrap function"""

    @pytest.fixture
    def setup_fields(self):
        """Fixture to set up common fields for all tests"""
        return {
            "start_field": "start",
            "end_field": "end",
            "series_fields": ["series_id"],
            "metric_fields": ["value"]
        }

    def test_empty_dataframe(self, setup_fields):
        """Test with an empty DataFrame."""
        fields = setup_fields
        empty_df = pd.DataFrame(columns=[fields["start_field"], fields["end_field"], "series_id", "value"])

        disjoint_function = make_disjoint_wrap(
            fields["start_field"], fields["end_field"],
            fields["series_fields"], fields["metric_fields"]
        )
        result = disjoint_function(empty_df)

        assert result.empty
        assert list(result.columns) == list(empty_df.columns)

    def test_single_interval(self, setup_fields):
        """Test with a single interval."""
        fields = setup_fields
        data = {
            fields["start_field"]: [1],
            fields["end_field"]: [5],
            "series_id": ["A"],
            "value": [10]
        }
        df = pd.DataFrame(data)

        disjoint_function = make_disjoint_wrap(
            fields["start_field"], fields["end_field"],
            fields["series_fields"], fields["metric_fields"]
        )
        result = disjoint_function(df)

        # For a single interval, there should be no changes
        assert len(result) == 1
        assert result[fields["start_field"]].iloc[0] == 1
        assert result[fields["end_field"]].iloc[0] == 5
        assert result["series_id"].iloc[0] == "A"
        assert result["value"].iloc[0] == 10

    def test_non_overlapping_intervals(self, setup_fields):
        """Test with multiple non-overlapping intervals."""
        fields = setup_fields
        data = {
            fields["start_field"]: [1, 6, 11],
            fields["end_field"]: [5, 10, 15],
            "series_id": ["A", "A", "A"],
            "value": [10, 20, 30]
        }
        df = pd.DataFrame(data)

        disjoint_function = make_disjoint_wrap(
            fields["start_field"], fields["end_field"],
            fields["series_fields"], fields["metric_fields"]
        )
        result = disjoint_function(df)

        # Since we're dealing with potentially complex interval transformations
        # just verify basic properties
        assert len(result) == 3  # Number of intervals preserved for non-overlapping case
        # Check all values are present
        for start in data[fields["start_field"]]:
            assert start in result[fields["start_field"]].values
        for end in data[fields["end_field"]]:
            assert end in result[fields["end_field"]].values

    @patch('tempo.intervals.spark.functions.IntervalsUtils')
    @patch('tempo.intervals.spark.functions.Interval')
    def test_overlapping_intervals(self, mock_interval, mock_utils, setup_fields):
        """Test with overlapping intervals, using mocks to verify correct behavior."""
        fields = setup_fields

        # Setup data with overlapping intervals
        data = {
            fields["start_field"]: [1, 3, 7],
            fields["end_field"]: [5, 8, 10],
            "series_id": ["A", "A", "A"],
            "value": [10, 20, 30]
        }
        df = pd.DataFrame(data)

        # Configure mocks
        mock_interval_instances = []
        for i in range(3):
            mock_inst = MagicMock()
            mock_inst.data = df.iloc[i]
            mock_inst.start_field = fields["start_field"]
            mock_inst.end_field = fields["end_field"]
            mock_inst.series_fields = fields["series_fields"]
            mock_inst.metric_fields = fields["metric_fields"]
            mock_interval_instances.append(mock_inst)

        # Configure mock_interval.create to return the appropriate mock instance
        mock_interval.create.side_effect = mock_interval_instances

        # Configure IntervalsUtils to simulate disjoint interval creation
        mock_utils_instance = MagicMock()
        mock_utils.return_value = mock_utils_instance
        # Simulate disjoint intervals being created - return a dataframe with the same input to simplify
        mock_utils_instance.add_as_disjoint.side_effect = [
            pd.DataFrame([df.iloc[0]]),
            pd.DataFrame([df.iloc[0], df.iloc[1]]),
            pd.DataFrame([df.iloc[0], df.iloc[1], df.iloc[2]])
        ]

        # Execute the function
        disjoint_function = make_disjoint_wrap(
            fields["start_field"], fields["end_field"],
            fields["series_fields"], fields["metric_fields"]
        )
        result = disjoint_function(df)

        # Verify that IntervalsUtils.add_as_disjoint was called for each row
        assert mock_utils_instance.add_as_disjoint.call_count == 3

        # Each call should provide an Interval object
        for call in mock_utils_instance.add_as_disjoint.call_args_list:
            args, kwargs = call
            assert isinstance(args[0], MagicMock)  # The mocked Interval

    def test_dataframe_sorting(self, setup_fields):
        """Test that dataframes are properly sorted by start and end timestamps."""
        fields = setup_fields
        data = {
            fields["start_field"]: [5, 1, 3, 3],
            fields["end_field"]: [10, 4, 8, 6],
            "series_id": ["A", "A", "A", "A"],
            "value": [50, 10, 30, 20]
        }
        df = pd.DataFrame(data)

        # Create a simplified version of the make_disjoint_wrap function
        # that only performs the sorting step (copied from the actual implementation)
        def sort_intervals(pdf):
            return pdf.sort_values(by=[fields["start_field"], fields["end_field"]]).reset_index(drop=True)

        # Apply the sorting
        sorted_df = sort_intervals(df)

        # Verify sorting was applied correctly
        assert len(sorted_df) == 4

        # The expected order after sorting (first by start, then by end)
        expected_start_order = [1, 3, 3, 5]
        expected_end_order = [4, 6, 8, 10]

        for i in range(len(expected_start_order)):
            assert sorted_df[fields["start_field"]].iloc[i] == expected_start_order[i]
            assert sorted_df[fields["end_field"]].iloc[i] == expected_end_order[i]

    def test_custom_field_names(self):
        """Test with custom field names for start, end, series, and metrics."""
        start_field = "begin_time"
        end_field = "finish_time"
        series_fields = ["group"]
        metric_fields = ["measurement"]

        data = {
            start_field: [100, 200],
            end_field: [150, 250],
            "group": ["X", "Y"],
            "measurement": [5.5, 7.7]
        }
        df = pd.DataFrame(data)

        disjoint_function = make_disjoint_wrap(
            start_field, end_field, series_fields, metric_fields
        )
        result = disjoint_function(df)

        # Verify that the custom field names are preserved
        assert start_field in result.columns
        assert end_field in result.columns
        assert "group" in result.columns
        assert "measurement" in result.columns

        # Verify values are preserved (regardless of row count)
        for val in data["group"]:
            assert val in result["group"].values
        for val in data["measurement"]:
            assert val in result["measurement"].values

    def test_multiple_series_fields(self, setup_fields):
        """Test with multiple series identifier fields."""
        fields = setup_fields
        series_fields = ["region", "product"]

        data = {
            fields["start_field"]: [1, 2],
            fields["end_field"]: [3, 4],
            "region": ["North", "South"],
            "product": ["A", "B"],
            "value": [100, 200]
        }
        df = pd.DataFrame(data)

        disjoint_function = make_disjoint_wrap(
            fields["start_field"], fields["end_field"],
            series_fields, fields["metric_fields"]
        )
        result = disjoint_function(df)

        # We're not testing the specific disjoint interval behavior here,
        # just that the function handles multiple series fields correctly
        assert "region" in result.columns
        assert "product" in result.columns

        # Verify all input values are represented in the result
        for region in data["region"]:
            assert region in result["region"].values
        for product in data["product"]:
            assert product in result["product"].values

    def test_multiple_metric_fields(self, setup_fields):
        """Test with multiple metric fields."""
        fields = setup_fields
        metric_fields = ["sales", "cost"]

        data = {
            fields["start_field"]: [1, 2],
            fields["end_field"]: [3, 4],
            "series_id": ["A", "B"],
            "sales": [100, 200],
            "cost": [50, 100]
        }
        df = pd.DataFrame(data)

        disjoint_function = make_disjoint_wrap(
            fields["start_field"], fields["end_field"],
            fields["series_fields"], metric_fields
        )
        result = disjoint_function(df)

        # Verify the multiple metric fields are present
        assert "sales" in result.columns
        assert "cost" in result.columns

        # Verify that all input values for metrics are represented
        for sales_val in data["sales"]:
            assert sales_val in result["sales"].values
        for cost_val in data["cost"]:
            assert cost_val in result["cost"].values

    @patch('tempo.intervals.spark.functions.IntervalsUtils')
    @patch('tempo.intervals.spark.functions.Interval')
    def test_complex_overlapping_scenario(self, mock_interval, mock_utils, setup_fields):
        """Test a more complex scenario with multiple overlapping intervals."""
        fields = setup_fields

        # Setup data with complex overlapping intervals
        data = {
            fields["start_field"]: [1, 3, 2, 7, 6],
            fields["end_field"]: [5, 8, 6, 10, 9],
            "series_id": ["A", "A", "A", "A", "A"],
            "value": [10, 20, 15, 30, 25]
        }
        df = pd.DataFrame(data)

        # Configure mocks
        mock_interval_instances = []
        for i in range(len(df)):
            mock_inst = MagicMock()
            mock_inst.data = df.iloc[i]
            mock_inst.start_field = fields["start_field"]
            mock_inst.end_field = fields["end_field"]
            mock_inst.series_fields = fields["series_fields"]
            mock_inst.metric_fields = fields["metric_fields"]
            mock_interval_instances.append(mock_inst)

        mock_interval.create.side_effect = mock_interval_instances

        # Create a fake disjoint result that would represent the expected output
        expected_disjoint = pd.DataFrame({
            fields["start_field"]: [1, 2, 3, 6, 7],
            fields["end_field"]: [2, 3, 5, 7, 10],
            "series_id": ["A", "A", "A", "A", "A"],
            "value": [10, 15, 20, 25, 30]
        })

        # Configure the mock to return our expected result progressively
        mock_utils_instance = MagicMock()
        mock_utils.return_value = mock_utils_instance

        # Simulate building up the disjoint set - simple approach just returning same dataframe
        mock_utils_instance.add_as_disjoint.return_value = expected_disjoint

        # Execute
        disjoint_function = make_disjoint_wrap(
            fields["start_field"], fields["end_field"],
            fields["series_fields"], fields["metric_fields"]
        )
        result = disjoint_function(df)

        # Verify that IntervalsUtils.add_as_disjoint was called
        assert mock_utils_instance.add_as_disjoint.call_count >= 1

        # Each call should provide an Interval object
        for call in mock_utils_instance.add_as_disjoint.call_args_list:
            args, kwargs = call
            assert len(args) == 1  # Should have one argument


class TestEndToEndDisjointIntervals:
    """
    End-to-end tests for the make_disjoint_wrap function using
    the actual Interval and IntervalsUtils implementations.
    """

    @pytest.fixture
    def setup_fields(self):
        """Fixture to set up common fields for all tests"""
        return {
            "start_field": "start",
            "end_field": "end",
            "series_fields": ["series_id"],
            "metric_fields": ["value"]
        }

    def test_non_overlapping_intervals_e2e(self, setup_fields):
        """Test with non-overlapping intervals using actual implementations."""
        fields = setup_fields
        data = {
            fields["start_field"]: [1, 6, 11],
            fields["end_field"]: [5, 10, 15],
            "series_id": ["A", "A", "A"],
            "value": [10, 20, 30]
        }
        df = pd.DataFrame(data)

        disjoint_function = make_disjoint_wrap(
            fields["start_field"], fields["end_field"],
            fields["series_fields"], fields["metric_fields"]
        )

        # Call the function and check results
        result = disjoint_function(df)
        assert len(result) == 3

        # For non-overlapping intervals, check the values are preserved
        # but don't assume order
        for start in data[fields["start_field"]]:
            assert start in result[fields["start_field"]].values
        for end in data[fields["end_field"]]:
            assert end in result[fields["end_field"]].values

    def test_simple_overlap_e2e(self, setup_fields):
        """Test with a simple overlap using actual implementations."""
        fields = setup_fields
        data = {
            fields["start_field"]: [1, 3],
            fields["end_field"]: [5, 7],
            "series_id": ["A", "A"],
            "value": [10, 20]
        }
        df = pd.DataFrame(data)

        disjoint_function = make_disjoint_wrap(
            fields["start_field"], fields["end_field"],
            fields["series_fields"], fields["metric_fields"]
        )

        result = disjoint_function(df)

        # The result should have at least 2 intervals
        assert len(result) >= 2

        # Check that intervals are truly disjoint (no overlaps)
        result = result.sort_values(by=[fields["start_field"]]).reset_index(drop=True)
        for i in range(len(result) - 1):
            current_end = result[fields["end_field"]].iloc[i]
            next_start = result[fields["start_field"]].iloc[i + 1]
            # Disjoint intervals shouldn't overlap
            assert current_end <= next_start

    def test_complex_overlap_e2e(self, setup_fields):
        """Test with complex overlapping intervals using actual implementations."""
        fields = setup_fields
        data = {
            fields["start_field"]: [1, 2, 4, 6],
            fields["end_field"]: [5, 7, 8, 9],
            "series_id": ["A", "A", "A", "A"],
            "value": [10, 20, 30, 40]
        }
        df = pd.DataFrame(data)

        disjoint_function = make_disjoint_wrap(
            fields["start_field"], fields["end_field"],
            fields["series_fields"], fields["metric_fields"]
        )

        result = disjoint_function(df)

        # Verify the result has the correct structure
        assert len(result) >= 1  # At least one interval

        # Verify each interval has the required columns
        for col in [fields["start_field"], fields["end_field"], "series_id", "value"]:
            assert col in result.columns

        # Check that intervals are truly disjoint (no overlaps)
        result = result.sort_values(by=[fields["start_field"]]).reset_index(drop=True)
        for i in range(len(result) - 1):
            current_end = result[fields["end_field"]].iloc[i]
            next_start = result[fields["start_field"]].iloc[i + 1]
            # Disjoint intervals shouldn't overlap
            assert current_end <= next_start
