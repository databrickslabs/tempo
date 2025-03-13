import re

import pytest
from pandas import Series

from tempo.intervals.core.boundaries import _BoundaryAccessor
from tempo.intervals.core.exceptions import InvalidDataTypeError, EmptyIntervalError, InvalidMetricColumnError, \
    InvalidSeriesColumnError
from tempo.intervals.core.interval import Interval


class TestInterval:
    def test_create_interval(self):
        data = Series(
            {"start": "2023-01-01", "end": "2023-01-02", "series1": "A", "series2": "B", "metric1": 10, "metric2": 15})
        interval = Interval.create(data, start_field="start", end_field="end", series_fields=["series1", "series2"],
                                   metric_fields=["metric1", "metric2"])

        assert interval.start == "2023-01-01"
        assert interval.start_field == "start"
        assert interval.end == "2023-01-02"
        assert interval.end_field == "end"
        assert interval.boundaries == ("2023-01-01", "2023-01-02")
        assert interval.series_fields == ["series1", "series2"]
        assert interval.metric_fields == ["metric1", "metric2"]

    def test_create_interval_none_start_field(self):
        data = Series(
            {"start": None, "end": "2023-01-02", "series1": "A", "series2": "B", "metric1": 10, "metric2": 15})
        interval = Interval.create(data, start_field="start", end_field="end", series_fields=["series1", "series2"],
                                   metric_fields=["metric1", "metric2"])

        assert interval.start is None
        assert interval.start_field == "start"
        assert interval.end == "2023-01-02"
        assert interval.end_field == "end"
        assert interval.boundaries == (None, "2023-01-02")
        assert interval.series_fields == ["series1", "series2"]
        assert interval.metric_fields == ["metric1", "metric2"]

    def test_create_interval_none_end_field(self):
        data = Series(
            {"start": "2023-01-01", "end": None, "series1": "A", "series2": "B", "metric1": 10, "metric2": 15})
        interval = Interval.create(data, start_field="start", end_field="end", series_fields=["series1", "series2"],
                                   metric_fields=["metric1", "metric2"])

        assert interval.start == "2023-01-01"
        assert interval.start_field == "start"
        assert interval.end is None
        assert interval.end_field == "end"
        assert interval.boundaries == ("2023-01-01", None)
        assert interval.series_fields == ["series1", "series2"]
        assert interval.metric_fields == ["metric1", "metric2"]

    def test_create_interval_series_fields_not_string(self):
        data = Series({"start": None, "end": "2023-01-02", 1: "A", "metric1": 10, "metric2": 15})

        with pytest.raises(InvalidSeriesColumnError, match="All series_fields must be strings"):
            Interval.create(data, start_field="start", end_field="end", series_fields=[1],
                            metric_fields=["metric1", "metric2"])

    def test_create_interval_series_fields_not_sequence(self):
        data = Series({"start": "2023-01-01", "end": "2023-01-05", "series-id1": "ABC"})
        with pytest.raises(InvalidSeriesColumnError, match=r"series_fields must be a sequence"):
            Interval.create(data, start_field="start", end_field="end", series_fields="series-id1")

    def test_create_interval_metrics_fields_not_string(self):
        data = Series({"start": None, "end": "2023-01-02", "series": "A", 1: 10, "metric": 15})

        with pytest.raises(InvalidMetricColumnError, match="All metric_fields must be strings"):
            Interval.create(data, start_field="start", end_field="end", series_fields=["series"],
                            metric_fields=[1, "metric"])

    def test_create_interval_metrics_fields_not_sequence(self):
        data = Series({"start": None, "end": "2023-01-02", "series": "A", 1: 10, "metric2": 15})

        with pytest.raises(InvalidMetricColumnError, match="metric_fields must be a sequence"):
            Interval.create(data, start_field="start", end_field="end", series_fields=["series"],
                            metric_fields="metric")

    def test_invalid_data_type(self):
        data = {"start": "2023-01-01", "end": "2023-01-02"}  # Not a pandas Series

        with pytest.raises(InvalidDataTypeError, match="Data must be a pandas Series"):
            Interval.create(data, start_field="start", end_field="end")

    def test_empty_data(self):
        data = Series(dtype="object")  # Empty pandas Series
        with pytest.raises(EmptyIntervalError, match=r"Data cannot be empty"):
            Interval.create(data, start_field="start", end_field="end")

    def test_update_start(self):
        data = Series({"start": "2023-01-01", "end": "2023-01-02", "metric": 10})
        interval = Interval.create(data, start_field="start", end_field="end")
        updated_interval = interval.update_start("2023-01-03")

        assert updated_interval.start == "2023-01-03"
        assert interval.end == updated_interval.end  # End remains the same
        assert interval._end == updated_interval._end  # End remains the same

    def test_update_end(self):
        data = Series({"start": "2023-01-01", "end": "2023-01-02", "metric": 10})
        interval = Interval.create(data, start_field="start", end_field="end")
        updated_interval = interval.update_end("2023-01-03")

        assert updated_interval.end == "2023-01-03"
        assert interval.start == updated_interval.start  # Start remains the same
        assert interval._start == updated_interval._start  # Start remains the same

    def test_contains(self):
        data1 = Series({"start": "2023-01-01", "end": "2023-01-05"})
        interval1 = Interval.create(data1, start_field="start", end_field="end")

        data2 = Series({"start": "2023-01-02", "end": "2023-01-04"})
        interval2 = Interval.create(data2, start_field="start", end_field="end")

        assert interval1.contains(interval2)
        assert not interval2.contains(interval1)

    def test_overlaps_with(self):
        data1 = Series({"start": "2023-01-01", "end": "2023-01-05"})
        interval1 = Interval.create(data1, start_field="start", end_field="end")

        data2 = Series({"start": "2023-01-04", "end": "2023-01-06"})
        interval2 = Interval.create(data2, start_field="start", end_field="end")

        assert interval1.overlaps_with(interval2)
        assert interval2.overlaps_with(interval1)

    def test_validate_metric_alignment(self):
        data1 = Series(
            {"start": "2023-01-01", "end": "2023-01-05", "series1": "A", "series2": "B", "metric1": 10, "metric2": 15})
        interval1 = Interval.create(data1, start_field="start", end_field="end", metric_fields=["metric1", "metric2"])

        data2 = Series({"start": "2023-01-02", "end": "2023-01-04", "metric1": 5, "metric2": 10})
        interval2 = Interval.create(data2, start_field="start", end_field="end", metric_fields=["metric1", "metric2"])

        result = interval1.validate_metrics_alignment(interval2)
        assert result.is_valid

    def test_validate_metric_alignment_invalid(self):
        data1 = Series(
            {"start": "2023-01-01", "end": "2023-01-05", "series1": "A", "series2": "B", "metric1": 10, "metric2": 15})
        interval1 = Interval.create(data1, start_field="start", end_field="end", metric_fields=["metric1", "metric2"])

        data2 = Series({"start": "2023-01-02", "end": "2023-01-04", "metric1": 5, "metric2": 10})
        interval2 = Interval.create(data2, start_field="start", end_field="end", metric_fields=["metric1"])

        expected_msg = re.escape("metric_fields don't match: ['metric1', 'metric2'] vs ['metric1']")
        with pytest.raises(InvalidMetricColumnError, match=expected_msg):
            interval1.validate_metrics_alignment(interval2)

    def test_validate_series_alignment(self):
        data1 = Series(
            {"start": "2023-01-01", "end": "2023-01-05", "series1": "A", "series2": "B", "metric1": 10, "metric2": 15})
        interval1 = Interval.create(data1, start_field="start", end_field="end", series_fields=["series1", "series2"],
                                    metric_fields=["metric1", "metric2"])

        data2 = Series({"start": "2023-01-02", "end": "2023-01-04", "metric1": 5, "metric2": 10})
        interval2 = Interval.create(data2, start_field="start", end_field="end", series_fields=["series1", "series2"],
                                    metric_fields=["metric1", "metric2"])

        result = interval1.validate_series_alignment(interval2)
        assert result.is_valid

    def test_validate_series_alignment_invalid(self):
        data1 = Series(
            {"start": "2023-01-01", "end": "2023-01-05", "series1": "A", "series2": "B", "metric1": 10, "metric2": 15})
        interval1 = Interval.create(data1, start_field="start", end_field="end", series_fields=["series1"],
                                    metric_fields=["metric1", "metric2"])

        data2 = Series({"start": "2023-01-02", "end": "2023-01-04", "metric1": 5, "metric2": 10})
        interval2 = Interval.create(data2, start_field="start", end_field="end", series_fields=["series1", "series2"],
                                    metric_fields=["metric1", "metric2"])

        expected_msg = re.escape("series_fields don't match: ['series1'] vs ['series1', 'series2']")
        with pytest.raises(InvalidSeriesColumnError, match=expected_msg):
            interval1.validate_series_alignment(interval2)

    def test_validate_not_point_in_time_valid_interval(self):
        data = Series({"start_time": 1, "end_time": 2})
        boundary_accessor = _BoundaryAccessor("start_time", "end_time")

        Interval._validate_not_point_in_time(data, boundary_accessor)

    def test_validate_not_point_in_time_point_in_time_interval(self):
        data = Series({"start_time": 1, "end_time": 1})
        boundary_accessor = _BoundaryAccessor("start_time", "end_time")

        with pytest.raises(InvalidDataTypeError, match="Point-in-Time Intervals are not supported"):
            Interval._validate_not_point_in_time(data, boundary_accessor)

    def test_validate_not_point_in_time_missing_boundary_fields(self):
        data = Series({"start_time": 1})
        boundary_accessor = _BoundaryAccessor("start_time", "end_time")

        with pytest.raises(KeyError):
            Interval._validate_not_point_in_time(data, boundary_accessor)


class TestStillValidLegacy:

    def test_update_interval_boundary_start(self):
        interval = Interval.create(Series(
            {"start": "2023-01-01T01:00:00", "end": "2023-01-01T02:00:00"}
        ), "start", "end")

        updated = interval.update_start("2023-01-01T01:30:00")
        assert updated.data["start"] == "2023-01-01T01:30:00"

    def test_update_interval_boundary_end(self):
        interval = Interval.create(Series(
            {"start": "2023-01-01T01:00:00", "end": "2023-01-01T02:00:00"}
        ), "start", "end")

        updated = interval.update_end("2023-01-01T02:30:00")
        assert updated.data["end"] == "2023-01-01T02:30:00"

    def test_update_interval_boundary_return_new_copy(self):
        interval = Interval.create(Series(
            {"start": "2023-01-01T01:00:00", "end": "2023-01-01T02:00:00"}
        ), "start", "end")

        updated = interval.update_start("2023-01-01T01:30:00")
        assert id(interval) != id(updated)
        assert interval.data["start"] == "2023-01-01T01:00:00"

    def test_merge_metrics_with_list_metric_merge_true(self):
        interval = Interval.create(Series({"start": "01:00", "end": "02:00", "value": 10}), "start", "end",
                                   metric_fields=["value"])
        other = Interval.create(Series({"start": "01:00", "end": "02:00", "value": 20}), "start", "end",
                                metric_fields=["value"])
        expected = Series({"start": "01:00", "end": "02:00", "value": 20})

        merged = interval.merge_metrics(other)
        assert merged.equals(expected)

    def test_merge_metrics_with_string_metric_column(self):
        interval = Interval.create(Series({"start": "01:00", "end": "02:00", "value": 10}), "start", "end",
                                   metric_fields=["value"])
        other = Interval.create(Series({"start": "01:00", "end": "02:00", "value": 20}), "start", "end",
                                metric_fields=["value"])
        expected = Series({"start": "01:00", "end": "02:00", "value": 20})

        merged = interval.merge_metrics(other)
        assert merged.equals(expected)

    def test_merge_metrics_with_string_metric_columns(self):
        interval = Interval.create(Series({"start": "01:00", "end": "02:00", "value1": 10, "value2": 20}), "start",
                                   "end",
                                   metric_fields=["value1", "value2"])
        other = Interval.create(Series(
            {"start": "01:00", "end": "02:00", "value1": 20, "value2": 30}
        ), "start", "end", metric_fields=["value1", "value2"])
        expected = Series(
            {"start": "01:00", "end": "02:00", "value1": 20, "value2": 30}
        )

        merged = interval.merge_metrics(other)
        assert merged.equals(expected)

    def test_merge_metrics_return_new_copy(self):
        interval = Interval.create(
            Series({"start": "01:00", "end": "02:00", "value": 10}),
            "start",
            "end", [], ["value"])
        other = Interval.create(
            Series({"start": "01:00", "end": "02:00", "value": 20}),
            "start",
            "end", [], ["value"])

        merged = interval.merge_metrics(other)
        assert id(interval) != id(merged)

    def test_merge_metrics_handle_nan_in_child(self):
        interval = Interval.create(
            Series({"start": "01:00", "end": "02:00", "value": 10}),
            "start",
            "end",
            [], ["value"]
        )
        other = Interval.create(
            Series({"start": "01:00", "end": "02:00", "value": float("nan")}),
            "start",
            "end",
            [], ["value"]
        )

        merged = interval.merge_metrics(other)
        assert merged["value"] == 10
